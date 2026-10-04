"""
Autonomous closed-loop bioreactor controller (software prototype)
===================================================================

What this is
------------
A working, self-contained version of the architecture in micro_org.txt:

  sensor telemetry -> safety interlock -> contamination detector
                   -> digital-twin phase logic -> MPC-lite micro-corrections
                   -> actuator commands (MQTT, or logged if no broker)

What is real here vs. simplified
--------------------------------
* Digital twin  : Monod kinetics solved with SciPy (no LSTM residual model yet).
* Anomaly model : scikit-learn IsolationForest, trained at startup on
                  SIMULATED healthy batches (replace with your real batch data).
* Control       : one-step-lookahead optimiser ("MPC-lite"), not full MPC/RL.
* Hardware      : none. Commands go to MQTT if a broker is reachable,
                  otherwise they are kept in an in-memory log.
* SAFETY        : the software hard limits are NOT a substitute for a physical
                  kill switch / thermal fuse on real hardware.

Install
-------
    pip install numpy scipy scikit-learn fastapi uvicorn paho-mqtt
(fastapi/uvicorn/paho-mqtt are optional: --demo works with only the first three)

Run
---
    python bioreactor_controller.py --demo          # no server needed
    python bioreactor_controller.py                 # dashboard and API on http://127.0.0.1:8000
    python bioreactor_controller.py --serve         # same as above, explicitly
    # or: uvicorn bioreactor_controller:app --reload

Example request (once serving)
------------------------------
    curl -X POST http://127.0.0.1:8000/bioreactor/telemetry-loop \
      -H "Content-Type: application/json" \
      -d '{"batch_id":"B1","temperature":30.1,"ph":6.98,"dissolved_oxygen":90,
           "co2_level":0.6,"optical_density":0.4,"elapsed_h":1.0}'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_DT_H = 0.25            # assumed sampling interval if elapsed_h not sent
TARGET_OD = 2.5                # biomass proxy at which growth -> production
PH_SETPOINT = 7.0
PH_DEADBAND = 0.10
PH_GAIN_PER_ML_BASE = 0.08     # assumed pH rise per mL of base (tune per vessel)
MAX_BASE_DOSE_ML = 2.0
MAX_NUTRIENT_PULSE_ML = 2.0
DRIFT_RATIO_THRESHOLD = 0.85   # observed OD / twin-expected OD
ANOMALY_WINDOW = 5             # look at the last N readings ...
ANOMALY_VOTES = 3              # ... and quarantine if >= this many are anomalous

# Independent hard limits (checked before any AI logic)
HARD_LIMITS = {
    "temperature": (20.0, 40.0),   # deg C
    "ph": (4.5, 8.5),
    "dissolved_oxygen": (3.0, 100.0),  # % saturation
}

SETPOINTS = {
    "GROWTH":     {"agitation_rpm": 400, "pump_feed_rate_ml_min": 2.0},
    "PRODUCTION": {"agitation_rpm": 250, "pump_feed_rate_ml_min": 1.2},
}

DEFAULT_KINETICS = dict(mu_max=0.35, Ks=0.5, Yxs=0.5, S0=20.0, X0=0.1,
                        alpha=0.2, beta=0.01)


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------
@dataclass
class Reading:
    batch_id: str
    temperature: float
    ph: float
    dissolved_oxygen: float
    co2_level: float
    optical_density: float
    elapsed_h: Optional[float] = None


@dataclass
class BatchState:
    n: int = 0
    phase: str = "GROWTH"
    quarantined: bool = False
    quarantine_reason: str = ""
    above_target: int = 0
    prev: Optional[tuple] = None          # (t, ph, do) of previous reading
    flags: Deque[int] = field(default_factory=lambda: deque(maxlen=ANOMALY_WINDOW))
    last_action: str = "NONE"
    last_reading: Optional[dict] = None
    last_expected_od: Optional[float] = None


# --------------------------------------------------------------------------
# Simulator (Monod growth + Luedeking-Piret product formation)
# --------------------------------------------------------------------------
def _kinetics_ode(_t, y, p):
    X, S, P = y
    S = max(S, 0.0)
    mu = p["mu_max"] * S / (p["Ks"] + S)
    return [mu * X, -mu * X / p["Yxs"], p["alpha"] * mu * X + p["beta"] * X]


def simulate_batch(rng: np.random.Generator, hours: float = 24.0,
                   dt: float = DEFAULT_DT_H, contaminated_at: Optional[float] = None,
                   jitter: float = 0.05) -> List[Reading]:
    """Generate one batch of noisy sensor readings. Optionally inject a
    contamination event (faster pH/DO drop, CO2 rise, erratic OD) at a given hour."""
    p = {k: v * (1 + rng.uniform(-jitter, jitter)) for k, v in DEFAULT_KINETICS.items()}
    t_eval = np.arange(0.0, hours + 1e-9, dt)
    sol = solve_ivp(_kinetics_ode, (0, hours), [p["X0"], p["S0"], 0.0],
                    t_eval=t_eval, args=(p,), method="LSODA")
    X, S, _P = sol.y
    mu = p["mu_max"] * np.maximum(S, 0) / (p["Ks"] + np.maximum(S, 0))

    out = []
    for i, t in enumerate(t_eval):
        od = 0.8 * X[i] + rng.normal(0, 0.03)
        ph = 7.0 - 0.008 * X[i] + rng.normal(0, 0.02)
        do = max(5.0, 95.0 - 4.5 * X[i]) + rng.normal(0, 0.8)
        co2 = 0.4 + 0.5 * mu[i] * X[i] + rng.normal(0, 0.03)
        temp = 30.0 + rng.normal(0, 0.15)
        if contaminated_at is not None and t > contaminated_at:
            c = t - contaminated_at
            ph -= 0.10 * c
            do -= 3.0 * c
            co2 += 0.25 * c
            od += rng.normal(0, 0.05 * min(c, 4.0))
        out.append(Reading("SIM", float(temp), float(ph), float(do),
                           float(co2), float(max(od, 0.0)), float(t)))
    return out


# --------------------------------------------------------------------------
# Digital twin (kinetic model; extension point for an LSTM residual model)
# --------------------------------------------------------------------------
class KineticTwin:
    def __init__(self, params: Optional[dict] = None, horizon_h: float = 72.0):
        p = dict(DEFAULT_KINETICS, **(params or {}))
        self._t = np.arange(0.0, horizon_h, 0.05)
        sol = solve_ivp(_kinetics_ode, (0, horizon_h), [p["X0"], p["S0"], 0.0],
                        t_eval=self._t, args=(p,), method="LSODA")
        self._od = 0.8 * sol.y[0]

    def expected_od(self, t_h: float) -> float:
        return float(np.interp(t_h, self._t, self._od))


# --------------------------------------------------------------------------
# Contamination / anomaly detector
# --------------------------------------------------------------------------
def make_features(r: Reading, prev: Optional[tuple], t: float) -> List[float]:
    if prev is None:
        d_ph = d_do = 0.0
    else:
        dt = max(t - prev[0], 1e-6)
        d_ph, d_do = (r.ph - prev[1]) / dt, (r.dissolved_oxygen - prev[2]) / dt
    return [r.temperature, r.ph, r.dissolved_oxygen, r.co2_level,
            r.optical_density, d_ph, d_do]


class AnomalyDetector:
    def __init__(self, model: IsolationForest):
        self.model = model

    @classmethod
    def train(cls, seed: int = 0, n_batches: int = 60) -> "AnomalyDetector":
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_batches):
            prev = None
            for r in simulate_batch(rng):
                rows.append(make_features(r, prev, r.elapsed_h))
                prev = (r.elapsed_h, r.ph, r.dissolved_oxygen)
        model = IsolationForest(n_estimators=200, contamination=0.005,
                                random_state=seed).fit(np.array(rows))
        return cls(model)

    def is_anomalous(self, features: List[float]) -> bool:
        return int(self.model.predict(np.array([features]))[0]) == -1


# --------------------------------------------------------------------------
# MPC-lite: one-step lookahead dosing
# --------------------------------------------------------------------------
def mpc_base_dose(ph: float) -> float:
    """Pick the base dose (mL) that minimises predicted pH error plus a small
    dose penalty. Only acts below the deadband."""
    if ph >= PH_SETPOINT - PH_DEADBAND:
        return 0.0
    doses = np.linspace(0.0, MAX_BASE_DOSE_ML, 21)
    cost = (ph + PH_GAIN_PER_ML_BASE * doses - PH_SETPOINT) ** 2 + 0.002 * doses ** 2
    return float(round(doses[int(np.argmin(cost))], 3))


def nutrient_pulse(observed_od: float, expected_od: float) -> float:
    """If biomass lags the twin's prediction, pulse substrate proportionally."""
    if expected_od <= 0.05:
        return 0.0
    ratio = observed_od / expected_od
    if ratio >= DRIFT_RATIO_THRESHOLD:
        return 0.0
    shortfall = DRIFT_RATIO_THRESHOLD - ratio
    return float(round(min(MAX_NUTRIENT_PULSE_ML, 4.0 * shortfall), 3))


# --------------------------------------------------------------------------
# Command dispatch (MQTT is opt-in, commands are always logged)
# --------------------------------------------------------------------------
class CommandDispatcher:
    def __init__(self):
        self.log: Deque[dict] = deque(maxlen=500)
        self.client = None
        # A publicly shared demo must never attempt to control hardware unless
        # an operator deliberately opts in through its environment.
        if os.getenv("MQTT_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
            print("[dispatcher] MQTT disabled; commands will be logged only.")
            return
        try:
            import paho.mqtt.client as mqtt
            try:  # paho-mqtt >= 2.0
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            except AttributeError:  # paho-mqtt 1.x
                self.client = mqtt.Client()
            self.client.connect(os.getenv("MQTT_BROKER", "localhost"),
                                int(os.getenv("MQTT_PORT", "1883")), 60)
            self.client.loop_start()
        except Exception as exc:  # paho missing or broker unreachable
            print(f"[dispatcher] MQTT unavailable ({exc.__class__.__name__}); "
                  "commands will be logged only.")
            self.client = None

    def send(self, batch_id: str, payload: dict) -> None:
        topic = f"bioreactor/{batch_id}/actuators"
        self.log.append({"topic": topic, "payload": payload})
        if self.client is not None:
            self.client.publish(topic, json.dumps(payload))


# --------------------------------------------------------------------------
# The controller
# --------------------------------------------------------------------------
class Controller:
    def __init__(self, dispatcher: Optional[CommandDispatcher] = None, seed: int = 0):
        self.detector = AnomalyDetector.train(seed)
        self.twin = KineticTwin()
        self.dispatcher = dispatcher or CommandDispatcher()
        self.batches: Dict[str, BatchState] = {}
        self.lock = threading.Lock()

    # -- helpers -----------------------------------------------------------
    def _command(self, batch_id: str, phase: str, **extra) -> dict:
        cmd = {"batch_id": batch_id, "phase": phase, "heater_state": 1,
               "cooling_jacket": 0, **SETPOINTS[phase],
               "base_pump_ml": 0.0, "nutrient_pulse_ml": 0.0,
               "status": "STABLE_AUTONOMOUS_RUN"}
        cmd.update(extra)
        return cmd

    def _quarantine(self, st: BatchState, batch_id: str, reason: str, alert: str) -> dict:
        st.quarantined, st.quarantine_reason = True, reason
        cmd = {"batch_id": batch_id, "phase": st.phase, "heater_state": 0,
               "cooling_jacket": 1, "pump_feed_rate_ml_min": 0.0,
               "agitation_rpm": 0, "base_pump_ml": 0.0, "nutrient_pulse_ml": 0.0,
               "status": alert}
        self.dispatcher.send(batch_id, cmd)
        st.last_action = "EMERGENCY_SHUTDOWN"
        return {"action": "EMERGENCY_SHUTDOWN", "reason": reason, "command": cmd}

    # -- main entry point ----------------------------------------------------
    def process(self, r: Reading) -> dict:
        with self.lock:
            st = self.batches.setdefault(r.batch_id, BatchState())
            st.n += 1
            t = r.elapsed_h if r.elapsed_h is not None else (st.n - 1) * DEFAULT_DT_H
            st.last_reading = r.__dict__.copy()

            if st.quarantined:
                return {"action": "QUARANTINE_ACTIVE", "reason": st.quarantine_reason,
                        "command": None}

            # 1) Safety interlock: hard limits, independent of the ML model
            for name, (lo, hi) in HARD_LIMITS.items():
                value = getattr(r, name)
                if not lo <= value <= hi:
                    return self._quarantine(
                        st, r.batch_id,
                        f"Hard limit violated: {name}={value} outside [{lo}, {hi}].",
                        "QUARANTINE_HARD_LIMIT")

            # 2) Contamination check (Isolation Forest + persistence vote)
            feats = make_features(r, st.prev, t)
            st.prev = (t, r.ph, r.dissolved_oxygen)
            st.flags.append(int(self.detector.is_anomalous(feats)))
            if sum(st.flags) >= ANOMALY_VOTES:
                return self._quarantine(
                    st, r.batch_id,
                    f"Anomalous metabolic fingerprint in {sum(st.flags)} of the "
                    f"last {len(st.flags)} readings.",
                    "QUARANTINE_CONTAMINATION_DETECTED")

            # 3) Growth -> production phase shift (latched, needs 2 consecutive hits)
            expected = self.twin.expected_od(t)
            st.last_expected_od = expected
            if st.phase == "GROWTH":
                st.above_target = st.above_target + 1 if r.optical_density >= TARGET_OD else 0
                if st.above_target >= 2:
                    st.phase = "PRODUCTION"
                    cmd = self._command(r.batch_id, "PRODUCTION",
                                        status="PHASE_SHIFT_TO_PRODUCTION")
                    self.dispatcher.send(r.batch_id, cmd)
                    st.last_action = "PHASE_SHIFT"
                    return {"action": "PHASE_SHIFT",
                            "reason": "Biomass target reached; lower agitation and "
                                      "production feed profile applied.",
                            "command": cmd}

            # 4) Micro-corrections: pH via MPC-lite, substrate drift via the twin
            base = mpc_base_dose(r.ph)
            pulse = nutrient_pulse(r.optical_density, expected) if st.phase == "GROWTH" else 0.0
            acted = base > 0 or pulse > 0
            cmd = self._command(r.batch_id, st.phase, base_pump_ml=base,
                                nutrient_pulse_ml=pulse,
                                status="MICRO_CORRECTION" if acted else "STABLE_AUTONOMOUS_RUN")
            self.dispatcher.send(r.batch_id, cmd)
            st.last_action = "MICRO_CORRECTION" if acted else "STABLE"
            return {"action": st.last_action,
                    "reason": "Applied small corrective doses." if acted else "Within bands.",
                    "command": cmd}

    def status(self, batch_id: str) -> Optional[dict]:
        with self.lock:
            st = self.batches.get(batch_id)
            if st is None:
                return None
            return {"batch_id": batch_id, "readings": st.n, "phase": st.phase,
                    "quarantined": st.quarantined, "quarantine_reason": st.quarantine_reason,
                    "last_action": st.last_action, "last_reading": st.last_reading,
                    "twin_expected_od": st.last_expected_od}

    def reset(self, batch_id: str) -> bool:
        """Operator acknowledgement: clear a quarantine / restart a batch record."""
        with self.lock:
            return self.batches.pop(batch_id, None) is not None


# --------------------------------------------------------------------------
# Optional FastAPI layer
# --------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field
    from fastapi.staticfiles import StaticFiles

    FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
    BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

    class BioreactorTelemetry(BaseModel):
        batch_id: str = Field(..., min_length=1, max_length=64)
        temperature: float = Field(..., ge=0, le=100, description="deg C")
        ph: float = Field(..., ge=0, le=14)
        dissolved_oxygen: float = Field(..., ge=0, le=100, description="% saturation")
        co2_level: float = Field(..., ge=0)
        optical_density: float = Field(..., ge=0, description="biomass proxy")
        elapsed_h: Optional[float] = Field(None, ge=0, description="hours since inoculation")

    app = FastAPI(title="Autonomous Bioreactor Controller")

    _controller: Optional[Controller] = None
    _controller_lock = threading.Lock()

    def get_controller() -> Controller:
        global _controller
        # Startup training is deliberately lazy so /health remains immediate.
        # The extra lock prevents simultaneous first requests making two controllers.
        with _controller_lock:
            if _controller is None:
                _controller = Controller()
        return _controller

    def validate_batch_id(batch_id: str) -> str:
        batch_id = batch_id.strip()
        if not BATCH_ID_PATTERN.fullmatch(batch_id):
            raise HTTPException(
                status_code=422,
                detail=("batch_id must be 1-64 characters of letters, numbers, "
                        "dots, hyphens, or underscores, beginning with a letter or number."),
            )
        return batch_id

    def validate_finite_telemetry(data: BioreactorTelemetry) -> None:
        values = [data.temperature, data.ph, data.dissolved_oxygen,
                  data.co2_level, data.optical_density]
        if data.elapsed_h is not None:
            values.append(data.elapsed_h)
        if not all(math.isfinite(value) for value in values):
            raise HTTPException(status_code=422, detail="Telemetry values must be finite numbers.")

    @app.get("/health")
    def health():
        return {"status": "ok", "mqtt_mode": "opt-in"}

    @app.post("/bioreactor/telemetry-loop")
    def telemetry_loop(data: BioreactorTelemetry):
        data.batch_id = validate_batch_id(data.batch_id)
        validate_finite_telemetry(data)
        payload = getattr(data, "model_dump", data.dict)()
        return get_controller().process(Reading(**payload))

    # Keep concrete routes before the parameterised batch route. This avoids
    # accidental route shadowing if more /bioreactor endpoints are added later.
    @app.get("/bioreactor/commands/recent")
    def recent_commands(limit: int = Query(20, ge=1, le=500)):
        return list(get_controller().dispatcher.log)[-limit:]

    @app.get("/bioreactor/{batch_id}/status")
    def batch_status(batch_id: str):
        batch_id = validate_batch_id(batch_id)
        s = get_controller().status(batch_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Unknown batch_id")
        return s

    @app.post("/bioreactor/{batch_id}/reset")
    def reset_batch(batch_id: str):
        batch_id = validate_batch_id(batch_id)
        if not get_controller().reset(batch_id):
            raise HTTPException(status_code=404, detail="Unknown batch_id")
        return {"status": "reset", "batch_id": batch_id}

    # This mount must be last: it is a catch-all for the SPA/static files and
    # would otherwise intercept API routes. The browser uses relative /bioreactor
    # URLs, so the UI and API share one origin through localhost or ngrok.
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")



except ImportError:
    app = None


def run_demo() -> None:
    """Run a short, logged-only simulated batch without starting an HTTP server."""
    controller = Controller()
    readings = simulate_batch(np.random.default_rng(7), hours=2.0)
    for reading in readings:
        result = controller.process(reading)
        print(f"t={reading.elapsed_h:4.2f}h action={result['action']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous bioreactor controller prototype")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="run a short simulated batch")
    mode.add_argument("--serve", action="store_true", help="serve the API and frontend")
    parser.add_argument("--host", default="127.0.0.1", help="bind host when using --serve")
    parser.add_argument("--port", default=8000, type=int, help="bind port when using --serve")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return
    if app is None:
        parser.error("Serving requires fastapi and uvicorn. Install: pip install fastapi uvicorn")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
