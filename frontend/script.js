/* The dashboard is served by FastAPI. Root-relative API paths therefore use
   the same localhost or ngrok origin that delivered this page. */
(() => {
    "use strict";

    const form = document.querySelector("#telemetry-form");
    const submitButton = document.querySelector("#submit-button");
    const message = document.querySelector("#request-message");
    const commandsPanel = document.querySelector("#commands-panel");
    const historyBody = document.querySelector("#history-body");
    const connection = document.querySelector("#connection-status");
    const connectionText = document.querySelector("#connection-text");
    const batchIdInput = document.querySelector("#batch_id");

    const input = (id) => document.querySelector(`#${id}`);

    function setConnection(connected, text) {
        connection?.classList.toggle("offline", !connected);
        if (connectionText) connectionText.textContent = text;
    }

    function setMessage(text, type = "") {
        if (!message) return;
        message.textContent = text;
        message.className = `request-message ${type}`.trim();
    }

    async function request(path, options = {}) {
        const { headers = {}, ...requestOptions } = options;
        const response = await fetch(path, {
            ...requestOptions,
            headers: { Accept: "application/json", ...headers },
        });
        const text = await response.text();
        let body = null;
        try {
            body = text ? JSON.parse(text) : null;
        } catch {
            // The API normally returns JSON; keep a useful error if a proxy does not.
        }
        if (!response.ok) {
            const detail = Array.isArray(body?.detail)
                ? body.detail.map((item) => item.msg).join("; ")
                : body?.detail || text || `Request failed (${response.status})`;
            throw new Error(detail);
        }
        return body;
    }

    function numericValue(id) {
        const value = Number(input(id)?.value);
        if (!Number.isFinite(value)) throw new Error(`Enter a valid value for ${id}.`);
        return value;
    }

    function telemetryPayload() {
        const batchId = batchIdInput?.value.trim() || "";
        if (!batchId) throw new Error("Enter a batch ID.");
        return {
            batch_id: batchId,
            temperature: numericValue("temperature"),
            ph: numericValue("ph"),
            dissolved_oxygen: numericValue("dissolved_oxygen"),
            co2_level: numericValue("co2_level"),
            optical_density: numericValue("optical_density"),
            elapsed_h: numericValue("elapsed_h"),
        };
    }

    function setMetric(id, value) {
        const element = input(id);
        if (element) element.textContent = String(value);
    }

    function updateDashboard(reading, result) {
        document.querySelector("#batch-display").textContent = reading.batch_id;
        setMetric("temperature-value", reading.temperature);
        setMetric("ph-value", reading.ph);
        setMetric("oxygen-value", reading.dissolved_oxygen);
        setMetric("co2-value", reading.co2_level);
        setMetric("od-value", reading.optical_density);

        const badge = document.querySelector("#status-badge");
        if (!badge) return;
        const action = result.action || "UNKNOWN";
        const isEmergency = action.includes("QUARANTINE") || action.includes("EMERGENCY");
        const needsAttention = action === "MICRO_CORRECTION" || action === "PHASE_SHIFT";
        badge.className = `status-badge ${isEmergency ? "error" : needsAttention ? "warning" : "stable"}`;
        badge.textContent = action.replaceAll("_", " ");
    }

    function commandField(label, value) {
        const item = document.createElement("div");
        item.className = "command-item";
        const title = document.createElement("span");
        title.textContent = label;
        const content = document.createElement("strong");
        content.textContent = String(value ?? "—");
        item.append(title, content);
        return item;
    }

    function renderCommand(command) {
        if (!commandsPanel || !command) return;
        commandsPanel.replaceChildren(
            commandField("Controller status", command.status),
            commandField("Phase", command.phase),
            commandField("Agitation", `${command.agitation_rpm ?? 0} rpm`),
            commandField("Feed rate", `${command.pump_feed_rate_ml_min ?? 0} mL/min`),
            commandField("Base dose", `${command.base_pump_ml ?? 0} mL`),
            commandField("Nutrient pulse", `${command.nutrient_pulse_ml ?? 0} mL`),
        );
    }

    function addHistoryRow(reading, result) {
        if (!historyBody) return;
        if (historyBody.querySelector(".empty-state")) historyBody.replaceChildren();
        const row = document.createElement("tr");
        [
            reading.batch_id,
            reading.temperature,
            reading.ph,
            reading.dissolved_oxygen,
            reading.co2_level,
            reading.optical_density,
            result.action || "—",
        ].forEach((value) => {
            const cell = document.createElement("td");
            cell.textContent = String(value);
            row.append(cell);
        });
        historyBody.prepend(row);
        while (historyBody.rows.length > 20) historyBody.deleteRow(-1);
    }

    async function submitTelemetry(event) {
        event.preventDefault();
        if (!form?.reportValidity()) return;
        let reading;
        try {
            reading = telemetryPayload();
        } catch (error) {
            setMessage(error.message, "error");
            return;
        }

        if (submitButton) submitButton.disabled = true;
        setMessage("Sending telemetry…");
        try {
            const result = await request("/bioreactor/telemetry-loop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(reading),
            });
            updateDashboard(reading, result);
            renderCommand(result.command);
            addHistoryRow(reading, result);
            setMessage(result.reason || "Telemetry processed.", "success");
        } catch (error) {
            setMessage(error.message || "The telemetry request failed.", "error");
        } finally {
            if (submitButton) submitButton.disabled = false;
        }
    }

    async function checkHealth() {
        try {
            const health = await request("/health");
            setConnection(true, `Backend connected • MQTT ${health.mqtt_mode || "disabled"}`);
        } catch {
            setConnection(false, "Backend unavailable — start this app with Python, not Live Server.");
        }
    }

    form?.addEventListener("submit", submitTelemetry);
    checkHealth();
})();
