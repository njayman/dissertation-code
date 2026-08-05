from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from devmind.environment import ScenarioConfig
from devmind.model_clients import BERTLargeCloud, DistilBERTEdge
from devmind.orchestrator import PolicyOrchestrator

_PRESETS = {
    "steady": ScenarioConfig.steady,
    "bursty": ScenarioConfig.bursty,
    "degraded_network": ScenarioConfig.degraded_network,
}


class ClientRequest(BaseModel):
    client_id: str
    scenario: str = "steady"
    base_rate: float = 4000.0
    burst_rate: float = 4000.0
    rtt_base: float = 40.0
    rtt_degraded: float = 40.0
    edge_stress_prob: float = 0.1
    edge_degrade_prob: float = 0.02
    max_samples: int = 200


def _scenario_from_request(req: ClientRequest) -> ScenarioConfig:
    if req.scenario in _PRESETS:
        base = _PRESETS[req.scenario]()
        base.name = req.client_id
        return base
    if req.scenario != "custom":
        raise HTTPException(400, f"unknown scenario '{req.scenario}', use steady/bursty/degraded_network/custom")
    return ScenarioConfig(
        name=req.client_id,
        base_rate=req.base_rate,
        burst_rate=req.burst_rate,
        rtt_base=req.rtt_base,
        rtt_degraded=req.rtt_degraded,
        edge_stress_prob=req.edge_stress_prob,
        edge_degrade_prob=req.edge_degrade_prob,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    orch = PolicyOrchestrator(
        library_dir=os.environ.get("DEVMIND_POLICY_LIBRARY_DIR", "policy_library"),
        log_path=os.environ.get("DEVMIND_DECISION_LOG", "docs/evaluation/orchestrator_decisions.jsonl"),
        edge_model=DistilBERTEdge(),
        cloud_model=BERTLargeCloud(),
        fine_tune_steps=int(os.environ.get("DEVMIND_FINE_TUNE_STEPS", "2000")),
        train_new_steps=int(os.environ.get("DEVMIND_TRAIN_NEW_STEPS", "8000")),
        eval_n_runs=1,
    )
    seed_path = os.environ.get("DEVMIND_POLICY_PATH", "ppo_policy.pt")
    if os.path.exists(seed_path):
        orch.register_seed_policy("seed", seed_path, validated_scenarios=["steady", "bursty", "degraded_network"])
    app.state.orchestrator = orch
    yield


app = FastAPI(title="DevMind Orchestrator Dashboard", version="0.1.0", lifespan=lifespan)


@app.get("/clients")
async def list_clients() -> list[dict]:
    orch: PolicyOrchestrator = app.state.orchestrator
    return [
        {
            "policy_id": pid,
            "clients_assigned": rec.clients_assigned,
            "validated_scenarios": rec.validated_scenarios,
        }
        for pid, rec in orch.library.items()
    ]


@app.post("/clients")
async def add_client(req: ClientRequest) -> dict:
    orch: PolicyOrchestrator = app.state.orchestrator
    scenario = _scenario_from_request(req)
    loop = asyncio.get_running_loop()
    decision = await loop.run_in_executor(None, orch.onboard, req.client_id, scenario, req.max_samples)
    assigned = next(pid for pid, rec in orch.library.items() if req.client_id in rec.clients_assigned)
    return {"client_id": req.client_id, "decision": decision.value, "policy_assigned": assigned}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "devmind-orchestrator"}


_DASHBOARD_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DevMind Policy Orchestrator</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.3rem; }
label { display: block; margin-top: 0.75rem; font-size: 0.9rem; }
input, select { width: 100%; padding: 0.4rem; box-sizing: border-box; }
button { margin-top: 1rem; padding: 0.5rem 1rem; }
table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; }
th, td { text-align: left; border-bottom: 1px solid #ccc; padding: 0.4rem; font-size: 0.85rem; }
#result { margin-top: 1rem; font-size: 0.9rem; white-space: pre-wrap; }
#custom-fields { display: none; }
</style>
</head>
<body>
<h1>DevMind Policy Orchestrator</h1>

<form id="add-form">
  <label>Client ID <input name="client_id" required></label>
  <label>Scenario
    <select name="scenario" id="scenario-select">
      <option value="steady">steady</option>
      <option value="bursty">bursty</option>
      <option value="degraded_network">degraded_network</option>
      <option value="custom">custom</option>
    </select>
  </label>
  <div id="custom-fields">
    <label>Base rate <input name="base_rate" type="number" value="4000"></label>
    <label>Burst rate <input name="burst_rate" type="number" value="4000"></label>
    <label>RTT base (ms) <input name="rtt_base" type="number" value="40"></label>
    <label>RTT degraded (ms) <input name="rtt_degraded" type="number" value="40"></label>
    <label>Edge stress prob <input name="edge_stress_prob" type="number" step="0.01" value="0.1"></label>
    <label>Edge degrade prob <input name="edge_degrade_prob" type="number" step="0.01" value="0.02"></label>
  </div>
  <button type="submit">Onboard client</button>
</form>

<div id="result"></div>

<table id="clients-table">
  <thead><tr><th>Policy</th><th>Clients</th><th>Validated scenarios</th></tr></thead>
  <tbody></tbody>
</table>

<script>
const scenarioSelect = document.getElementById("scenario-select");
const customFields = document.getElementById("custom-fields");
scenarioSelect.addEventListener("change", () => {
  customFields.style.display = scenarioSelect.value === "custom" ? "block" : "none";
});

async function refreshClients() {
  const res = await fetch("/clients");
  const rows = await res.json();
  const tbody = document.querySelector("#clients-table tbody");
  tbody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.policy_id}</td><td>${row.clients_assigned.join(", ")}</td><td>${row.validated_scenarios.join(", ")}</td>`;
    tbody.appendChild(tr);
  }
}

document.getElementById("add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const body = Object.fromEntries(form.entries());
  const resultEl = document.getElementById("result");
  resultEl.textContent = "Onboarding (this can take a while for fine-tune/train-new decisions)...";
  const res = await fetch("/clients", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  resultEl.textContent = res.ok ? JSON.stringify(data, null, 2) : `Error: ${data.detail}`;
  await refreshClients();
});

refreshClients();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return _DASHBOARD_HTML


def main() -> None:
    port = int(os.environ.get("DEVMIND_ORCH_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
