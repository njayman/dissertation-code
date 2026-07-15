from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from devmind.agent import AgenticOrchestrator, PPONetwork
from devmind.cascade import CascadeController, RequestOutcome
from devmind.edge import EdgeDevice, MiscalibrationClassifier
from devmind.medallion import DynamicMetricRegistry, GoldNormalizer, MetricSource, SilverEnricher
from devmind.model_clients import DistilBERTEdge

_DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ppo_policy.pt")


def _load_policy() -> PPONetwork:
    policy = PPONetwork()
    path = os.environ.get("DEVMIND_POLICY_PATH", _DEFAULT_POLICY_PATH)
    if os.path.exists(path):
        policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        raise RuntimeError(
            f"No trained policy found at {path}. Train one with `python -m devmind.trainer` "
            "or set DEVMIND_POLICY_PATH, before starting the gateway."
        )
    return policy


class InferenceRequest(BaseModel):
    text: str
    sla_budget_ms: float = 300.0
    edge_confidence: float | None = None


class InferenceResponse(BaseModel):
    request_id: str
    tier: str
    confident: bool
    sla_met: bool
    latency_ms: float
    fallback_triggered: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.edge_model = DistilBERTEdge()
    edge = EdgeDevice()
    registry = DynamicMetricRegistry()
    registry.register(MetricSource("edge_context", "EdgeContextReport", lambda: edge.last_report or edge.emit_report(0.5)))
    registry.register(MetricSource("cloud_queue_depth", "int", lambda: 0))
    registry.register(MetricSource("rtt_ms", "float", lambda: 40.0))
    registry.register(MetricSource("sla_remaining_ms", "float", lambda: 300.0))
    registry.register(MetricSource("energy_mj", "float", lambda: 0.0))
    registry.register(MetricSource("traffic_intensity", "float", lambda: 4000.0))

    silver = SilverEnricher()
    gold = GoldNormalizer()
    agent = AgenticOrchestrator(_load_policy())
    controller = CascadeController(agent, edge, registry, silver, gold)
    app.state.controller = controller
    yield


app = FastAPI(title="DevMind Gateway", version="0.1.0", lifespan=lifespan)


@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest) -> InferenceResponse:
    controller: CascadeController = app.state.controller
    if req.edge_confidence is not None:
        confidence = req.edge_confidence
    else:
        edge_model: DistilBERTEdge = app.state.edge_model
        # ponytail: offload the blocking transformer forward pass to the
        # default threadpool so it doesn't stall the event loop under
        # concurrent requests. Move to a separate TorchServe-style process
        # (like the cloud tier already implies) if throughput needs more
        # than one process's worth of threads.
        result = await asyncio.get_running_loop().run_in_executor(None, edge_model.predict, req.text)
        confidence = result.confidence
    outcome = controller.process(str(uuid.uuid4()), confidence)
    return InferenceResponse(
        request_id=outcome.request_id,
        tier=outcome.tier,
        confident=outcome.accuracy > 0.6,
        sla_met=outcome.sla_met,
        latency_ms=outcome.latency_ms,
        fallback_triggered=outcome.fallback_triggered,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "devmind-gateway"}


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
