from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from devmind.agent import AgenticOrchestrator
from devmind.edge import EdgeDevice
from devmind.medallion import DynamicMetricRegistry, GoldNormalizer, SilverEnricher
from devmind.model_clients import CloudClient, DistilBERTEdge, InferenceResult
from devmind.models import Action, BronzeMetricSnapshot


@dataclass
class RequestOutcome:
    request_id: str
    tier: str
    latency_ms: float
    sla_met: bool
    accuracy: float
    fallback_triggered: bool
    action: Action
    confidence_raw: float
    sla_margin_ms: float = 0.0
    trust_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tier": self.tier,
            "latency_ms": self.latency_ms,
            "sla_met": self.sla_met,
            "accuracy": self.accuracy,
            "fallback_triggered": self.fallback_triggered,
            "action": self.action.name,
            "confidence_raw": self.confidence_raw,
            "sla_margin_ms": self.sla_margin_ms,
            "trust_score": self.trust_score,
        }


class CascadeController:

    def __init__(
        self,
        agent: AgenticOrchestrator,
        edge: EdgeDevice,
        registry: DynamicMetricRegistry,
        silver: SilverEnricher,
        gold: GoldNormalizer,
        edge_model: DistilBERTEdge,
        cloud_client: CloudClient,
        edge_timeout_s: float = 2.0,
    ):
        self.agent = agent
        self.edge = edge
        self.registry = registry
        self.silver = silver
        self.gold = gold
        self.edge_model = edge_model
        self.cloud_client = cloud_client
        self.edge_timeout_s = edge_timeout_s

    async def process(
        self,
        request_id: str,
        text: str,
        sla_budget_ms: float = 300.0,
        true_label: int | None = None,
    ) -> RequestOutcome:
        loop = asyncio.get_running_loop()
        try:
            edge_result = await asyncio.wait_for(
                loop.run_in_executor(None, self.edge_model.predict, text, true_label),
                timeout=self.edge_timeout_s,
            )
        except Exception:
            self.edge.mark_unreachable()
            return await self._unreachable_fallback(request_id, text, true_label, sla_budget_ms)

        report = self.edge.emit_report(
            edge_result.confidence,
            edge_result.is_correct if true_label is not None else None,
            sla_budget_ms,
        )
        bronze = self.registry.snapshot()
        bronze.sla_budget_ms = sla_budget_ms
        bronze.sla_remaining_ms = sla_budget_ms
        silver = self.silver.enrich(bronze)
        gold = self.gold.normalize(silver)
        action, fallback = self.agent.decide(gold)

        tier, latency, accuracy = await self._dispatch(action, text, true_label, edge_result, bronze)
        sla_met = latency <= bronze.sla_budget_ms
        self.agent.reflect(latency, sla_met, accuracy, tier)
        self.edge.update_from_outcome(latency, sla_met, accuracy)
        return RequestOutcome(
            request_id=request_id,
            tier=tier,
            latency_ms=latency,
            sla_met=sla_met,
            accuracy=accuracy,
            fallback_triggered=fallback,
            action=action,
            confidence_raw=edge_result.confidence,
            sla_margin_ms=report.sla_margin_ms,
            trust_score=report.trust_score,
        )

    async def _unreachable_fallback(
        self, request_id: str, text: str, true_label: int | None, sla_budget_ms: float
    ) -> RequestOutcome:
        cloud_result = await self.cloud_client.predict(text, true_label)
        accuracy = float(cloud_result.is_correct) if true_label is not None else (
            0.5 + 0.5 * cloud_result.confidence
        )
        sla_met = cloud_result.latency_ms <= sla_budget_ms
        self.agent.reflect(cloud_result.latency_ms, sla_met, accuracy, "cloud")
        return RequestOutcome(
            request_id=request_id,
            tier="cloud",
            latency_ms=cloud_result.latency_ms,
            sla_met=sla_met,
            accuracy=accuracy,
            fallback_triggered=True,
            action=Action.ESCALATE_TO_CLOUD,
            confidence_raw=0.0,
            sla_margin_ms=0.0,
            trust_score=0.0,
        )

    async def _dispatch(
        self,
        action: Action,
        text: str,
        true_label: int | None,
        edge_result: InferenceResult,
        bronze: BronzeMetricSnapshot,
    ) -> tuple[str, float, float]:
        if action == Action.ESCALATE_TO_CLOUD:
            cloud_result = await self.cloud_client.predict(text, true_label)
            accuracy = float(cloud_result.is_correct) if true_label is not None else (
                0.5 + 0.5 * cloud_result.confidence
            )
            return "cloud", cloud_result.latency_ms, accuracy
        accuracy = float(edge_result.is_correct) if true_label is not None else (
            0.5 + 0.5 * bronze.edge_context.confidence_calibrated
        )
        return "edge", edge_result.latency_ms, accuracy
