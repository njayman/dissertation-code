from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from devmind.agent import AgenticOrchestrator
from devmind.edge import EdgeDevice
from devmind.medallion import DynamicMetricRegistry, GoldNormalizer, MetricSource, SilverEnricher
from devmind.models import Action, BronzeMetricSnapshot, EdgeContextReport, GoldStateVector


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
    ):
        self.agent = agent
        self.edge = edge
        self.registry = registry
        self.silver = silver
        self.gold = gold

    def process(self, request_id: str, confidence_raw: float) -> RequestOutcome:
        report = self.edge.emit_report(confidence_raw)
        bronze = self.registry.snapshot()
        silver = self.silver.enrich(bronze)
        gold = self.gold.normalize(silver)
        action, fallback = self.agent.decide(gold)
        tier, latency, accuracy = self._dispatch(action, confidence_raw, bronze)
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
            confidence_raw=confidence_raw,
            sla_margin_ms=report.sla_margin_ms,
            trust_score=report.trust_score,
        )

    def _dispatch(self, action: Action, confidence: float, bronze: BronzeMetricSnapshot) -> tuple[str, float, float]:
        if action == Action.ROUTE_TO_EDGE:
            latency = 50.0 + bronze.rtt_ms * 0.5 + bronze.edge_context.resource_stress.cpu * 30.0
            accuracy = 0.5 + 0.5 * bronze.edge_context.confidence_calibrated
            return "edge", latency, float(accuracy)
        if action == Action.ESCALATE_TO_CLOUD:
            latency = bronze.rtt_ms + 100.0
            return "cloud", latency, 0.88
        return self._dispatch(Action.ROUTE_TO_EDGE, confidence, bronze)
