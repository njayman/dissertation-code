from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from devmind.models import Action, GoldStateVector


class PPONetwork(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, n_actions: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy(x)
        value = self.value(x)
        return logits, value

    def get_action_and_entropy(self, state: torch.Tensor) -> tuple[int, float]:
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = int(dist.sample().item())
        entropy = float(dist.entropy().item())
        return action, entropy


@dataclass
class PerceptionCache:
    confidence: float | None = None
    cloud_queue_depth: int | None = None
    edge_rtt: float | None = None
    sla_remaining: float | None = None
    edge_cpu_load: float | None = None
    calibration_delta: float | None = None
    error_rate: float | None = None
    operational_state: str | None = None


class MCPSkillInterface:
    def __init__(self, gold_vector: GoldStateVector):
        self._gold = gold_vector
        self._skills: dict[str, bool] = {
            "get_model_confidence": True,
            "get_cloud_queue_depth": True,
            "get_edge_rtt": True,
            "get_sla_remaining": True,
            "get_edge_cpu_load": True,
            "get_edge_calibration_delta": False,
            "get_edge_error_rate": False,
            "get_edge_operational_state": False,
        }

    def list_tools(self) -> list[str]:
        return [name for name, active in self._skills.items() if active]

    def register_skill(self, name: str) -> None:
        if name in self._skills:
            self._skills[name] = True

    def call(self, skill_name: str) -> Any:
        g = self._gold
        match skill_name:
            case "get_model_confidence":
                return g[0]
            case "get_cloud_queue_depth":
                return g[1]
            case "get_edge_rtt":
                return g[2]
            case "get_sla_remaining":
                return g[3]
            case "get_edge_cpu_load":
                return g[4]
            case "get_edge_calibration_delta":
                return g[10] if g.mask[10] else None
            case "get_edge_error_rate":
                return g[11] if g.mask[11] else None
            case "get_edge_operational_state":
                return g[12]
        return None

    def perceive(self) -> PerceptionCache:
        return PerceptionCache(
            confidence=self.call("get_model_confidence"),
            cloud_queue_depth=self.call("get_cloud_queue_depth"),
            edge_rtt=self.call("get_edge_rtt"),
            sla_remaining=self.call("get_sla_remaining"),
            edge_cpu_load=self.call("get_edge_cpu_load"),
            calibration_delta=self.call("get_edge_calibration_delta"),
            error_rate=self.call("get_edge_error_rate"),
            operational_state=self.call("get_edge_operational_state"),
        )


class AgenticOrchestrator:
    ENTROPY_FALLBACK_THRESHOLD = 0.9
    FALLBACK_THRESHOLD = 0.9

    def __init__(self, policy: PPONetwork | None = None, device: str = "cpu"):
        self.policy = policy or PPONetwork()
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
        self.log_buffer: list[dict[str, Any]] = []

    def perceive(self, gold: GoldStateVector) -> PerceptionCache:
        mcp = MCPSkillInterface(gold)
        return mcp.perceive()

    def reason(self, gold: GoldStateVector) -> tuple[int, float]:
        state = torch.from_numpy(gold.slots).float().unsqueeze(0).to(self.device)
        return self.policy.get_action_and_entropy(state)

    def act(self, action: int, mcp: MCPSkillInterface) -> Action:
        if action == Action.QUERY_EXTENDED_CONTEXT:
            mcp.register_skill("get_edge_calibration_delta")
            mcp.register_skill("get_edge_error_rate")
            mcp.register_skill("get_edge_operational_state")
            return Action.QUERY_EXTENDED_CONTEXT
        return Action(action)

    def decide(self, gold: GoldStateVector) -> tuple[Action, bool]:
        action, entropy = self.reason(gold)
        fallback = entropy > self.ENTROPY_FALLBACK_THRESHOLD
        if fallback:
            confidence = gold[0]
            action = Action.ESCALATE_TO_CLOUD if confidence < self.FALLBACK_THRESHOLD else Action.ROUTE_TO_EDGE
            return action, fallback
        if action == Action.QUERY_EXTENDED_CONTEXT:
            return Action.ROUTE_TO_EDGE, False
        return Action(action), False

    def reflect(self, latency_ms: float, sla_met: bool, accuracy: float, tier: str) -> None:
        self.log_buffer.append(
            {"latency_ms": latency_ms, "sla_met": sla_met, "accuracy": accuracy, "tier": tier}
        )
