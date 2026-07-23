"""Policy Orchestration Layer (Component 7).

Sits above the PPO Agent (agent.py) and the Bidirectional Loop. Where the
Bidirectional Loop improves ONE deployed policy from its own outcome history,
this layer answers a fleet-level question at client onboarding / drift-detection
time: does an existing policy in the library already cover this client's
traffic scenario (reuse), does it need adapting (fine-tune), or is nothing in
the library close enough (train new)?

Runs offline/governance-time only - never in the per-request fast loop. The
per-request perception-reason-act-reflect loop in agent.py is untouched; it
always executes one already-assigned frozen policy.

Every decision is evaluated in the existing Gymnasium simulator using the
existing evaluation harness (EvalMetrics: accuracy, P95 latency, SLA
violation rate, escalation rate, energy) and written to an append-only
decision log, this is the evidence trail for "why does this client run this
policy."
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import torch

from devmind.agent import AgenticOrchestrator, PPONetwork
from devmind.environment import InferenceGatewayEnv, ScenarioConfig
from devmind.evaluation import EvalMetrics, make_env, ppo_policy, run_episode
from devmind.trainer import PPOTrainer


class PolicyDecision(str, Enum):
    REUSE = "reuse"
    FINE_TUNE = "fine_tune"
    TRAIN_NEW = "train_new"


@dataclass
class PolicyRecord:
    policy_id: str
    checkpoint_path: str
    validated_scenarios: list[str] = field(default_factory=list)
    clients_assigned: list[str] = field(default_factory=list)


@dataclass
class ToleranceThresholds:
    """The 'task capacity' envelope (Kessler et al., Lifetime Policy Reuse,
    arXiv:2106.01741) a policy must stay within to be reused as-is."""

    max_sla_violation_rate: float = 0.15
    min_accuracy: float = 0.80
    max_escalation_rate: float = 0.60


def _meets_tolerance(m: EvalMetrics, t: ToleranceThresholds) -> bool:
    return (
        m.sla_violation_rate <= t.max_sla_violation_rate
        and m.accuracy >= t.min_accuracy
        and m.escalation_rate <= t.max_escalation_rate
    )


def select_decision(
    candidates: dict[str, EvalMetrics], thresholds: ToleranceThresholds
) -> tuple[PolicyDecision, str | None]:
    """Pure promotion rule (champion-challenger style): reuse the best
    in-tolerance policy, fine-tune the closest miss, or signal that a fresh
    policy is needed if the library is empty. No I/O, no training - kept
    separate from onboard() so the decision logic is unit-testable without a
    simulator or GPU."""
    fits = {pid: m for pid, m in candidates.items() if _meets_tolerance(m, thresholds)}
    if fits:
        best = max(fits, key=lambda pid: fits[pid].accuracy - fits[pid].sla_violation_rate)
        return PolicyDecision.REUSE, best
    if candidates:
        closest = max(candidates, key=lambda pid: candidates[pid].accuracy - candidates[pid].sla_violation_rate)
        return PolicyDecision.FINE_TUNE, closest
    return PolicyDecision.TRAIN_NEW, None


def dominant_signal(m: EvalMetrics, thresholds: ToleranceThresholds) -> str:
    """Cheap explainability hook: which metric drove the decision, not a
    SHAP-style attribution pass."""
    gaps = {
        "sla_violation_rate": m.sla_violation_rate - thresholds.max_sla_violation_rate,
        "accuracy": thresholds.min_accuracy - m.accuracy,
        "escalation_rate": m.escalation_rate - thresholds.max_escalation_rate,
    }
    worst = max(gaps, key=gaps.get)
    return worst if gaps[worst] > 0 else "within_tolerance"


class PolicyOrchestrator:
    def __init__(
        self,
        library_dir: str = "policy_library",
        thresholds: ToleranceThresholds | None = None,
        edge_model: Any = None,
        cloud_model: Any = None,
        log_path: str | None = None,
        fine_tune_steps: int = 5_000,
        train_new_steps: int = 50_000,
    ):
        self.library_dir = library_dir
        self.thresholds = thresholds or ToleranceThresholds()
        self.library: dict[str, PolicyRecord] = {}
        self._edge_model = edge_model
        self._cloud_model = cloud_model
        self.fine_tune_steps = fine_tune_steps
        self.train_new_steps = train_new_steps
        self.log_path = log_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "docs", "evaluation", "orchestrator_decisions.jsonl"
        )
        os.makedirs(self.library_dir, exist_ok=True)

    def register_seed_policy(self, policy_id: str, checkpoint_path: str, validated_scenarios: list[str]) -> None:
        self.library[policy_id] = PolicyRecord(policy_id, checkpoint_path, validated_scenarios)

    def onboard(self, client: str, scenario: ScenarioConfig, max_samples: int = 500) -> PolicyDecision:
        candidates = {
            pid: self._evaluate(rec.checkpoint_path, scenario, max_samples)
            for pid, rec in self.library.items()
        }
        decision, chosen = select_decision(candidates, self.thresholds)

        if decision == PolicyDecision.FINE_TUNE:
            chosen = self._fine_tune(chosen, scenario)
        elif decision == PolicyDecision.TRAIN_NEW:
            chosen = self._train_new(client, scenario)

        rec = self.library[chosen]
        if scenario.name not in rec.validated_scenarios:
            rec.validated_scenarios.append(scenario.name)
        if client not in rec.clients_assigned:
            rec.clients_assigned.append(client)

        self._log_decision(client, scenario, candidates, decision, chosen)
        return decision

    def _evaluate(self, checkpoint_path: str, scenario: ScenarioConfig, max_samples: int) -> EvalMetrics:
        ppo = PPONetwork()
        ppo.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        agent = AgenticOrchestrator(ppo)
        env = make_env(scenario, max_samples, self._edge_model, self._cloud_model)
        return run_episode(env, ppo_policy(agent), desc=f"orchestrator/{scenario.name}")

    def _train(self, scenario: ScenarioConfig, total_steps: int, init_state_dict: dict | None = None) -> PPONetwork:
        env = InferenceGatewayEnv(scenario, edge_model=self._edge_model, cloud_model=self._cloud_model)
        trainer = PPOTrainer(env)
        if init_state_dict is not None:
            trainer.policy.load_state_dict(init_state_dict)
        step = 0
        while step < total_steps:
            trainer.collect_rollout(2048)
            trainer.train()
            step += 2048
        return trainer.policy

    def _fine_tune(self, base_policy_id: str, scenario: ScenarioConfig, steps: int | None = None) -> str:
        base = self.library[base_policy_id]
        state_dict = torch.load(base.checkpoint_path, map_location="cpu", weights_only=True)
        policy = self._train(scenario, steps or self.fine_tune_steps, init_state_dict=state_dict)
        new_id = f"{base_policy_id}_ft_{scenario.name}"
        path = os.path.join(self.library_dir, f"{new_id}.pt")
        torch.save(policy.state_dict(), path)
        self.library[new_id] = PolicyRecord(new_id, path)
        return new_id

    def _train_new(self, client: str, scenario: ScenarioConfig, steps: int | None = None) -> str:
        policy = self._train(scenario, steps or self.train_new_steps)
        new_id = f"{client}_{scenario.name}"
        path = os.path.join(self.library_dir, f"{new_id}.pt")
        torch.save(policy.state_dict(), path)
        self.library[new_id] = PolicyRecord(new_id, path)
        return new_id

    def _log_decision(
        self,
        client: str,
        scenario: ScenarioConfig,
        candidates: dict[str, EvalMetrics],
        decision: PolicyDecision,
        chosen: str,
    ) -> None:
        signal = dominant_signal(candidates[chosen], self.thresholds) if chosen in candidates else "n/a_new_policy"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client": client,
            "scenario": scenario.name,
            "candidates_evaluated": {pid: vars(m) for pid, m in candidates.items()},
            "decision": decision.value,
            "policy_assigned": chosen,
            "dominant_signal": signal,
        }
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


CLIENT_SCENARIOS: dict[str, ScenarioConfig] = {
    "client_streamforge": ScenarioConfig.bursty(),
    "client_nhs": ScenarioConfig.steady(),
    "client_babcock": ScenarioConfig.degraded_network(),
    "client_newco": ScenarioConfig(
        name="client_newco",
        base_rate=4000,
        burst_rate=4000,
        edge_stress_prob=0.35,
        edge_degrade_prob=0.10,
    ),
}


def run_ablation_7(
    seed_policy_path: str = "ppo_policy.pt",
    max_samples: int = 500,
    fine_tune_steps: int = 5_000,
    train_new_steps: int = 50_000,
    edge_model: Any = None,
    cloud_model: Any = None,
) -> dict[str, Any]:
    """Ablation Run 7: single shared policy across every client vs the Policy
    Orchestration Layer choosing reuse/fine-tune/train-new per client. Isolates
    the value of fleet-level governance (Component 7) from single-policy
    generalisation, which Run 6's held-out scenario already measures."""
    shared_ppo = PPONetwork()
    shared_ppo.load_state_dict(torch.load(seed_policy_path, map_location="cpu", weights_only=True))
    shared_agent = AgenticOrchestrator(shared_ppo)

    shared_results: dict[str, EvalMetrics] = {}
    for client, scenario in CLIENT_SCENARIOS.items():
        env = make_env(scenario, max_samples, edge_model, cloud_model)
        shared_results[client] = run_episode(env, ppo_policy(shared_agent), desc=f"run7_shared/{client}")

    orch = PolicyOrchestrator(
        edge_model=edge_model,
        cloud_model=cloud_model,
        fine_tune_steps=fine_tune_steps,
        train_new_steps=train_new_steps,
    )
    orch.register_seed_policy(
        "seed", seed_policy_path, validated_scenarios=["steady", "bursty", "degraded_network"]
    )

    decisions: dict[str, PolicyDecision] = {}
    orchestrated_results: dict[str, EvalMetrics] = {}
    for client, scenario in CLIENT_SCENARIOS.items():
        decisions[client] = orch.onboard(client, scenario, max_samples=max_samples)
        assigned = next(pid for pid, rec in orch.library.items() if client in rec.clients_assigned)
        orchestrated_results[client] = orch._evaluate(orch.library[assigned].checkpoint_path, scenario, max_samples)

    return {
        "shared": shared_results,
        "orchestrated": orchestrated_results,
        "decisions": {k: v.value for k, v in decisions.items()},
    }


def main_ablation_7() -> None:
    """Standalone entry point (like trainer.py's train_agent) since Run 7
    trains real policies and shouldn't slow down every `devmind-eval` sweep.
    Run explicitly: `uv run devmind-ablation7`."""
    import datetime
    import time

    from devmind.evaluation import print_results, save_results
    from devmind.model_clients import BERTLargeCloud, DistilBERTEdge

    print("Loading models (one-time)...")
    t0 = time.perf_counter()
    edge_model = DistilBERTEdge()
    cloud_model = BERTLargeCloud()
    print(f"Models loaded in {time.perf_counter() - t0:.1f}s")

    result = run_ablation_7(edge_model=edge_model, cloud_model=cloud_model)

    text_buffer: list[str] = []
    json_buffer: list[dict] = []
    save_results(result["shared"], "RUN 7: Single Shared Policy", text_buffer, json_buffer)
    save_results(result["orchestrated"], "RUN 7: Policy Orchestration Layer", text_buffer, json_buffer)
    print_results(result["shared"], title="RUN 7: Single Shared Policy")
    print_results(result["orchestrated"], title="RUN 7: Policy Orchestration Layer")
    print("\nPer-client decisions:", result["decisions"])
    text_buffer.append(f"\nPer-client decisions: {result['decisions']}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(out_dir, f"run7_{timestamp}.txt"), "w") as f:
        f.write("\n".join(text_buffer))
    with open(os.path.join(out_dir, f"run7_{timestamp}.json"), "w") as f:
        json.dump(json_buffer, f, indent=2)
    print(f"\nResults saved to {out_dir}/run7_{timestamp}.*")


def demo() -> None:
    t = ToleranceThresholds()

    good = EvalMetrics(accuracy=0.9, sla_violation_rate=0.05, escalation_rate=0.3)
    close_miss = EvalMetrics(accuracy=0.7, sla_violation_rate=0.2, escalation_rate=0.5)

    decision, chosen = select_decision({"p1": good}, t)
    assert decision == PolicyDecision.REUSE and chosen == "p1"

    decision, chosen = select_decision({"p1": close_miss}, t)
    assert decision == PolicyDecision.FINE_TUNE and chosen == "p1"

    decision, chosen = select_decision({}, t)
    assert decision == PolicyDecision.TRAIN_NEW and chosen is None

    assert dominant_signal(close_miss, t) == "accuracy"
    assert dominant_signal(good, t) == "within_tolerance"

    print("orchestrator self-check passed")


if __name__ == "__main__":
    demo()
