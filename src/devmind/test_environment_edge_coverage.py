from __future__ import annotations

from devmind.environment import InferenceGatewayEnv, ScenarioConfig
from devmind.evaluation import evaluate_baselines, run_episode
from devmind.models import Action, GoldStateVector, OperationalState


def test_unreachable_prob_forces_cloud_override_and_updates_gold_mask() -> None:
    scenario = ScenarioConfig.steady()
    scenario.edge_unreachable_prob = 1.0
    env = InferenceGatewayEnv(scenario, max_samples=10)
    env.reset()
    state, _, _, _, _ = env.step(int(Action.ROUTE_TO_EDGE))
    assert env._state.edge_unreachable_events == 1
    assert env._state.escalations == 1
    assert env._state.edge_routes == 0

    gold = GoldStateVector(slots=state, mask=None)
    assert gold.slots[12] == 1.0


def test_unreachable_prob_zero_never_overrides() -> None:
    scenario = ScenarioConfig.steady()
    scenario.edge_unreachable_prob = 0.0
    env = InferenceGatewayEnv(scenario, max_samples=10)
    env.reset()
    for _ in range(5):
        env.step(int(Action.ROUTE_TO_EDGE))
    assert env._state.edge_unreachable_events == 0
    assert env._state.edge_routes == 5


def test_memory_disk_stress_adds_edge_latency() -> None:
    scenario = ScenarioConfig.steady()
    env = InferenceGatewayEnv(scenario, max_samples=5)
    env.reset()
    env._edge_device.apply_stress(memory=1.0, disk_io=1.0, cpu=0.0, thermal=0.0)
    reward, latency, sla_met, accuracy = env._inference_step(int(Action.ROUTE_TO_EDGE))
    assert latency > 60.0


def test_eval_metrics_carry_component7_evidence_fields() -> None:
    results = evaluate_baselines(ScenarioConfig.steady(), n_runs=1, max_samples=20)
    always_edge = results["always_edge"]
    assert 0.0 <= always_edge.trust_score <= 1.0
    assert always_edge.sla_margin_ms != 0.0 or always_edge.trust_score != 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
