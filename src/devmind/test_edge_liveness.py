from __future__ import annotations

import time

from devmind.edge import EdgeDevice
from devmind.medallion import GoldNormalizer, SilverEnricher
from devmind.models import BronzeMetricSnapshot, OperationalState, ResourceStress


def test_fresh_device_is_unreachable_until_first_contact() -> None:
    edge = EdgeDevice(stale_timeout_s=0.05)
    assert edge.is_unreachable
    assert edge.last_report is None


def test_contact_clears_unreachable_then_staleness_retrips_it() -> None:
    edge = EdgeDevice(stale_timeout_s=0.05)
    edge.emit_report(confidence_raw=0.8)
    assert not edge.is_unreachable
    time.sleep(0.1)
    assert edge.is_unreachable
    assert edge.last_report.operational_state == OperationalState.UNREACHABLE


def test_mark_unreachable_trips_immediately() -> None:
    edge = EdgeDevice(stale_timeout_s=5.0)
    edge.emit_report(confidence_raw=0.8)
    assert not edge.is_unreachable
    edge.mark_unreachable()
    assert edge.is_unreachable


def test_heartbeat_refreshes_liveness_without_a_request() -> None:
    edge = EdgeDevice(stale_timeout_s=0.05)
    edge.emit_report(confidence_raw=0.8)
    time.sleep(0.03)
    edge.heartbeat(ResourceStress(cpu=0.7))
    time.sleep(0.03)
    assert not edge.is_unreachable


def test_unreachable_zero_masks_resource_slots_in_gold_vector() -> None:
    edge = EdgeDevice(stale_timeout_s=0.01)
    edge.emit_report(confidence_raw=0.8)
    time.sleep(0.02)
    assert edge.is_unreachable

    bronze = BronzeMetricSnapshot(edge_context=edge.last_report, sla_remaining_ms=300.0, rtt_ms=40.0)
    silver = SilverEnricher().enrich(bronze)
    assert silver.stale
    gold = GoldNormalizer.normalize(silver)
    assert gold.mask[4:10].sum() == 0
    assert gold.slots[4:10].sum() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
