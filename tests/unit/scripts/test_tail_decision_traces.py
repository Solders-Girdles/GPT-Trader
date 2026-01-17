from __future__ import annotations

from scripts.ops import tail_decision_traces


def test_resolve_decision_id_prefers_decision_id() -> None:
    payload = {"decision_id": "decision-1", "client_order_id": "client-1"}
    assert tail_decision_traces._resolve_decision_id(payload) == "decision-1"


def test_resolve_decision_id_falls_back_to_client_order_id() -> None:
    payload = {"client_order_id": "client-2"}
    assert tail_decision_traces._resolve_decision_id(payload) == "client-2"
