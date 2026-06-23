"""Tests for the reduce-only canary smoke script's live-order gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import scripts.monitoring.canary_reduce_only_test as canary_reduce_only_test


class _FakeClient:
    def __init__(self, auth: object) -> None:
        self.auth = auth
        self.api_mode = ""
        self.placed_orders: list[dict[str, object]] = []
        self.cancelled_order_ids: list[str] = []

    def place_order(self, order: dict[str, object]) -> dict[str, str]:
        self.placed_orders.append(order)
        return {"order_id": "order-123"}

    def cancel_orders(self, order_ids: list[str]) -> dict[str, object]:
        self.cancelled_order_ids.extend(order_ids)
        return {"cancelled": order_ids}


def test_live_mode_requires_human_approved_confirmation_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_credentials_are_resolved(credentials_file: str | None) -> None:
        raise AssertionError("credentials should not be resolved before live confirmation")

    monkeypatch.setattr(
        canary_reduce_only_test, "_resolve_credentials", fail_if_credentials_are_resolved
    )

    result = canary_reduce_only_test.main(["--live"])

    assert result == 2
    assert "--confirm-human-approved-live-order" in capsys.readouterr().out


def test_live_mode_with_confirmation_can_place_and_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_clients: list[_FakeClient] = []

    monkeypatch.setattr(
        canary_reduce_only_test,
        "_resolve_credentials",
        lambda credentials_file: SimpleNamespace(
            key_name="organizations/test/apiKeys/key", private_key="key", source="test"
        ),
    )
    monkeypatch.setattr(canary_reduce_only_test, "create_cdp_jwt_auth", lambda **kwargs: kwargs)

    def build_client(auth: object) -> _FakeClient:
        client = _FakeClient(auth)
        fake_clients.append(client)
        return client

    monkeypatch.setattr(canary_reduce_only_test, "CoinbaseClient", build_client)
    monkeypatch.setattr(canary_reduce_only_test.time, "sleep", lambda seconds: None)

    result = canary_reduce_only_test.main(["--live", "--confirm-human-approved-live-order"])

    assert result == 0
    assert len(fake_clients) == 1
    assert len(fake_clients[0].placed_orders) == 1
    assert fake_clients[0].placed_orders[0]["order_configuration"] == {
        "limit_limit_gtc": {
            "base_size": "0.001",
            "limit_price": "10.0",
            "post_only": True,
            "reduce_only": True,
        }
    }
    assert fake_clients[0].cancelled_order_ids == ["order-123"]
