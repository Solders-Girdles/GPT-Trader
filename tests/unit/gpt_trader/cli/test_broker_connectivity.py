from __future__ import annotations

from argparse import Namespace

import gpt_trader.cli.commands.broker_connectivity as broker_cmd


def test_broker_check_coinbase_success(monkeypatch, capsys) -> None:
    class StubClient:
        def __init__(self) -> None:
            self.closed = False
            self.requested_path = ""

        def _get_endpoint_path(self, endpoint_name: str) -> str:
            assert endpoint_name == "time"
            return "/api/v3/brokerage/time"

        def get(self, path: str):
            self.requested_path = path
            return {"iso": "2025-01-01T00:00:00Z"}

        def close(self) -> None:
            self.closed = True

    client = StubClient()
    captured: dict[str, object] = {}

    def fake_build_coinbase_client(*, timeout: int | None):
        captured["timeout"] = timeout
        return client

    monkeypatch.setattr(broker_cmd, "_build_coinbase_client", fake_build_coinbase_client)

    exit_code = broker_cmd._handle_coinbase_check(
        Namespace(output_format="text", endpoint="time", timeout=11)
    )

    assert exit_code == 0
    assert captured["timeout"] == 11
    assert client.requested_path == "/api/v3/brokerage/time"
    assert client.closed is True
    out = capsys.readouterr().out
    assert "Broker check OK" in out
    assert "Raw response" in out
    assert "2025-01-01T00:00:00Z" in out


def test_broker_check_coinbase_failure_malformed_payload(monkeypatch, capsys) -> None:
    class StubClient:
        def __init__(self) -> None:
            self.closed = False

        def _get_endpoint_path(self, endpoint_name: str) -> str:
            assert endpoint_name == "time"
            return "/api/v3/brokerage/time"

        def get(self, path: str):
            return {"raw": "<html>bad</html>"}

        def close(self) -> None:
            self.closed = True

    client = StubClient()

    monkeypatch.setattr(broker_cmd, "_build_coinbase_client", lambda *, timeout: client)

    exit_code = broker_cmd._handle_coinbase_check(
        Namespace(output_format="text", endpoint="time", timeout=5)
    )

    assert exit_code == 1
    assert client.closed is True
    out = capsys.readouterr().out
    assert "Broker check FAILED" in out
    assert "Malformed JSON" in out
    assert "<html>bad</html>" in out
