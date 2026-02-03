from __future__ import annotations

from argparse import Namespace

import gpt_trader.cli.commands.coinbase as coinbase_cmd


def test_coinbase_connectivity_success(monkeypatch, capsys) -> None:
    class StubClient:
        def __init__(self) -> None:
            self.closed = False

        def get_time(self):
            return {"iso": "2025-01-01T00:00:00Z"}

        def close(self) -> None:
            self.closed = True

    client = StubClient()
    monkeypatch.setattr(coinbase_cmd, "_build_coinbase_client", lambda: client)

    exit_code = coinbase_cmd._handle_connectivity(Namespace(output_format="text"))

    assert exit_code == 0
    assert client.closed is True
    out = capsys.readouterr().out
    assert "Coinbase connectivity OK" in out
    assert "2025-01-01T00:00:00Z" in out


def test_coinbase_connectivity_failure(monkeypatch, capsys) -> None:
    class StubClient:
        def __init__(self) -> None:
            self.closed = False

        def get_time(self):
            raise RuntimeError("boom")

        def close(self) -> None:
            self.closed = True

    client = StubClient()
    monkeypatch.setattr(coinbase_cmd, "_build_coinbase_client", lambda: client)

    exit_code = coinbase_cmd._handle_connectivity(Namespace(output_format="text"))

    assert exit_code == 1
    assert client.closed is True
    out = capsys.readouterr().out
    assert "Coinbase connectivity FAILED" in out
    assert "RuntimeError: boom" in out
