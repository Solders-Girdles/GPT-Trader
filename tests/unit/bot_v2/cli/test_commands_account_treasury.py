from __future__ import annotations

import json
from argparse import Namespace

import pytest

import bot_v2.cli.commands.account as account_cmd
import bot_v2.cli.commands.treasury as treasury_cmd


def test_account_snapshot_prints_result(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def fake_build_config(args, *, skip):
        captured["skip"] = set(skip)
        return "config"

    class SnapshotTelemetry:
        def supports_snapshots(self) -> bool:
            return True

        def collect_snapshot(self) -> dict[str, object]:
            return {"balance": 42}

    shutdown_called = {"count": 0}

    class StubBot:
        def __init__(self):
            self.account_telemetry = SnapshotTelemetry()

        async def shutdown(self):
            shutdown_called["count"] += 1

    monkeypatch.setattr(account_cmd.services, "build_config_from_args", fake_build_config)
    monkeypatch.setattr(
        account_cmd.services,
        "instantiate_bot",
        lambda config: StubBot(),
    )

    args = Namespace(profile="dev", account_command="snapshot")
    exit_code = account_cmd._handle_snapshot(args)

    assert exit_code == 0
    assert "account_command" in captured["skip"]
    out = capsys.readouterr().out
    assert json.loads(out)["balance"] == 42
    assert shutdown_called["count"] == 1


def test_account_snapshot_raises_when_unavailable(monkeypatch):
    class StubBot:
        account_telemetry = None

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubBot.shutdown_called = False

    monkeypatch.setattr(
        account_cmd.services,
        "build_config_from_args",
        lambda *_, **__: "config",
    )
    monkeypatch.setattr(
        account_cmd.services,
        "instantiate_bot",
        lambda config: StubBot(),
    )

    with pytest.raises(RuntimeError):
        account_cmd._handle_snapshot(Namespace(profile="dev", account_command="snapshot"))

    assert StubBot.shutdown_called is True


def test_treasury_convert_invokes_manager(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def fake_build_config(args, *, skip):
        captured["skip"] = set(skip)
        return "config"

    class StubManager:
        def convert(self, payload, *, commit=True):
            captured["payload"] = payload
            captured["commit"] = commit
            return {"status": "ok"}

        def move_funds(self, payload):
            raise AssertionError("Not used in convert test")

    shutdown_called = {"count": 0}

    class StubBot:
        def __init__(self):
            self.account_manager = StubManager()

        async def shutdown(self):
            shutdown_called["count"] += 1

    monkeypatch.setattr(treasury_cmd.services, "build_config_from_args", fake_build_config)
    monkeypatch.setattr(treasury_cmd.services, "instantiate_bot", lambda config: StubBot())

    args = Namespace(
        profile="dev",
        treasury_command="convert",
        from_asset="USD",
        to_asset="USDC",
        amount="100",
    )
    exit_code = treasury_cmd._handle_convert(args)

    assert exit_code == 0
    assert captured["payload"] == {"from": "USD", "to": "USDC", "amount": "100"}
    assert captured["commit"] is True
    assert "treasury_command" in captured["skip"]
    out = capsys.readouterr().out
    assert json.loads(out)["status"] == "ok"
    assert shutdown_called["count"] == 1


def test_treasury_move_invokes_manager(monkeypatch, capsys):
    class StubManager:
        def move_funds(self, payload):
            StubManager.payload = payload
            return {"moved": True}

        def convert(self, payload, *, commit=True):
            raise AssertionError("Not used in move test")

    class StubBot:
        def __init__(self):
            self.account_manager = StubManager()

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubManager.payload = None
    StubBot.shutdown_called = False

    monkeypatch.setattr(
        treasury_cmd.services,
        "build_config_from_args",
        lambda *_, **__: "config",
    )
    monkeypatch.setattr(treasury_cmd.services, "instantiate_bot", lambda config: StubBot())

    args = Namespace(
        profile="dev",
        treasury_command="move",
        from_portfolio="primary",
        to_portfolio="vault",
        amount="250",
    )
    exit_code = treasury_cmd._handle_move(args)

    assert exit_code == 0
    assert StubManager.payload == {
        "from_portfolio": "primary",
        "to_portfolio": "vault",
        "amount": "250",
    }
    assert StubBot.shutdown_called is True
    out = capsys.readouterr().out
    assert json.loads(out)["moved"] is True


def test_treasury_errors_when_manager_missing(monkeypatch):
    class StubBot:
        account_manager = None

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubBot.shutdown_called = False

    monkeypatch.setattr(
        treasury_cmd.services,
        "build_config_from_args",
        lambda *_, **__: "config",
    )
    monkeypatch.setattr(treasury_cmd.services, "instantiate_bot", lambda config: StubBot())

    with pytest.raises(RuntimeError):
        treasury_cmd._handle_convert(
            Namespace(
                profile="dev",
                treasury_command="convert",
                from_asset="USD",
                to_asset="USDC",
                amount="10",
            )
        )

    assert StubBot.shutdown_called is False
