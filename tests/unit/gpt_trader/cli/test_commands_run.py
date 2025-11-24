from __future__ import annotations

import asyncio
import logging
import signal
from argparse import Namespace

import gpt_trader.cli.commands.run as run_cmd
from gpt_trader.orchestration.configuration import ConfigValidationError


def _run_coroutine(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_execute_logs_specific_message_for_symbol_error(monkeypatch, caplog):
    def fake_build_config(*_, **__):
        raise ConfigValidationError(["symbols overrides must be non-empty"])

    monkeypatch.setattr(run_cmd.services, "build_config_from_args", fake_build_config)
    monkeypatch.setattr(run_cmd.services, "instantiate_bot", lambda config: None)

    caplog.set_level(logging.ERROR, logger=run_cmd.logger.name)
    result = run_cmd.execute(Namespace(dev_fast=False, profile="dev"))

    assert result == 1
    assert caplog.records[-1].message == "Symbols must be non-empty"


def test_execute_logs_generic_validation_error(monkeypatch, caplog):
    def fake_build_config(*_, **__):
        raise ConfigValidationError(["something else failed"])

    monkeypatch.setattr(run_cmd.services, "build_config_from_args", fake_build_config)
    monkeypatch.setattr(run_cmd.services, "instantiate_bot", lambda config: None)

    caplog.set_level(logging.ERROR, logger=run_cmd.logger.name)
    result = run_cmd.execute(Namespace(dev_fast=False, profile="dev"))

    assert result == 1
    assert caplog.records[-1].message == "something else failed"


def test_execute_invokes_run_bot(monkeypatch):
    monkeypatch.setattr(run_cmd.services, "build_config_from_args", lambda *_, **__: "config")

    class StubBot:
        pass

    stub_bot = StubBot()
    monkeypatch.setattr(run_cmd.services, "instantiate_bot", lambda config: stub_bot)

    captured = {}

    def fake_run_bot(bot, *, single_cycle):
        captured["bot"] = bot
        captured["single_cycle"] = single_cycle
        return 0

    monkeypatch.setattr(run_cmd, "_run_bot", fake_run_bot)

    result = run_cmd.execute(Namespace(dev_fast=True, profile="dev"))

    assert result == 0
    assert captured["bot"] is stub_bot
    assert captured["single_cycle"] is True


def test_run_bot_sets_signal_handlers_and_runs(monkeypatch):
    handlers: dict[int, object] = {}
    monkeypatch.setattr(
        run_cmd.signal,
        "signal",
        lambda sig, handler: handlers.setdefault(sig, handler),
    )
    monkeypatch.setattr(run_cmd.asyncio, "run", _run_coroutine)

    class StubBot:
        def __init__(self):
            self.running = True
            self.single_cycle = None

        async def run(self, *, single_cycle: bool):
            self.single_cycle = single_cycle

    bot = StubBot()
    result = run_cmd._run_bot(bot, single_cycle=True)

    assert result == 0
    assert bot.single_cycle is True
    assert signal.SIGINT in handlers
    assert signal.SIGTERM in handlers

    # Invoke the registered handler to ensure it toggles the bot flag.
    handlers[signal.SIGINT](signal.SIGINT, None)
    assert bot.running is False
