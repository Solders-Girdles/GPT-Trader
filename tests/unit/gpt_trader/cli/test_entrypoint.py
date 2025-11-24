from __future__ import annotations

import importlib
import logging
import sys
from unittest.mock import Mock

import pytest

import gpt_trader.config.runtime_settings as runtime_settings_module


@pytest.fixture
def reload_cli(monkeypatch, runtime_settings_factory):
    """Reload gpt_trader.cli with deterministic settings and logging stubs."""

    def _loader(env_overrides: dict[str, str] | None = None):
        # Ensure dotenv loading and logging setup are no-ops during import
        monkeypatch.setattr("dotenv.load_dotenv", lambda *_, **__: None)
        configure_mock = Mock()
        monkeypatch.setattr("gpt_trader.logging.configure_logging", configure_mock)

        settings = runtime_settings_factory(
            env_overrides=env_overrides or {},
        )
        real_loader = runtime_settings_module.load_runtime_settings
        monkeypatch.setattr(
            runtime_settings_module,
            "load_runtime_settings",
            lambda env=None: settings if env is None else real_loader(env),
        )

        if "gpt_trader.cli" in sys.modules:
            module = sys.modules["gpt_trader.cli"]
        else:
            module = importlib.import_module("gpt_trader.cli")
        module = importlib.reload(module)
        return module, settings, configure_mock

    return _loader


def test_main_dispatches_to_handler(monkeypatch, reload_cli):
    module, settings, configure_mock = reload_cli()

    captured: dict[str, object] = {}

    def fake_execute(args):
        captured["args"] = args
        return 7

    monkeypatch.setattr(module.run, "execute", fake_execute)

    exit_code = module.main(["run", "--profile", "canary"])

    assert exit_code == 7
    assert configure_mock.called
    cli_services = importlib.import_module("gpt_trader.cli.services")
    assert cli_services.OVERRIDE_SETTINGS is settings
    assert captured["args"].profile == "canary"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["run"]),
        (["run", "--dry-run"], ["run", "--dry-run"]),
        (["--help"], ["run", "--help"]),
        (["orders", "preview"], ["orders", "preview"]),
        (["--dry-run"], ["run", "--dry-run"]),
        (["unknown", "--flag"], ["run", "unknown", "--flag"]),
        (["orders", "--help"], ["orders", "--help"]),
    ],
)
def test_ensure_command_normalizes_invocation(argv, expected, reload_cli):
    module, _, _ = reload_cli()
    assert module._ensure_command(argv) == expected


def test_env_flag_uses_runtime_snapshot(reload_cli):
    module, _, _ = reload_cli(env_overrides={"PERPS_DEBUG": "true"})
    assert module._env_flag("PERPS_DEBUG") is True
    assert module._env_flag("MISSING_FLAG") is False


def test_maybe_enable_debug_logging(monkeypatch, reload_cli):
    module, _, _ = reload_cli(env_overrides={"PERPS_DEBUG": "yes"})

    perps_logger = logging.getLogger("gpt_trader.features.brokerages.coinbase")
    orchestrator_logger = logging.getLogger("gpt_trader.orchestration")
    original_perps_level = perps_logger.level
    original_orchestrator_level = orchestrator_logger.level

    perps_logger.setLevel(logging.INFO)
    orchestrator_logger.setLevel(logging.WARNING)
    try:
        module._maybe_enable_debug_logging()
        assert perps_logger.level == logging.DEBUG
        assert orchestrator_logger.level == logging.DEBUG
    finally:
        perps_logger.setLevel(original_perps_level)
        orchestrator_logger.setLevel(original_orchestrator_level)


def test_main_defaults_to_run_command(monkeypatch, reload_cli):
    module, _, _ = reload_cli()

    invoked: dict[str, object] = {}

    def fake_execute(args):
        invoked["args"] = args
        return None

    monkeypatch.setattr(module.run, "execute", fake_execute)

    exit_code = module.main(["--dry-run"])

    assert exit_code == 0
    assert invoked["args"].command == "run"
    assert invoked["args"].dry_run is True
