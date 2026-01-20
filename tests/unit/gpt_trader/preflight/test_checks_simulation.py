"""Tests for dry-run simulation preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.checks.simulation import simulate_dry_run
from gpt_trader.preflight.core import PreflightCheck


@dataclass(frozen=True, slots=True)
class SimulationStubs:
    bot_config: MagicMock
    create_container: MagicMock
    config: MagicMock
    container: MagicMock
    bot: MagicMock


@pytest.fixture
def simulation_stubs(monkeypatch: pytest.MonkeyPatch) -> SimulationStubs:
    import gpt_trader.app.config as config_module
    import gpt_trader.app.container as container_module

    config = MagicMock(name="config")
    bot = MagicMock(name="bot")
    bot.engine = MagicMock(name="engine")
    container = MagicMock(name="container")
    container.create_bot.return_value = bot

    bot_config = MagicMock(name="BotConfig")
    bot_config.from_profile.return_value = config
    monkeypatch.setattr(config_module, "BotConfig", bot_config)

    create_container = MagicMock(name="create_application_container", return_value=container)
    monkeypatch.setattr(container_module, "create_application_container", create_container)

    return SimulationStubs(
        bot_config=bot_config,
        create_container=create_container,
        config=config,
        container=container,
        bot=bot,
    )


class TestSimulateDryRun:
    """Test dry-run simulation."""

    def test_passes_when_simulation_succeeds(self, simulation_stubs: SimulationStubs) -> None:
        """Should pass when bot construction succeeds."""
        checker = PreflightCheck(profile="dev", verbose=True)

        result = simulate_dry_run(checker)

        assert result is True
        assert any("Application container initialized" in s for s in checker.successes)
        assert any("TradingBot constructed" in s for s in checker.successes)
        assert any("Trading engine available" in s for s in checker.successes)
        assert any("Dry-run simulation passed" in s for s in checker.successes)

    def test_passes_when_engine_is_none(self, simulation_stubs: SimulationStubs) -> None:
        """Should pass even when engine is None (some configs)."""
        checker = PreflightCheck(profile="dev")

        simulation_stubs.bot.engine = None
        result = simulate_dry_run(checker)

        assert result is True
        # Should not have "engine available" success since engine is None
        assert not any("Trading engine available" in s for s in checker.successes)

    def test_fails_on_config_error(self, simulation_stubs: SimulationStubs) -> None:
        """Should fail when config creation fails."""
        checker = PreflightCheck(profile="dev")

        simulation_stubs.bot_config.from_profile.side_effect = Exception("Config error")
        result = simulate_dry_run(checker)

        assert result is False
        assert any("Dry-run simulation failed" in e for e in checker.errors)

    def test_fails_on_container_error(self, simulation_stubs: SimulationStubs) -> None:
        """Should fail when container creation fails."""
        checker = PreflightCheck(profile="dev")

        simulation_stubs.create_container.side_effect = Exception("Container init failed")
        result = simulate_dry_run(checker)

        assert result is False
        assert any("Dry-run simulation failed" in e for e in checker.errors)

    def test_fails_on_bot_creation_error(self, simulation_stubs: SimulationStubs) -> None:
        """Should fail when bot creation fails."""
        checker = PreflightCheck(profile="dev")

        simulation_stubs.container.create_bot.side_effect = Exception("Bot creation failed")
        result = simulate_dry_run(checker)

        assert result is False
        assert any("Dry-run simulation failed" in e for e in checker.errors)

    def test_uses_correct_config_params(self, simulation_stubs: SimulationStubs) -> None:
        """Should create config with dry_run=True and mock_broker=True."""
        checker = PreflightCheck(profile="canary")

        simulate_dry_run(checker)

        simulation_stubs.bot_config.from_profile.assert_called_once_with(
            profile="canary", dry_run=True, mock_broker=True
        )

    def test_prints_section_header(
        self,
        simulation_stubs: SimulationStubs,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        simulation_stubs.bot_config.from_profile.side_effect = Exception("Skip")
        simulate_dry_run(checker)

        captured = capsys.readouterr()
        assert "DRY-RUN SIMULATION" in captured.out

    def test_logs_profile_info_when_verbose(
        self,
        simulation_stubs: SimulationStubs,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Should log config info when verbose."""
        checker = PreflightCheck(profile="canary", verbose=True)

        simulate_dry_run(checker)

        captured = capsys.readouterr()
        assert "canary" in captured.out
