"""Tests for dry-run simulation preflight checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.checks.simulation import simulate_dry_run
from gpt_trader.preflight.core import PreflightCheck


class TestSimulateDryRun:
    """Test dry-run simulation."""

    def test_passes_when_simulation_succeeds(self) -> None:
        """Should pass when bot construction succeeds."""
        checker = PreflightCheck(profile="dev", verbose=True)

        mock_config = MagicMock()
        mock_container = MagicMock()
        mock_bot = MagicMock()
        mock_bot.engine = MagicMock()
        mock_container.create_bot.return_value = mock_bot

        with (
            patch("gpt_trader.orchestration.configuration.BotConfig") as mock_bot_config,
            patch("gpt_trader.app.container.create_application_container") as mock_create_container,
        ):
            mock_bot_config.from_profile.return_value = mock_config
            mock_create_container.return_value = mock_container

            result = simulate_dry_run(checker)

        assert result is True
        assert any("container initialized" in s for s in checker.successes)
        assert any("TradingBot constructed" in s for s in checker.successes)
        assert any("engine available" in s for s in checker.successes)
        assert any("simulation passed" in s for s in checker.successes)

    def test_passes_when_engine_is_none(self) -> None:
        """Should pass even when engine is None (some configs)."""
        checker = PreflightCheck(profile="dev")

        mock_config = MagicMock()
        mock_container = MagicMock()
        mock_bot = MagicMock()
        mock_bot.engine = None  # No engine
        mock_container.create_bot.return_value = mock_bot

        with (
            patch("gpt_trader.orchestration.configuration.BotConfig") as mock_bot_config,
            patch("gpt_trader.app.container.create_application_container") as mock_create_container,
        ):
            mock_bot_config.from_profile.return_value = mock_config
            mock_create_container.return_value = mock_container

            result = simulate_dry_run(checker)

        assert result is True
        # Should not have "engine available" success since engine is None
        assert not any("engine available" in s for s in checker.successes)

    def test_fails_on_config_error(self) -> None:
        """Should fail when config creation fails."""
        checker = PreflightCheck(profile="dev")

        with patch("gpt_trader.orchestration.configuration.BotConfig") as mock_bot_config:
            mock_bot_config.from_profile.side_effect = Exception("Config error")

            result = simulate_dry_run(checker)

        assert result is False
        assert any("simulation failed" in e for e in checker.errors)

    def test_fails_on_container_error(self) -> None:
        """Should fail when container creation fails."""
        checker = PreflightCheck(profile="dev")

        mock_config = MagicMock()

        with (
            patch("gpt_trader.orchestration.configuration.BotConfig") as mock_bot_config,
            patch("gpt_trader.app.container.create_application_container") as mock_create_container,
        ):
            mock_bot_config.from_profile.return_value = mock_config
            mock_create_container.side_effect = Exception("Container init failed")

            result = simulate_dry_run(checker)

        assert result is False
        assert any("simulation failed" in e for e in checker.errors)

    def test_fails_on_bot_creation_error(self) -> None:
        """Should fail when bot creation fails."""
        checker = PreflightCheck(profile="dev")

        mock_config = MagicMock()
        mock_container = MagicMock()
        mock_container.create_bot.side_effect = Exception("Bot creation failed")

        with (
            patch("gpt_trader.orchestration.configuration.BotConfig") as mock_bot_config,
            patch("gpt_trader.app.container.create_application_container") as mock_create_container,
        ):
            mock_bot_config.from_profile.return_value = mock_config
            mock_create_container.return_value = mock_container

            result = simulate_dry_run(checker)

        assert result is False
        assert any("simulation failed" in e for e in checker.errors)

    def test_uses_correct_config_params(self) -> None:
        """Should create config with dry_run=True and mock_broker=True."""
        checker = PreflightCheck(profile="canary")

        mock_config = MagicMock()
        mock_container = MagicMock()
        mock_bot = MagicMock()
        mock_bot.engine = MagicMock()
        mock_container.create_bot.return_value = mock_bot

        with (
            patch("gpt_trader.orchestration.configuration.BotConfig") as mock_bot_config,
            patch("gpt_trader.app.container.create_application_container") as mock_create_container,
        ):
            mock_bot_config.from_profile.return_value = mock_config
            mock_create_container.return_value = mock_container

            simulate_dry_run(checker)

            # Verify config was created with correct params
            mock_bot_config.from_profile.assert_called_once_with(
                profile="canary", dry_run=True, mock_broker=True
            )

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        with patch("gpt_trader.orchestration.configuration.BotConfig") as mock_bot_config:
            mock_bot_config.from_profile.side_effect = Exception("Skip")
            simulate_dry_run(checker)

        captured = capsys.readouterr()
        assert "DRY-RUN SIMULATION" in captured.out

    def test_logs_profile_info_when_verbose(self, capsys: pytest.CaptureFixture) -> None:
        """Should log config info when verbose."""
        checker = PreflightCheck(profile="canary", verbose=True)

        mock_config = MagicMock()
        mock_container = MagicMock()
        mock_bot = MagicMock()
        mock_bot.engine = MagicMock()
        mock_container.create_bot.return_value = mock_bot

        with (
            patch("gpt_trader.orchestration.configuration.BotConfig") as mock_bot_config,
            patch("gpt_trader.app.container.create_application_container") as mock_create_container,
        ):
            mock_bot_config.from_profile.return_value = mock_config
            mock_create_container.return_value = mock_container

            simulate_dry_run(checker)

        captured = capsys.readouterr()
        assert "canary" in captured.out
