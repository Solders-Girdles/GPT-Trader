"""Tests for risk configuration preflight checks."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.checks.risk import check_risk_configuration
from gpt_trader.preflight.core import PreflightCheck


class TestCheckRiskConfiguration:
    """Test risk configuration validation."""

    def test_passes_with_valid_config(self) -> None:
        """Should pass when all risk parameters are within bounds."""
        checker = PreflightCheck(profile="dev")

        mock_config = MagicMock()
        mock_config.max_leverage = 3
        mock_config.daily_loss_limit = Decimal("500")
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is True
        assert any("Max leverage" in s for s in checker.successes)

    def test_fails_with_unsafe_leverage(self) -> None:
        """Should fail when leverage is outside safe range."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 15  # Outside 1-10
        mock_config.daily_loss_limit = Decimal("500")
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is False
        assert any("UNSAFE" in e for e in checker.errors)

    def test_fails_with_zero_daily_loss_limit(self) -> None:
        """Should fail when daily loss limit is zero or negative."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 3
        mock_config.daily_loss_limit = Decimal("0")  # Invalid
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is False

    def test_fails_with_low_liquidation_buffer(self) -> None:
        """Should fail when liquidation buffer is below 10%."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 3
        mock_config.daily_loss_limit = Decimal("500")
        mock_config.min_liquidation_buffer_pct = Decimal("0.05")  # Below 0.10
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is False

    def test_fails_with_excessive_position_size(self) -> None:
        """Should fail when position size exceeds 25%."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 3
        mock_config.daily_loss_limit = Decimal("500")
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.50")  # Above 0.25
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is False

    def test_fails_with_slippage_guard_out_of_range(self) -> None:
        """Should fail when slippage guard is outside 10-100 bps."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 3
        mock_config.daily_loss_limit = Decimal("500")
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 5  # Below 10
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is False

    def test_warns_when_kill_switch_enabled(self) -> None:
        """Should warn when kill switch is enabled."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 3
        mock_config.daily_loss_limit = Decimal("500")
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = True  # Kill switch on
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is True  # Still passes, just warns
        assert any("Kill switch" in w for w in checker.warnings)

    def test_warns_when_reduce_only_mode(self) -> None:
        """Should warn when reduce-only mode is enabled."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 3
        mock_config.daily_loss_limit = Decimal("500")
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = True  # Reduce-only on

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is True
        assert any("Reduce-only" in w for w in checker.warnings)

    def test_warns_on_high_daily_loss_limit(self) -> None:
        """Should warn when daily loss limit exceeds $1000."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 3
        mock_config.daily_loss_limit = Decimal("5000")  # High limit
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is True
        assert any("seems high" in w for w in checker.warnings)

    def test_warns_on_aggressive_leverage(self) -> None:
        """Should warn when leverage exceeds 5x."""
        checker = PreflightCheck(profile="prod")

        mock_config = MagicMock()
        mock_config.max_leverage = 8  # High but within 1-10
        mock_config.daily_loss_limit = Decimal("500")
        mock_config.min_liquidation_buffer_pct = Decimal("0.15")
        mock_config.max_position_pct_per_symbol = Decimal("0.10")
        mock_config.slippage_guard_bps = 50
        mock_config.kill_switch_enabled = False
        mock_config.reduce_only_mode = False

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.return_value = mock_config
            result = check_risk_configuration(checker)

        assert result is True
        assert any("aggressive" in w for w in checker.warnings)

    def test_fails_on_exception(self) -> None:
        """Should fail when config loading throws an exception."""
        checker = PreflightCheck(profile="prod")

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.side_effect = Exception("Config error")
            result = check_risk_configuration(checker)

        assert result is False
        assert any("Failed to validate" in e for e in checker.errors)

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        with patch("gpt_trader.orchestration.configuration.RiskConfig") as mock_risk_config:
            mock_risk_config.from_env.side_effect = Exception("Skip")
            check_risk_configuration(checker)

        captured = capsys.readouterr()
        assert "RISK MANAGEMENT" in captured.out
