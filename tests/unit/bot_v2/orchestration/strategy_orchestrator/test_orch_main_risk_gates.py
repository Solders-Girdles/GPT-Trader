"""Tests for StrategyOrchestrator risk gate validation and kill switch logic.

This module tests:
- Kill switch detection
- Volatility circuit breaker checks
- Market data staleness checks
- Kill switch early returns
- Warning message logging
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.live_trade.risk_runtime import (
    CircuitBreakerAction,
    CircuitBreakerOutcome,
)
from bot_v2.orchestration.strategy_orchestrator import (
    StrategyOrchestrator,
    SymbolProcessingContext,
)


class TestKillSwitchEngaged:
    """Test _kill_switch_engaged method."""

    def test_returns_true_when_enabled(self, orchestrator, mock_bot):
        """Test returns True when kill switch enabled."""
        mock_bot.risk_manager.config.kill_switch_enabled = True

        engaged = orchestrator._kill_switch_engaged()

        assert engaged is True

    def test_returns_false_when_disabled(self, orchestrator, mock_bot):
        """Test returns False when kill switch disabled."""
        mock_bot.risk_manager.config.kill_switch_enabled = False

        engaged = orchestrator._kill_switch_engaged()

        assert engaged is False


class TestRunRiskGates:
    """Test _run_risk_gates method."""

    def test_returns_true_when_all_gates_pass(self, orchestrator, mock_bot):
        """Test returns True when all risk gates pass."""
        marks = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=False, action=CircuitBreakerAction.NONE, reason=None
            )
        )
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=False)

        context = SymbolProcessingContext(
            symbol="BTC-PERP",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=list(marks),
            product=None,
        )

        result = orchestrator._run_risk_gates(context)

        assert result is True

    def test_returns_false_when_kill_switch_triggered(self, orchestrator, mock_bot):
        """Test returns False when volatility CB triggers kill switch."""
        marks = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=True,
                action=CircuitBreakerAction.KILL_SWITCH,
                reason="volatility spike",
            )
        )

        context = SymbolProcessingContext(
            symbol="BTC-PERP",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=list(marks),
            product=None,
        )

        result = orchestrator._run_risk_gates(context)

        assert result is False

    def test_returns_false_when_mark_stale(self, orchestrator, mock_bot):
        """Test returns False when market data is stale."""
        marks = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=False, action=CircuitBreakerAction.NONE, reason=None
            )
        )
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=True)

        context = SymbolProcessingContext(
            symbol="BTC-PERP",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=list(marks),
            product=None,
        )

        result = orchestrator._run_risk_gates(context)

        assert result is False


class TestKillSwitchLogic:
    """Test kill-switch engagement and emergency logic."""

    @pytest.mark.asyncio
    async def test_kill_switch_enabled_skips_processing(
        self, mock_bot, test_balance, test_position
    ) -> None:
        """Test that enabled kill switch prevents all processing."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Enable kill switch
        mock_bot.risk_manager.config.kill_switch_enabled = True

        with patch("bot_v2.orchestration.strategy_orchestrator.emit_metric"):
            await orchestrator.process_symbol(
                "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
            )

        # Should not execute any decisions when kill switch is enabled
        mock_bot.execute_decision.assert_not_called()
        # Should log kill switch warning (covered by kill switch engagement)

    def test_kill_switch_enabled_property_access(self, mock_bot) -> None:
        """Test kill switch configuration access patterns."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Test with kill switch disabled (default)
        mock_bot.risk_manager.config.kill_switch_enabled = False
        assert orchestrator._kill_switch_engaged() is False

        # Test with kill switch enabled
        mock_bot.risk_manager.config.kill_switch_enabled = True
        assert orchestrator._kill_switch_engaged() is True

        # Test when kill_switch_enabled attribute doesn't exist
        del mock_bot.risk_manager.config.kill_switch_enabled
        assert orchestrator._kill_switch_engaged() is False

    @pytest.mark.asyncio
    async def test_kill_switch_preparation_context_early_return(
        self, mock_bot, test_balance, test_position
    ) -> None:
        """Test that kill switch causes _prepare_context to return None early."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Enable kill switch
        mock_bot.risk_manager.config.kill_switch_enabled = True

        context = await orchestrator._prepare_context(
            "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
        )

        # Should return None when kill switch is engaged
        assert context is None

    @pytest.mark.asyncio
    async def test_kill_switch_logs_warning_message(
        self, mock_bot, test_balance, test_position
    ) -> None:
        """Test that kill switch engagement logs appropriate warning."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Enable kill switch
        mock_bot.risk_manager.config.kill_switch_enabled = True

        with patch("bot_v2.orchestration.strategy_orchestrator.logger") as mock_logger:
            await orchestrator._prepare_context(
                "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
            )

        # Should log warning about kill switch being enabled
        mock_logger.warning.assert_called_once_with("Kill switch enabled - skipping trading loop")
