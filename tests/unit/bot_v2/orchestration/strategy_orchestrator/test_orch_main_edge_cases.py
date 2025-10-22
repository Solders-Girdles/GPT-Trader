"""Tests for StrategyOrchestrator edge cases and position validation.

This module tests:
- Invalid position fraction handling
- Execution error logging
- No marks warnings
- Zero equity handling
- Short position scenarios
- Position state attribute validation
- Context preparation failures
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.live_trade.risk_runtime import (
    CircuitBreakerAction,
    CircuitBreakerOutcome,
)
from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
)
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator


class TestStrategyOrchestratorEdgeCases:
    """Test edge cases and error handling scenarios in StrategyOrchestrator."""

    @pytest.mark.asyncio
    async def test_spot_profile_invalid_position_fraction_warning(self, orchestrator, mock_bot):
        """Test warning logged when spot profile has invalid position_fraction."""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.config.symbols = ["BTC-PERP"]
        mock_bot.config.perps_position_fraction = None

        # Mock spot profiles to return invalid fraction
        mock_spot_profiles = Mock()
        mock_spot_profiles.load.return_value = {"BTC-PERP": {"position_fraction": "invalid_value"}}

        orchestrator._spot_profiles = mock_spot_profiles

        # This should trigger the warning on lines 86-93
        with pytest.MonkeyPatch().context() as m:
            mock_logger = Mock()
            m.setattr("bot_v2.orchestration.strategy_orchestrator.logger", mock_logger)

            orchestrator.init_strategy()

            # Should log warning about invalid position_fraction
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert (
                "Invalid position_fraction=invalid_value for BTC-PERP; using default"
                in call_args[0][0] % call_args[0][1:]
            )
            assert call_args[1]["operation"] == "strategy_init"
            assert call_args[1]["stage"] == "spot_fraction"
            assert call_args[1]["symbol"] == "BTC-PERP"

    @pytest.mark.asyncio
    async def test_perps_invalid_position_fraction_warning(self, orchestrator, mock_bot):
        """Test warning logged when perps position_fraction is invalid."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.perps_position_fraction = "not_a_number"

        with pytest.MonkeyPatch().context() as m:
            mock_logger = Mock()
            m.setattr("bot_v2.orchestration.strategy_orchestrator.logger", mock_logger)

            orchestrator.init_strategy()

            # Should log warning about invalid PERPS_POSITION_FRACTION
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert (
                "Invalid PERPS_POSITION_FRACTION=not_a_number; using default"
                in call_args[0][0] % call_args[0][1:]
            )
            assert call_args[1]["operation"] == "strategy_init"
            assert call_args[1]["stage"] == "perps_fraction"

    @pytest.mark.asyncio
    async def test_process_symbol_execution_error_logging(
        self, orchestrator, mock_bot, test_balance
    ):
        """Test error logging when execution fails."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[])

        # Set up marks and strategy
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        strategy = Mock(spec=BaselinePerpsStrategy)
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide = Mock(return_value=decision)
        state.strategy = strategy

        product = Mock()
        mock_bot.get_product = Mock(return_value=product)
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=False)
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=False, action=CircuitBreakerAction.NONE, reason=None
            )
        )

        # Mock execute_decision to raise an exception
        mock_bot.execute_decision = Mock(side_effect=Exception("Execution failed"))

        with pytest.MonkeyPatch().context() as m:
            mock_logger = Mock()
            m.setattr("bot_v2.orchestration.strategy_orchestrator.logger", mock_logger)

            # Should not raise exception, but should log error
            await orchestrator.process_symbol("BTC-PERP")

            # Should log execution error with expected structure
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Error processing %s: %s" in call_args[0][0]
            assert call_args[1]["operation"] == "strategy_execute"
            assert call_args[1]["stage"] == "process_symbol"
            assert call_args[1]["symbol"] == "BTC-PERP"
            assert call_args[1]["exc_info"] is True

    @pytest.mark.asyncio
    async def test_prepare_context_no_marks_warning(self, orchestrator, mock_bot, test_balance):
        """Test warning logged when no marks are available."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[])

        # Clear marks to trigger warning
        state = mock_bot.runtime_state
        state.mark_windows.clear()

        with pytest.MonkeyPatch().context() as m:
            mock_logger = Mock()
            m.setattr("bot_v2.orchestration.strategy_orchestrator.logger", mock_logger)

            # Should return None and log warning
            result = await orchestrator._prepare_context("BTC-PERP", None, None)

            assert result is None
            mock_logger.warning.assert_called()
            # Check that any warning call mentions "No marks for BTC-PERP"
            warning_calls = mock_logger.warning.call_args_list
            assert any("No marks for BTC-PERP" in str(call) for call in warning_calls)

    def test_adjust_equity_zero_position_quantity(self, orchestrator):
        """Test equity adjustment with zero position quantity."""
        equity = Decimal("10000")
        position_quantity = Decimal("0")
        marks = [Decimal("50000")]
        symbol = "BTC-PERP"

        result = orchestrator._adjust_equity(equity, position_quantity, marks, symbol)

        # Should return original equity when position quantity is zero
        assert result == equity

    def test_adjust_equity_with_position_and_marks(self, orchestrator):
        """Test equity adjustment with position and marks."""
        equity = Decimal("10000")
        position_quantity = Decimal("2")  # Long position
        marks = [Decimal("50000"), Decimal("51000")]
        symbol = "BTC-PERP"

        result = orchestrator._adjust_equity(equity, position_quantity, marks, symbol)

        # Should adjust equity based on position value
        assert isinstance(result, Decimal)
        # For long position, should add position value to equity
        position_value = position_quantity * marks[-1]
        expected = equity + position_value
        assert result == expected

    def test_adjust_equity_short_position(self, orchestrator):
        """Test equity adjustment with short position."""
        equity = Decimal("10000")
        position_quantity = Decimal("-1")  # Short position
        marks = [Decimal("50000"), Decimal("49000")]
        symbol = "BTC-PERP"

        result = orchestrator._adjust_equity(equity, position_quantity, marks, symbol)

        # For short position, should handle differently
        assert isinstance(result, Decimal)

    def test_build_position_state_long_position(self, orchestrator, test_position):
        """Test building position state for long position."""
        test_position.side = "long"
        test_position.quantity = Decimal("2")
        positions_lookup = {"BTC-PERP": test_position}

        position_state, position_quantity = orchestrator._build_position_state(
            "BTC-PERP", positions_lookup
        )

        assert position_quantity == Decimal("2")
        assert position_state is not None

    def test_build_position_state_no_position(self, orchestrator):
        """Test building position state when no position exists."""
        positions_lookup = {}

        position_state, position_quantity = orchestrator._build_position_state(
            "BTC-PERP", positions_lookup
        )

        assert position_quantity == Decimal("0")
        assert position_state is None

    def test_get_marks_uses_window(self, orchestrator, mock_bot):
        """Test _get_marks uses mark window from runtime state."""
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        expected_marks = [Decimal("50000"), Decimal("51000"), Decimal("52000")]
        state.mark_windows["BTC-PERP"] = expected_marks

        result = orchestrator._get_marks("BTC-PERP")

        assert result == expected_marks

    def test_get_marks_empty_window(self, orchestrator, mock_bot):
        """Test _get_marks with empty mark window."""
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = []

        result = orchestrator._get_marks("BTC-PERP")

        assert result == []

    def test_record_decision(self, orchestrator, mock_bot):
        """Test decision recording in runtime state."""
        decision = Decision(action=Action.BUY, reason="test_signal")

        orchestrator._record_decision("BTC-PERP", decision)

        state = mock_bot.runtime_state
        assert "BTC-PERP" in state.last_decisions
        assert state.last_decisions["BTC-PERP"] == decision


class TestPositionStateBuildingAndValidation:
    """Test position state building and validation workflows."""

    def test_position_state_building_with_complete_position(self, mock_bot) -> None:
        """Test position state building with complete position data."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Create mock position with all required attributes
        mock_pos = Mock()
        mock_pos.symbol = "BTC-PERP"
        mock_pos.side = "long"
        mock_pos.entry_price = Decimal("50000")
        mock_pos.quantity = Decimal("2")

        positions_lookup = {"BTC-PERP": mock_pos}

        position_state, position_quantity = orchestrator._build_position_state(
            "BTC-PERP", positions_lookup
        )

        assert position_quantity == Decimal("2")
        assert position_state is not None
        assert position_state["quantity"] == Decimal("2")
        assert position_state["side"] == "long"
        assert position_state["entry"] == Decimal("50000")

    def test_position_state_building_missing_attributes(self, mock_bot) -> None:
        """Test position state building with missing position attributes."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Create mock position with minimal attributes
        mock_pos = Mock()
        mock_pos.symbol = "BTC-PERP"
        # Missing side, entry_price, quantity

        positions_lookup = {"BTC-PERP": mock_pos}

        position_state, position_quantity = orchestrator._build_position_state(
            "BTC-PERP", positions_lookup
        )

        # Should handle missing attributes gracefully
        assert position_quantity == Decimal("0")  # default from quantity_from
        assert position_state is not None
        assert position_state["quantity"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_ensure_positions_with_provided_map(self, mock_bot, test_position) -> None:
        """Test position lookup when map is provided."""
        orchestrator = StrategyOrchestrator(mock_bot)

        provided_positions = {"BTC-PERP": test_position}

        result = await orchestrator._ensure_positions(provided_positions)

        # Should return the provided map unchanged
        assert result == provided_positions
        # Should not call broker.list_positions
        mock_bot.broker.list_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_positions_fetches_from_broker(self, mock_bot, test_position) -> None:
        """Test position lookup when map is None and needs fetching."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock broker positions
        test_positions = [test_position]
        mock_bot.broker.list_positions.return_value = test_positions

        result = await orchestrator._ensure_positions(None)

        # Should call broker and return symbol-keyed dict
        mock_bot.broker.list_positions.assert_called_once()
        assert "BTC-PERP" in result

    @pytest.mark.asyncio
    async def test_prepare_context_no_marks_early_return(self, mock_bot, test_balance, test_position) -> None:
        """Test that missing marks causes _prepare_context to return None."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock empty marks
        with patch.object(orchestrator, "_get_marks", return_value=[]):
            with patch("bot_v2.orchestration.strategy_orchestrator.logger") as mock_logger:
                context = await orchestrator._prepare_context(
                    "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
                )

        # Should return None when no marks available
        assert context is None
        # Should log warning about missing marks
        mock_logger.warning.assert_called_once()
        assert "No marks for" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_prepare_context_zero_equity_early_return(self, mock_bot, test_balance, test_position) -> None:
        """Test that zero equity causes _prepare_context to return None."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock zero equity adjustment
        with patch.object(orchestrator, "_get_marks", return_value=[Decimal("50000")]):
            with patch.object(orchestrator, "_adjust_equity", return_value=Decimal("0")):
                with patch("bot_v2.orchestration.strategy_orchestrator.logger") as mock_logger:
                    context = await orchestrator._prepare_context(
                        "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
                    )

        # Should return None when equity is zero
        assert context is None
        # Should log error about no equity info
        mock_logger.error.assert_called_once()
        assert "No equity info for" in str(mock_logger.error.call_args)

    def test_extract_equity_with_usd_balance(self, mock_bot) -> None:
        """Test equity extraction with USD balance."""
        orchestrator = StrategyOrchestrator(mock_bot)

        usd_balance = Mock()
        usd_balance.asset = "USD"
        usd_balance.total = Decimal("1000")

        balances = [usd_balance]
        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("1000")

    def test_extract_equity_with_usdc_balance(self, mock_bot) -> None:
        """Test equity extraction with USDC balance."""
        orchestrator = StrategyOrchestrator(mock_bot)

        usdc_balance = Mock()
        usdc_balance.asset = "USDC"
        usdc_balance.total = Decimal("2000")

        balances = [usdc_balance]
        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("2000")

    def test_extract_equity_no_cash_assets(self, mock_bot) -> None:
        """Test equity extraction when no cash assets are available."""
        orchestrator = StrategyOrchestrator(mock_bot)

        btc_balance = Mock()
        btc_balance.asset = "BTC"
        btc_balance.total = Decimal("1")

        balances = [btc_balance]
        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("0")
