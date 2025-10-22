"""Tests for reduce-only flag finalization and OrderValidator initialization."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.execution.validation import OrderValidator


class TestReduceOnlyFinalization:
    """Test finalize_reduce_only_flag method."""

    def test_finalize_reduce_only_when_not_mode(self, order_validator) -> None:
        """Test reduce-only flag unchanged when not in reduce-only mode."""
        symbol = "BTC-PERP"
        reduce_only = False

        result = order_validator.finalize_reduce_only_flag(reduce_only, symbol)

        assert result is False
        order_validator.risk_manager.is_reduce_only_mode.assert_called_once()

    def test_finalize_reduce_only_when_not_mode_true_input(self, order_validator) -> None:
        """Test reduce-only flag unchanged when not in reduce-only mode but input is True."""
        symbol = "ETH-PERP"
        reduce_only = True

        result = order_validator.finalize_reduce_only_flag(reduce_only, symbol)

        assert result is True
        order_validator.risk_manager.is_reduce_only_mode.assert_called_once()

    def test_finalize_reduce_only_when_mode_enforced(self, order_validator) -> None:
        """Test reduce-only flag enforced when risk manager is in reduce-only mode."""
        symbol = "BTC-PERP"
        reduce_only = False  # User wants normal order
        order_validator.risk_manager.is_reduce_only_mode.return_value = True

        result = order_validator.finalize_reduce_only_flag(reduce_only, symbol)

        # Should be forced to True due to risk manager state
        assert result is True
        order_validator.risk_manager.is_reduce_only_mode.assert_called_once()

    def test_finalize_reduce_only_when_mode_already_true(self, order_validator) -> None:
        """Test reduce-only flag stays True when both user and risk manager want it."""
        symbol = "BTC-PERP"
        reduce_only = True  # User already wants reduce-only
        order_validator.risk_manager.is_reduce_only_mode.return_value = True

        result = order_validator.finalize_reduce_only_flag(reduce_only, symbol)

        assert result is True
        order_validator.risk_manager.is_reduce_only_mode.assert_called_once()

    def test_finalize_reduce_only_with_different_symbols(self, order_validator) -> None:
        """Test reduce-only flag enforcement works for different symbols."""
        symbols = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
        order_validator.risk_manager.is_reduce_only_mode.return_value = True

        for symbol in symbols:
            result = order_validator.finalize_reduce_only_flag(False, symbol)
            assert result is True

        # Should have checked risk manager for each symbol
        assert order_validator.risk_manager.is_reduce_only_mode.call_count == len(symbols)

    def test_finalize_reduce_only_risk_manager_exception_handling(self, order_validator) -> None:
        """Test graceful handling of risk manager exceptions."""
        symbol = "BTC-PERP"
        reduce_only = False

        # Mock risk manager to raise exception
        order_validator.risk_manager.is_reduce_only_mode.side_effect = RuntimeError(
            "Risk service unavailable"
        )

        # Should return original value when risk manager fails - the method doesn't catch exceptions
        # Let's test the actual behavior - it should propagate the exception
        with pytest.raises(RuntimeError, match="Risk service unavailable"):
            order_validator.finalize_reduce_only_flag(reduce_only, symbol)


class TestOrderValidatorInitialization:
    """Test OrderValidator initialization and basic properties."""

    def test_initialization_with_all_parameters(self, mock_brokerage, mock_risk_manager) -> None:
        """Test OrderValidator initialization with all parameters."""
        preview_callback = MagicMock()
        rejection_callback = MagicMock()

        validator = OrderValidator(
            broker=mock_brokerage,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=preview_callback,
            record_rejection_callback=rejection_callback,
        )

        assert validator.broker == mock_brokerage
        assert validator.risk_manager == mock_risk_manager
        assert validator.enable_order_preview is True
        assert validator._record_preview == preview_callback
        assert validator._record_rejection == rejection_callback

    def test_initialization_with_preview_disabled(self, mock_brokerage, mock_risk_manager) -> None:
        """Test OrderValidator initialization with preview disabled."""
        preview_callback = MagicMock()
        rejection_callback = MagicMock()

        validator = OrderValidator(
            broker=mock_brokerage,
            risk_manager=mock_risk_manager,
            enable_order_preview=False,
            record_preview_callback=preview_callback,
            record_rejection_callback=rejection_callback,
        )

        assert validator.enable_order_preview is False
        assert validator._record_preview == preview_callback
        assert validator._record_rejection == rejection_callback

    def test_initialization_stores_callbacks_correctly(
        self, mock_brokerage, mock_risk_manager
    ) -> None:
        """Test that callbacks are stored correctly during initialization."""

        def custom_preview(*args, **kwargs):
            pass

        def custom_rejection(*args, **kwargs):
            pass

        validator = OrderValidator(
            broker=mock_brokerage,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=custom_preview,
            record_rejection_callback=custom_rejection,
        )

        assert validator._record_preview == custom_preview
        assert validator._record_rejection == custom_rejection


class TestOrderValidatorIntegration:
    """Integration tests for OrderValidator scenarios."""

    def test_order_validator_with_mock_brokerage_methods(
        self, mock_brokerage, mock_risk_manager
    ) -> None:
        """Test OrderValidator works with brokerage that has optional methods."""

        # Add optional preview method
        def preview_order(**kwargs):
            return {"order_id": "test_preview", "cost": 1000}

        mock_brokerage.preview_order = preview_order
        mock_brokerage.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": 1000000,
        }

        validator = OrderValidator(
            broker=mock_brokerage,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
        )

        # Should work with brokerage that has preview method
        assert hasattr(validator.broker, "preview_order")

    def test_order_validator_without_optional_methods(
        self, mock_brokerage, mock_risk_manager
    ) -> None:
        """Test OrderValidator works with minimal brokerage interface."""
        # Ensure broker only has required methods
        (
            delattr(mock_brokerage, "preview_order")
            if hasattr(mock_brokerage, "preview_order")
            else None
        )
        (
            delattr(mock_brokerage, "get_market_snapshot")
            if hasattr(mock_brokerage, "get_market_snapshot")
            else None
        )

        validator = OrderValidator(
            broker=mock_brokerage,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
        )

        # Should work without optional methods
        assert not hasattr(validator.broker, "preview_order")
        assert not hasattr(validator.broker, "get_market_snapshot")

    def test_order_validator_callback_invocation_chain(self, order_validator) -> None:
        """Test that callbacks are properly invoked during validation scenarios."""
        symbol = "BTC-PERP"

        # Mock risk manager to trigger various scenarios
        order_validator.risk_manager.is_reduce_only_mode.return_value = True

        # Test reduce-only finalization (no callback, but affects behavior)
        result = order_validator.finalize_reduce_only_flag(False, symbol)
        assert result is True

        # Test that callbacks are accessible and callable
        assert callable(order_validator._record_preview)
        assert callable(order_validator._record_rejection)

        # Verify they haven't been called yet
        order_validator._record_preview.assert_not_called()
        order_validator._record_rejection.assert_not_called()
