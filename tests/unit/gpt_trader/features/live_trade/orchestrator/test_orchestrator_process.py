"""Tests for StrategyOrchestrator.process_symbol() control flow and errors."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.features.live_trade.orchestrator.orchestrator import StrategyOrchestrator
from gpt_trader.features.live_trade.strategies.perps_baseline import Action


class TestProcessSymbol:
    """Test process_symbol method control flow."""

    @pytest.fixture
    def mock_orchestrator(self) -> StrategyOrchestrator:
        """Create an orchestrator with mocked methods."""
        orchestrator = StrategyOrchestrator.__new__(StrategyOrchestrator)
        orchestrator._bot = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    async def test_process_symbol_skips_when_context_is_none(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that process_symbol returns early if _prepare_context returns None."""
        mock_orchestrator._prepare_context = AsyncMock(return_value=None)
        mock_orchestrator._resolve_decision = AsyncMock()
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._prepare_context.assert_called_once()
        mock_orchestrator._resolve_decision.assert_not_called()
        mock_orchestrator._record_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_calls_resolve_decision(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that process_symbol calls _resolve_decision with context."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.HOLD  # Not an execution action

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._resolve_decision.assert_called_once_with(mock_context)

    @pytest.mark.asyncio
    async def test_process_symbol_records_decision(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that process_symbol records decision."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.HOLD

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._record_decision.assert_called_once_with("BTC-PERP-USDC", mock_decision)

    @pytest.mark.asyncio
    async def test_process_symbol_with_balances(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that process_symbol passes balances to _prepare_context."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.HOLD

        mock_balances = [Mock(), Mock()]

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC", balances=mock_balances)

        args, kwargs = mock_orchestrator._prepare_context.call_args
        passed_balances = kwargs.get("balances")
        if passed_balances is None and len(args) > 1:
            passed_balances = args[1]
        assert passed_balances == mock_balances

    @pytest.mark.asyncio
    async def test_process_symbol_with_position_map(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that process_symbol passes position_map to _prepare_context."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.HOLD

        mock_position_map = {"BTC-PERP-USDC": Mock()}

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC", position_map=mock_position_map)

        args, kwargs = mock_orchestrator._prepare_context.call_args
        passed_position_map = kwargs.get("position_map")
        if passed_position_map is None and len(args) > 2:
            passed_position_map = args[2]
        assert passed_position_map == mock_position_map


class TestProcessSymbolErrorHandling:
    """Test error handling in process_symbol."""

    @pytest.fixture
    def mock_orchestrator(self) -> StrategyOrchestrator:
        """Create an orchestrator with mocked methods."""
        orchestrator = StrategyOrchestrator.__new__(StrategyOrchestrator)
        orchestrator._bot = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    async def test_process_symbol_handles_prepare_context_error(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that errors in _prepare_context are caught and logged."""
        mock_orchestrator._prepare_context = AsyncMock(
            side_effect=RuntimeError("Context preparation failed")
        )

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_handles_resolve_decision_error(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that errors in _resolve_decision are caught and logged."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(
            side_effect=ValueError("Decision resolution failed")
        )

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_handles_execution_error(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that errors in execute_decision are caught and logged."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.BUY

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()
        mock_orchestrator._bot.execute_decision.side_effect = RuntimeError("Order failed")

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._record_decision.assert_called_once_with("BTC-PERP-USDC", mock_decision)
        mock_orchestrator._bot.execute_decision.assert_called_once()
