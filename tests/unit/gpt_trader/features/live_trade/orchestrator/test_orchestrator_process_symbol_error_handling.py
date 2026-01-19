"""Tests for StrategyOrchestrator.process_symbol() error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.features.live_trade.orchestrator.orchestrator import StrategyOrchestrator
from gpt_trader.features.live_trade.strategies.perps_baseline import Action


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
