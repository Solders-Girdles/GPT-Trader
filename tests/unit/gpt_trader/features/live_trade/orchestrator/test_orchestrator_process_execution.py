"""Tests for StrategyOrchestrator.process_symbol() execution behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.features.live_trade.orchestrator.orchestrator import StrategyOrchestrator
from gpt_trader.features.live_trade.strategies.perps_baseline import Action


class TestProcessSymbolExecution:
    """Test process_symbol execution decisions."""

    @pytest.fixture
    def mock_orchestrator(self) -> StrategyOrchestrator:
        """Create an orchestrator with mocked methods."""
        orchestrator = StrategyOrchestrator.__new__(StrategyOrchestrator)
        orchestrator._bot = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    async def test_process_symbol_executes_buy_decision(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that BUY decision triggers execution."""
        mock_product = Mock()
        mock_mark = Mock()
        mock_position_state = Mock()

        mock_context = Mock()
        mock_context.product = mock_product
        mock_context.marks = [mock_mark]
        mock_context.position_state = mock_position_state

        mock_decision = Mock()
        mock_decision.action = Action.BUY

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._bot.execute_decision.assert_called_once_with(
            "BTC-PERP-USDC",
            mock_decision,
            mock_mark,
            mock_product,
            mock_position_state,
        )

    @pytest.mark.asyncio
    async def test_process_symbol_skips_execution_when_product_missing(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that execution is skipped when product is None."""
        mock_context = Mock()
        mock_context.product = None
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.BUY

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._record_decision.assert_called_once()
        mock_orchestrator._bot.execute_decision.assert_not_called()


class TestProcessSymbolActions:
    """Test all Action types in process_symbol."""

    @pytest.fixture
    def mock_orchestrator(self) -> StrategyOrchestrator:
        """Create an orchestrator with mocked methods."""
        orchestrator = StrategyOrchestrator.__new__(StrategyOrchestrator)
        orchestrator._bot = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "action,should_execute",
        [
            (Action.BUY, True),
            (Action.SELL, True),
            (Action.CLOSE, True),
            (Action.HOLD, False),
        ],
    )
    async def test_action_execution_mapping(
        self,
        mock_orchestrator: StrategyOrchestrator,
        action: Action,
        should_execute: bool,
    ) -> None:
        """Test that only BUY, SELL, CLOSE trigger execution."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = action

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        if should_execute:
            mock_orchestrator._bot.execute_decision.assert_called_once()
        else:
            mock_orchestrator._bot.execute_decision.assert_not_called()


class TestProcessSymbolMultipleMarks:
    """Test process_symbol with multiple marks."""

    @pytest.fixture
    def mock_orchestrator(self) -> StrategyOrchestrator:
        """Create an orchestrator with mocked methods."""
        orchestrator = StrategyOrchestrator.__new__(StrategyOrchestrator)
        orchestrator._bot = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    async def test_uses_latest_mark_for_execution(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that the latest mark (marks[-1]) is used for execution."""
        mark_old = Mock(name="mark_old")
        mark_latest = Mock(name="mark_latest")

        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [mark_old, mark_latest]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.BUY

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        call_args = mock_orchestrator._bot.execute_decision.call_args
        assert call_args[0][2] is mark_latest
