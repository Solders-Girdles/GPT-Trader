"""Comprehensive tests for StrategyOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from gpt_trader.features.live_trade.strategies.perps_baseline import Action
from gpt_trader.orchestration.strategy_orchestrator.orchestrator import StrategyOrchestrator


class TestStrategyOrchestratorClassStructure:
    """Test StrategyOrchestrator class structure."""

    def test_requires_context_attribute(self) -> None:
        """Test that requires_context is True."""
        assert StrategyOrchestrator.requires_context is True

    def test_inherits_from_mixins(self) -> None:
        """Test that StrategyOrchestrator inherits from expected mixins."""
        from gpt_trader.orchestration.strategy_orchestrator.context import ContextBuilderMixin
        from gpt_trader.orchestration.strategy_orchestrator.decision import DecisionEngineMixin
        from gpt_trader.orchestration.strategy_orchestrator.initialization import (
            StrategyInitializationMixin,
        )
        from gpt_trader.orchestration.strategy_orchestrator.spot_filters import SpotFiltersMixin

        assert issubclass(StrategyOrchestrator, SpotFiltersMixin)
        assert issubclass(StrategyOrchestrator, DecisionEngineMixin)
        assert issubclass(StrategyOrchestrator, ContextBuilderMixin)
        assert issubclass(StrategyOrchestrator, StrategyInitializationMixin)


class TestProcessSymbol:
    """Test process_symbol method."""

    @pytest.fixture
    def mock_orchestrator(self) -> StrategyOrchestrator:
        """Create an orchestrator with mocked methods."""
        orchestrator = StrategyOrchestrator.__new__(StrategyOrchestrator)

        # Mock _bot for execute_decision
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
    async def test_process_symbol_executes_sell_decision(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that SELL decision triggers execution."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.SELL

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._bot.execute_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_symbol_executes_close_decision(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that CLOSE decision triggers execution."""
        mock_context = Mock()
        mock_context.product = Mock()
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.CLOSE

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        mock_orchestrator._bot.execute_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_symbol_does_not_execute_hold(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that HOLD decision does not trigger execution."""
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

        mock_orchestrator._bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_skips_execution_when_product_missing(
        self, mock_orchestrator: StrategyOrchestrator
    ) -> None:
        """Test that execution is skipped when product is None."""
        mock_context = Mock()
        mock_context.product = None  # Missing product
        mock_context.marks = [Mock()]
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.BUY  # Would normally execute

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        # Decision should still be recorded
        mock_orchestrator._record_decision.assert_called_once()
        # But execution should be skipped
        mock_orchestrator._bot.execute_decision.assert_not_called()

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

        call_args = mock_orchestrator._prepare_context.call_args
        assert call_args[1].get("balances") == mock_balances or call_args[0][1] == mock_balances

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

        mock_orchestrator._prepare_context.assert_called_once()


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

        # Should not raise - error should be caught and logged
        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        # Verify execution was not attempted
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

        # Should not raise
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

        # Should not raise
        await mock_orchestrator.process_symbol("BTC-PERP-USDC")


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
        mock_context.marks = [mark_old, mark_latest]  # Latest is at index -1
        mock_context.position_state = Mock()

        mock_decision = Mock()
        mock_decision.action = Action.BUY

        mock_orchestrator._prepare_context = AsyncMock(return_value=mock_context)
        mock_orchestrator._resolve_decision = AsyncMock(return_value=mock_decision)
        mock_orchestrator._record_decision = Mock()

        await mock_orchestrator.process_symbol("BTC-PERP-USDC")

        call_args = mock_orchestrator._bot.execute_decision.call_args
        # Third argument should be the latest mark
        assert call_args[0][2] is mark_latest
