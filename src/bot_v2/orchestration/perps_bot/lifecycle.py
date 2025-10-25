"""Lifecycle management helpers."""

from __future__ import annotations

from typing import Any, cast

from bot_v2.logging import correlation_context


class PerpsBotLifecycleMixin:
    """Trading lifecycle entry points."""

    async def run(self, single_cycle: bool = False) -> None:
        # Create correlation context for the entire bot run
        with correlation_context(operation="bot_run", bot_id=self.bot_id):
            await self.lifecycle_manager.run(single_cycle)

    async def run_cycle(self) -> None:
        # Create correlation context for each trading cycle
        with correlation_context(operation="trading_cycle", bot_id=self.bot_id):
            await self.strategy_coordinator.run_cycle()

    async def _fetch_current_state(self) -> dict[str, Any]:
        state = await self.strategy_coordinator._fetch_current_state()
        return cast(dict[str, Any], state)

    async def _validate_configuration_and_handle_drift(self, current_state: dict[str, Any]) -> bool:
        result = await self.strategy_coordinator._validate_configuration_and_handle_drift(
            current_state
        )
        return bool(result)

    async def _execute_trading_cycle(self, trading_state: dict[str, Any]) -> None:
        await self.strategy_coordinator._execute_trading_cycle(trading_state)

    async def update_marks(self) -> None:
        await self.strategy_coordinator.update_marks()

    async def shutdown(self) -> None:
        # Create correlation context for shutdown
        with correlation_context(operation="bot_shutdown", bot_id=self.bot_id):
            await self.lifecycle_manager.shutdown()
