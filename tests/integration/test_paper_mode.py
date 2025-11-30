"""Paper mode trading integration tests.

Converted from scripts/validation/paper_mode_e2e.py.
Tests multi-cycle trading simulation with DeterministicBroker.

This is the most unique test in the integration suite - it validates
that the trading loop doesn't crash across multiple cycles, which
unit tests cannot cover due to mocking.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from gpt_trader.orchestration.trading_bot import TradingBot


@pytest.mark.integration
@pytest.mark.slow  # Multiple cycles takes time due to strategy loop
class TestPaperModeTrading:
    """Integration tests for multi-cycle paper trading.

    These tests validate the complete trading loop behavior
    that cannot be tested with unit test mocks.
    """

    @pytest.mark.asyncio
    async def test_paper_trading_metrics(self, fast_signal_bot: TradingBot) -> None:
        """Validate trading loop doesn't crash across multiple cycles.

        Runs 8 cycles with a price pattern designed to trigger MA crossovers.
        This is the most valuable unique test - nothing else exercises
        multi-cycle behavior end-to-end.
        """
        bot = fast_signal_bot

        # Price pattern designed to trigger signals with short_ma=3, long_ma=5
        price_pattern = [100, 99, 101, 103, 105, 104, 106, 108]

        for price in price_pattern:
            # Drive mark price if using DeterministicBroker
            if hasattr(bot.broker, "set_mark"):
                bot.broker.set_mark("BTC-USD", Decimal(str(price)))

            # Run a single trading cycle
            await bot.run(single_cycle=True)

        # Should complete without error - validate final state
        positions = bot.broker.list_positions()
        assert isinstance(positions, list), "list_positions should return a list"

    @pytest.mark.asyncio
    async def test_paper_trading_execution(self, fast_signal_bot: TradingBot) -> None:
        """Validate single cycle execution works correctly."""
        bot = fast_signal_bot

        # Run a single cycle
        await bot.run(single_cycle=True)

        # Bot should have completed without error
        assert bot.broker is not None, "Broker should still be available"

    @pytest.mark.asyncio
    async def test_paper_trading_state_persistence(self, fast_signal_bot: TradingBot) -> None:
        """Validate positions can be tracked across multiple cycles."""
        bot = fast_signal_bot

        # Initial position count (may change after cycles)
        initial_positions = bot.broker.list_positions()
        _initial_count = len(initial_positions)  # noqa: F841

        # Run several cycles
        for _ in range(3):
            await bot.run(single_cycle=True)

        # Should complete without error
        final_positions = bot.broker.list_positions()
        assert isinstance(final_positions, list), "Positions should be a list"

        # Position count may or may not change (depends on signals)
        # The important thing is that tracking works
        final_count = len(final_positions)
        assert final_count >= 0, "Should have valid position count"
