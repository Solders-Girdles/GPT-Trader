"""Tests for VWAPSignal customization and metadata."""

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.signals.protocol import StrategyContext
from gpt_trader.features.live_trade.signals.vwap import (
    VWAPSignal,
    VWAPSignalConfig,
)
from gpt_trader.features.live_trade.strategies.base import MarketDataContext


@pytest.fixture
def base_context() -> StrategyContext:
    """Create a basic strategy context."""
    return StrategyContext(
        symbol="BTC-USD",
        current_mark=Decimal("50000"),
        position_state=None,
        recent_marks=[Decimal("49900"), Decimal("49950"), Decimal("50000")],
        equity=Decimal("10000"),
        product=None,
        market_data=None,
    )


def test_custom_thresholds(base_context: StrategyContext) -> None:
    """Test signal with custom thresholds."""
    custom_signal = VWAPSignal(
        VWAPSignalConfig(
            deviation_threshold=0.02,  # 2%
            min_trades=10,
        )
    )

    # 1.5% deviation would trigger default but not custom
    base_context.market_data = MarketDataContext(
        trade_volume_stats={"count": 50, "vwap": Decimal("50750")}
    )
    result = custom_signal.generate(base_context)

    assert result.strength == 0.0
    assert result.metadata["reason"] == "near_vwap"


def test_metadata_contains_expected_fields(base_context: StrategyContext) -> None:
    """Test that metadata contains all expected fields."""
    signal = VWAPSignal()
    base_context.market_data = MarketDataContext(
        trade_volume_stats={
            "count": 50,
            "vwap": Decimal("51000"),
            "volume": Decimal("1000"),
            "avg_size": Decimal("20"),
        }
    )
    result = signal.generate(base_context)

    assert "vwap" in result.metadata
    assert "current_price" in result.metadata
    assert "deviation_pct" in result.metadata
    assert "trade_count" in result.metadata
    assert "volume" in result.metadata
    assert "avg_size" in result.metadata
    assert "reason" in result.metadata
