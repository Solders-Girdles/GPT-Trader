"""Unit tests for EnsembleStrategy signals."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.ensemble import (
    EnsembleStrategy,
    EnsembleStrategyConfig,
)


def test_initialization() -> None:
    strategy = EnsembleStrategy()
    assert len(strategy.signals) == 3
    assert strategy.combiner is not None


def test_trend_component_positive_in_uptrend() -> None:
    strategy = EnsembleStrategy()
    prices = [Decimal("100") + Decimal(i) for i in range(30)]

    decision = strategy.decide(
        symbol="BTC-USD",
        current_mark=prices[-1],
        position_state=None,
        recent_marks=prices,
        equity=Decimal("10000"),
        product=None,
    )

    components = (decision.indicators or {}).get("components", {})
    trend_component = components.get("trend_ma", {})
    assert trend_component.get("raw") > 0


def test_mean_reversion_component_positive_when_price_low() -> None:
    import math

    from gpt_trader.features.live_trade.signals.mean_reversion import MeanReversionSignalConfig

    config = EnsembleStrategyConfig(
        mean_reversion_config=MeanReversionSignalConfig(z_entry_threshold=1.0)
    )
    strategy = EnsembleStrategy(config=config)

    prices = [Decimal("100") + Decimal(str(10 * math.sin(i * 0.5))) for i in range(50)]
    prices[-1] = Decimal("80")

    decision = strategy.decide(
        symbol="BTC-USD",
        current_mark=prices[-1],
        position_state=None,
        recent_marks=prices,
        equity=Decimal("10000"),
        product=None,
    )

    components = (decision.indicators or {}).get("components", {})
    mr_component = components.get("mean_reversion_z", {})
    assert mr_component.get("raw") > 0
