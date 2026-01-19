from __future__ import annotations

from gpt_trader.app.config import BotConfig
from gpt_trader.features.live_trade.factory import create_strategy
from gpt_trader.features.live_trade.strategies.ensemble import (
    EnsembleStrategy,
    EnsembleStrategyConfig,
)


def test_create_strategy_ensemble_parses_dict_config() -> None:
    config = BotConfig(
        symbols=["BTC-USD"],
        strategy_type="ensemble",
        ensemble_config={
            "buy_threshold": 0.1,
            "combiner_config": {"adx_period": 14},
        },
    )

    strategy = create_strategy(config)

    assert isinstance(strategy, EnsembleStrategy)
    assert isinstance(strategy.config, EnsembleStrategyConfig)
    assert strategy.config.buy_threshold == 0.1
    assert strategy.config.combiner_config.adx_period == 14


def test_create_strategy_ensemble_falls_back_on_invalid_dict_config() -> None:
    config = BotConfig(
        symbols=["BTC-USD"],
        strategy_type="ensemble",
        ensemble_config={"unknown_field": 1},
    )

    strategy = create_strategy(config)

    assert isinstance(strategy, EnsembleStrategy)
    assert isinstance(strategy.config, EnsembleStrategyConfig)
    assert strategy.config.buy_threshold == EnsembleStrategyConfig().buy_threshold
