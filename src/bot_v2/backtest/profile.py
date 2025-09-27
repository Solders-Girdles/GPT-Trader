from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional

import yaml

from .spot import (
    BollingerMeanReversionStrategy,
    MomentumOscillatorStrategy,
    MovingAverageCrossStrategy,
    SpotBacktestConfig,
    SpotBacktester,
    TrendStrengthStrategy,
    VolatilityFilteredStrategy,
    VolumeConfirmationStrategy,
    load_candles_from_parquet,
)


@dataclass
class StrategySpec:
    symbol: str
    strategy: object
    config: SpotBacktestConfig


def _decimal(value: Optional[float | str | int]) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(str(value))


def build_strategy_from_section(section: Dict) -> object:
    base_type = section.get("type", "ma").lower()
    if base_type == "ma":
        short = int(section.get("short_window", 5))
        long = int(section.get("long_window", 40))
        strategy: object = MovingAverageCrossStrategy(short_window=short, long_window=long)
    elif base_type == "bollinger":
        strategy = BollingerMeanReversionStrategy(
            window=int(section.get("window", 20)),
            num_std=_decimal(section.get("num_std", 2)) or Decimal("2"),
        )
    else:
        raise ValueError(f"Unsupported strategy type '{base_type}'")

    vol_cfg = section.get("volatility_filter")
    if isinstance(vol_cfg, dict):
        strategy = VolatilityFilteredStrategy(
            base_strategy=strategy,
            window=int(vol_cfg.get("window", 14)),
            min_vol=_decimal(vol_cfg.get("min_vol", 0)),
            max_vol=_decimal(vol_cfg.get("max_vol", 1)),
        )

    volma_cfg = section.get("volume_filter")
    if isinstance(volma_cfg, dict):
        strategy = VolumeConfirmationStrategy(
            base_strategy=strategy,
            window=int(volma_cfg.get("window", 10)),
            multiplier=_decimal(volma_cfg.get("multiplier", 1)) or Decimal("1"),
        )

    momentum_cfg = section.get("momentum_filter")
    if isinstance(momentum_cfg, dict):
        strategy = MomentumOscillatorStrategy(
            base_strategy=strategy,
            window=int(momentum_cfg.get("window", 14)),
            overbought=_decimal(momentum_cfg.get("overbought", 70)) or Decimal("70"),
            oversold=_decimal(momentum_cfg.get("oversold", 30)) or Decimal("30"),
        )

    trend_cfg = section.get("trend_filter")
    if isinstance(trend_cfg, dict):
        strategy = TrendStrengthStrategy(
            base_strategy=strategy,
            window=int(trend_cfg.get("window", 10)),
            min_slope=_decimal(trend_cfg.get("min_slope", 0)) or Decimal("0"),
        )

    return strategy


def load_profile(path: Path) -> Dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def build_strategy_spec(path: Path, symbol: str) -> StrategySpec:
    profile = load_profile(path)
    strategies = profile.get("strategy", {})
    keys = [symbol, symbol.lower(), symbol.upper()]
    if "-" in symbol:
        base = symbol.split("-")[0]
        keys.extend([base, base.lower(), base.upper()])
    strategy_section = None
    for key in keys:
        if key in strategies:
            strategy_section = strategies[key]
            break
    if not strategy_section:
        raise KeyError(f"Strategy config for {symbol} not found in {path}")

    strategy = build_strategy_from_section(strategy_section)
    risk_cfg = profile.get("risk", {})
    config = SpotBacktestConfig(
        initial_cash=Decimal(str(risk_cfg.get("initial_cash", 10000))),
        commission_bps=Decimal(str(risk_cfg.get("commission_bps", 2.5))),
    )
    return StrategySpec(symbol=symbol, strategy=strategy, config=config)


def run_profile_backtest(profile_path: Path, symbol: str, parquet_path: Path):
    spec = build_strategy_spec(profile_path, symbol)
    bars = load_candles_from_parquet(parquet_path)
    backtester = SpotBacktester(bars, spec.strategy, spec.config)
    return backtester.run()
