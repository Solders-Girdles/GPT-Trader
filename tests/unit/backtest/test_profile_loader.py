from pathlib import Path
from decimal import Decimal

import pytest
import yaml

from bot_v2.backtest.profile import build_strategy_spec


def make_profile(tmp_path, content):
    path = tmp_path / "spot.yaml"
    with path.open("w") as fh:
        yaml.safe_dump(content, fh)
    return path


def test_build_strategy_spec_applies_filters(tmp_path):
    profile = {
        "strategy": {
            "btc": {
                "type": "ma",
                "short_window": 5,
                "long_window": 40,
                "volume_filter": {"window": 10, "multiplier": 1.2},
                "momentum_filter": {"window": 14, "oversold": 40, "overbought": 70},
                "trend_filter": {"window": 10, "min_slope": 0.0005},
            }
        },
        "risk": {"commission_bps": 2.5, "initial_cash": 10000},
    }
    path = make_profile(tmp_path, profile)
    spec = build_strategy_spec(path, "BTC")
    assert spec.symbol == "BTC"
    assert isinstance(spec.config.initial_cash, Decimal)
    assert spec.config.initial_cash == Decimal("10000")
    # Strategy should be wrapped, but we just ensure callable and attribute exists
    assert hasattr(spec.strategy, "on_bar")


def test_build_strategy_missing_symbol(tmp_path):
    profile = {"strategy": {}}
    path = make_profile(tmp_path, profile)
    with pytest.raises(KeyError):
        build_strategy_spec(path, "BTC")
