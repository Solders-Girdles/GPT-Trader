import datetime as dt
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.risk_calculations import (
    effective_mmr,
    effective_symbol_leverage_cap,
    evaluate_daytime_window,
)


class _Config:
    max_leverage = 5
    leverage_max_per_symbol = {"BTC-PERP": 10}
    day_leverage_max_per_symbol = {"BTC-PERP": 6}
    night_leverage_max_per_symbol = {"BTC-PERP": 3}
    default_maintenance_margin_rate = 0.01
    day_mmr_per_symbol = {"BTC-PERP": 0.015}
    night_mmr_per_symbol = {"BTC-PERP": 0.02}
    daytime_start_utc = "09:00"
    daytime_end_utc = "17:00"


class _Logger:
    def __init__(self):
        self.debug_calls = []

    def debug(self, msg, *args, **kwargs):
        self.debug_calls.append(msg % args if args else msg)


@pytest.fixture
def config():
    return _Config()


def test_evaluate_daytime_window_day(config):
    now = dt.datetime(2025, 1, 6, 10, 0, tzinfo=dt.timezone.utc)
    assert evaluate_daytime_window(config, now) is True


def test_evaluate_daytime_window_night(config):
    now = dt.datetime(2025, 1, 6, 20, 0, tzinfo=dt.timezone.utc)
    assert evaluate_daytime_window(config, now) is False


def test_effective_leverage_cap_respects_day_schedule(config):
    logger = _Logger()
    cap = effective_symbol_leverage_cap(
        "BTC-PERP",
        config,
        now=dt.datetime(2025, 1, 6, 10, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=None,
        logger=logger,
    )
    assert cap == 6


def test_effective_leverage_cap_provider_override(config):
    logger = _Logger()
    cap = effective_symbol_leverage_cap(
        "BTC-PERP",
        config,
        now=dt.datetime(2025, 1, 6, 22, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=lambda symbol: {"max_leverage": 2},
        logger=logger,
    )
    assert cap == 2


def test_effective_mmr_prefers_provider(config):
    logger = _Logger()
    mmr = effective_mmr(
        "BTC-PERP",
        config,
        now=dt.datetime(2025, 1, 6, 22, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=lambda symbol: {"maintenance_margin_rate": 0.03},
        logger=logger,
    )
    assert mmr == Decimal("0.03")


def test_effective_mmr_falls_back_to_schedule(config):
    logger = _Logger()
    mmr = effective_mmr(
        "BTC-PERP",
        config,
        now=dt.datetime(2025, 1, 6, 21, 0, tzinfo=dt.timezone.utc),
        risk_info_provider=None,
        logger=logger,
    )
    assert mmr == Decimal("0.02")
