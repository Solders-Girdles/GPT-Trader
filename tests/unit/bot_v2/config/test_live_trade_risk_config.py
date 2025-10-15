from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from bot_v2.config.live_trade_config import RiskConfig


class StubSettings:
    def __init__(self, raw_env: dict[str, str]) -> None:
        self.raw_env = raw_env


def test_risk_config_alias_maps_exposure() -> None:
    config = RiskConfig(max_total_exposure_pct=0.6)
    assert config.max_exposure_pct == 0.6


def test_risk_config_from_env_parses_values() -> None:
    env = {
        "RISK_MAX_LEVERAGE": "7",
        "RISK_LEVERAGE_MAX_PER_SYMBOL": "BTC-PERP:5,ETH-PERP:3",
        "RISK_DAYTIME_START_UTC": "08:00",
        "RISK_DAYTIME_END_UTC": "18:00",
        "RISK_DAY_LEVERAGE_MAX_PER_SYMBOL": "BTC-PERP:4",
        "RISK_DAY_MMR_PER_SYMBOL": "BTC-PERP:0.01",
        "RISK_NIGHT_MMR_PER_SYMBOL": "BTC-PERP:0.015",
        "RISK_MAX_NOTIONAL_PER_SYMBOL": "BTC-PERP:10000,ETH-PERP:5000",
        "RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION": "false",
        "RISK_MIN_LIQUIDATION_BUFFER_PCT": "0.2",
        "RISK_DEFAULT_MMR": "0.007",
        "RISK_DAILY_LOSS_LIMIT": "250",
        "RISK_MAX_TOTAL_EXPOSURE_PCT": "0.75",
        "RISK_MAX_POSITION_PCT_PER_SYMBOL": "0.25",
        "RISK_SLIPPAGE_GUARD_BPS": "40",
        "RISK_KILL_SWITCH_ENABLED": "1",
        "RISK_REDUCE_ONLY_MODE": "true",
        "RISK_MAX_MARK_STALENESS_SECONDS": "90",
        "RISK_ENABLE_DYNAMIC_POSITION_SIZING": "yes",
        "RISK_POSITION_SIZING_METHOD": "risk_based",
        "RISK_POSITION_SIZING_MULTIPLIER": "1.5",
        "RISK_ENABLE_MARKET_IMPACT_GUARD": "true",
        "RISK_MAX_MARKET_IMPACT_BPS": "12",
        "RISK_ENABLE_VOLATILITY_CB": "true",
        "RISK_MAX_INTRADAY_VOL": "0.3",
        "RISK_VOL_WINDOW_PERIODS": "10",
        "RISK_CB_COOLDOWN_MIN": "45",
        "RISK_VOL_WARNING_THRESH": "0.12",
        "RISK_VOL_REDUCE_ONLY_THRESH": "0.18",
        "RISK_VOL_KILL_SWITCH_THRESH": "0.22",
    }

    config = RiskConfig.from_env(settings=StubSettings(env))

    assert config.max_leverage == 7
    assert config.leverage_max_per_symbol == {"BTC-PERP": 5, "ETH-PERP": 3}
    assert config.daytime_start_utc == "08:00"
    assert config.day_leverage_max_per_symbol == {"BTC-PERP": 4}
    assert config.day_mmr_per_symbol == {"BTC-PERP": 0.01}
    assert config.night_mmr_per_symbol == {"BTC-PERP": 0.015}
    assert config.enable_pre_trade_liq_projection is False
    assert config.min_liquidation_buffer_pct == 0.2
    assert config.default_maintenance_margin_rate == 0.007
    assert config.daily_loss_limit == Decimal("250")
    assert config.max_exposure_pct == 0.75
    assert config.max_position_pct_per_symbol == 0.25
    assert config.max_notional_per_symbol == {
        "BTC-PERP": Decimal("10000"),
        "ETH-PERP": Decimal("5000"),
    }
    assert config.slippage_guard_bps == 40
    assert config.kill_switch_enabled is True
    assert config.reduce_only_mode is True
    assert config.max_mark_staleness_seconds == 90
    assert config.enable_dynamic_position_sizing is True
    assert config.position_sizing_method == "risk_based"
    assert config.position_sizing_multiplier == 1.5
    assert config.enable_market_impact_guard is True
    assert config.max_market_impact_bps == 12
    assert config.enable_volatility_circuit_breaker is True
    assert config.max_intraday_volatility_threshold == 0.3
    assert config.volatility_window_periods == 10
    assert config.circuit_breaker_cooldown_minutes == 45
    assert config.volatility_warning_threshold == 0.12
    assert config.volatility_reduce_only_threshold == 0.18
    assert config.volatility_kill_switch_threshold == 0.22


def test_risk_config_from_env_defaults_when_missing() -> None:
    config = RiskConfig.from_env(settings=StubSettings({}))

    assert config.max_leverage == 5
    assert config.enable_pre_trade_liq_projection is True
    assert config.max_exposure_pct == 0.8
    assert config.slippage_guard_bps == 50
    assert config.kill_switch_enabled is False
    assert config.max_notional_per_symbol == {}


def test_risk_config_from_json_coerces_types(tmp_path: Path) -> None:
    payload = {
        "max_leverage": "8",
        "daily_loss_limit": "500",
        "enable_pre_trade_liq_projection": False,
        "default_maintenance_margin_rate": "0.01",
        "max_mark_staleness_seconds": "120",
        "enable_dynamic_position_sizing": True,
        "position_sizing_multiplier": 2.0,
        "kill_switch_enabled": True,
        "daytime_start_utc": "",
        "day_mmr_per_symbol": {"BTC-PERP": "0.02"},
    }
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(payload))

    config = RiskConfig.from_json(str(path))

    assert config.max_leverage == 8
    assert config.daily_loss_limit == Decimal("500")
    assert config.enable_pre_trade_liq_projection is False
    assert config.default_maintenance_margin_rate == 0.01
    assert config.max_mark_staleness_seconds == 120
    assert config.enable_dynamic_position_sizing is True
    assert config.position_sizing_multiplier == 2.0
    assert config.kill_switch_enabled is True
    assert config.daytime_start_utc is None
    assert config.day_mmr_per_symbol == {"BTC-PERP": 0.02}


def test_risk_config_to_dict_serializes_decimal() -> None:
    config = RiskConfig(
        daily_loss_limit=Decimal("123.45"),
        max_notional_per_symbol={"BTC-PERP": Decimal("1000")},
    )

    serialized = config.to_dict()

    assert serialized["daily_loss_limit"] == "123.45"
    assert serialized["max_notional_per_symbol"]["BTC-PERP"] == "1000"
