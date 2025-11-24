from __future__ import annotations

import json
from decimal import Decimal

from gpt_trader.config.live_trade_config import RiskConfig


def test_riskconfig_to_dict_serializes_decimals():
    config = RiskConfig(
        daily_loss_limit=Decimal("123.45"),
        max_notional_per_symbol={"BTC-PERP": Decimal("10000")},
    )
    d = config.to_dict()
    assert isinstance(d["daily_loss_limit"], str)
    assert d["daily_loss_limit"] == "123.45"
    assert d["max_notional_per_symbol"]["BTC-PERP"] == "10000"


def test_riskconfig_from_json_type_normalization(tmp_path):
    p = tmp_path / "risk.json"
    payload = {
        "max_leverage": 7,
        "min_liquidation_buffer_pct": 0.2,
        "daily_loss_limit": "300",
        "max_exposure_pct": 0.9,
        "max_position_pct_per_symbol": 0.3,
        "leverage_max_per_symbol": {"BTC-PERP": 6},
        "max_notional_per_symbol": {"ETH-PERP": "5000"},
        "slippage_guard_bps": 75,
        "kill_switch_enabled": False,
        "reduce_only_mode": False,
    }
    p.write_text(json.dumps(payload))

    config = RiskConfig.from_json(str(p))
    assert config.max_leverage == 7
    assert config.min_liquidation_buffer_pct == 0.2
    assert isinstance(config.daily_loss_limit, Decimal)
    assert config.daily_loss_limit == Decimal("300")
    assert config.max_exposure_pct == 0.9
    assert config.max_position_pct_per_symbol == 0.3
    assert config.leverage_max_per_symbol == {"BTC-PERP": 6}
    assert config.max_notional_per_symbol["ETH-PERP"] == Decimal("5000")
    assert config.slippage_guard_bps == 75
