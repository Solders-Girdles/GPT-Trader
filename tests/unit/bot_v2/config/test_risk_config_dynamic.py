from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.bot_v2.config.live_trade_config import RiskConfig


def test_risk_config_from_env_dynamic_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISK_ENABLE_DYNAMIC_POSITION_SIZING", "1")
    monkeypatch.setenv("RISK_POSITION_SIZING_METHOD", "intelligent")
    monkeypatch.setenv("RISK_POSITION_SIZING_MULTIPLIER", "1.25")
    monkeypatch.setenv("RISK_ENABLE_MARKET_IMPACT_GUARD", "1")
    monkeypatch.setenv("RISK_MAX_MARKET_IMPACT_BPS", "45")

    config = RiskConfig.from_env()

    assert config.enable_dynamic_position_sizing is True
    assert config.position_sizing_method == "intelligent"
    assert pytest.approx(config.position_sizing_multiplier, rel=1e-6) == 1.25
    assert config.enable_market_impact_guard is True
    assert config.max_market_impact_bps == 45


def test_risk_config_from_json_normalizes_dynamic_fields(tmp_path) -> None:
    payload = {
        "enable_dynamic_position_sizing": True,
        "position_sizing_method": "volume_scaled",
        "position_sizing_multiplier": 0.75,
        "enable_market_impact_guard": True,
        "max_market_impact_bps": 30,
        "daily_loss_limit": "150",
    }
    config_path = tmp_path / "risk.json"
    config_path.write_text(json.dumps(payload))

    config = RiskConfig.from_json(str(config_path))

    assert config.enable_dynamic_position_sizing is True
    assert config.position_sizing_method == "volume_scaled"
    assert pytest.approx(config.position_sizing_multiplier, rel=1e-6) == 0.75
    assert config.enable_market_impact_guard is True
    assert config.max_market_impact_bps == 30
    assert config.daily_loss_limit == Decimal("150")
