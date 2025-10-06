import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.config.env_utils import EnvVarError


def test_from_env_populates_mappings(monkeypatch):
    monkeypatch.setenv("RISK_LEVERAGE_MAX_PER_SYMBOL", "BTC-PERP:10")
    monkeypatch.setenv("RISK_DAY_MMR_PER_SYMBOL", "BTC-PERP:0.01")
    monkeypatch.setenv("RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION", "true")

    config = RiskConfig.from_env()

    assert config.leverage_max_per_symbol == {"BTC-PERP": 10}
    assert config.day_mmr_per_symbol == {"BTC-PERP": 0.01}
    assert config.enable_pre_trade_liq_projection is True


def test_from_env_invalid_mapping_raises(monkeypatch):
    monkeypatch.setenv("RISK_LEVERAGE_MAX_PER_SYMBOL", "BTC-PERP-10")

    with pytest.raises(EnvVarError):
        RiskConfig.from_env()


def test_from_env_invalid_boolean_raises(monkeypatch):
    monkeypatch.setenv("RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION", "notabool")

    with pytest.raises(EnvVarError):
        RiskConfig.from_env()
