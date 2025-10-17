import pytest

from bot_v2.config.env_utils import EnvVarError
from bot_v2.config.live_trade_config import RISK_CONFIG_ENV_KEYS, RiskConfig
from bot_v2.orchestration.runtime_settings import load_runtime_settings


def test_from_env_populates_mappings() -> None:
    env = {
        "RISK_LEVERAGE_MAX_PER_SYMBOL": "BTC-PERP:10",
        "RISK_DAY_MMR_PER_SYMBOL": "BTC-PERP:0.01",
        "RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION": "true",
    }
    settings = load_runtime_settings(env)
    snapshot = settings.snapshot_env(RISK_CONFIG_ENV_KEYS)

    assert snapshot["RISK_LEVERAGE_MAX_PER_SYMBOL"] == "BTC-PERP:10"
    assert snapshot["RISK_DAY_MMR_PER_SYMBOL"] == "BTC-PERP:0.01"
    assert snapshot["RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION"] == "true"

    config = RiskConfig.from_env(settings=settings)

    assert config.leverage_max_per_symbol == {"BTC-PERP": 10}
    assert config.day_mmr_per_symbol == {"BTC-PERP": 0.01}
    assert config.enable_pre_trade_liq_projection is True


def test_from_env_invalid_mapping_raises() -> None:
    settings = load_runtime_settings({"RISK_LEVERAGE_MAX_PER_SYMBOL": "BTC-PERP-10"})
    with pytest.raises(EnvVarError):
        RiskConfig.from_env(settings=settings)


def test_from_env_invalid_boolean_raises() -> None:
    settings = load_runtime_settings({"RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION": "notabool"})
    with pytest.raises(EnvVarError):
        RiskConfig.from_env(settings=settings)


def test_from_env_invalid_percentage_raises() -> None:
    settings = load_runtime_settings({"RISK_MAX_EXPOSURE_PCT": "1.5"})
    with pytest.raises(EnvVarError):
        RiskConfig.from_env(settings=settings)


def test_from_env_invalid_alias_percentage_raises() -> None:
    settings = load_runtime_settings({"RISK_MAX_TOTAL_EXPOSURE_PCT": "1.2"})
    with pytest.raises(EnvVarError):
        RiskConfig.from_env(settings=settings)


def test_from_env_legacy_exposure_alias() -> None:
    settings = load_runtime_settings({"RISK_MAX_TOTAL_EXPOSURE_PCT": "0.65"})
    config = RiskConfig.from_env(settings=settings)

    assert config.max_exposure_pct == 0.65
