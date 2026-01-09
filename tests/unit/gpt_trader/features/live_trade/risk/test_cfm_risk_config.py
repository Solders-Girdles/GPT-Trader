"""Tests for CFM-specific risk configuration in gpt_trader.features.live_trade.risk module."""

import json
import os
import tempfile
from decimal import Decimal

from gpt_trader.features.live_trade.risk.config import RiskConfig


class TestCFMRiskConfigDefaults:
    """Tests for CFM risk configuration default values."""

    def test_cfm_max_leverage_default(self):
        """Default CFM max leverage is 5."""
        config = RiskConfig()
        assert config.cfm_max_leverage == 5

    def test_cfm_min_liquidation_buffer_pct_default(self):
        """Default CFM min liquidation buffer is 15%."""
        config = RiskConfig()
        assert config.cfm_min_liquidation_buffer_pct == 0.15

    def test_cfm_max_exposure_pct_default(self):
        """Default CFM max exposure is 80%."""
        config = RiskConfig()
        assert config.cfm_max_exposure_pct == 0.8

    def test_cfm_max_position_size_pct_default(self):
        """Default CFM max position size is 25%."""
        config = RiskConfig()
        assert config.cfm_max_position_size_pct == 0.25

    def test_cfm_leverage_max_per_symbol_default(self):
        """Default CFM per-symbol leverage limits is empty."""
        config = RiskConfig()
        assert config.cfm_leverage_max_per_symbol == {}

    def test_cfm_max_notional_per_symbol_default(self):
        """Default CFM per-symbol notional limits is empty."""
        config = RiskConfig()
        assert config.cfm_max_notional_per_symbol == {}

    def test_cfm_day_leverage_max_per_symbol_default(self):
        """Default CFM day leverage limits is empty."""
        config = RiskConfig()
        assert config.cfm_day_leverage_max_per_symbol == {}

    def test_cfm_night_leverage_max_per_symbol_default(self):
        """Default CFM night leverage limits is empty."""
        config = RiskConfig()
        assert config.cfm_night_leverage_max_per_symbol == {}


class TestCFMRiskConfigValues:
    """Tests for setting CFM risk configuration values."""

    def test_set_cfm_max_leverage(self):
        """Can set CFM max leverage."""
        config = RiskConfig(cfm_max_leverage=10)
        assert config.cfm_max_leverage == 10

    def test_set_cfm_min_liquidation_buffer_pct(self):
        """Can set CFM min liquidation buffer."""
        config = RiskConfig(cfm_min_liquidation_buffer_pct=0.20)
        assert config.cfm_min_liquidation_buffer_pct == 0.20

    def test_set_cfm_max_exposure_pct(self):
        """Can set CFM max exposure."""
        config = RiskConfig(cfm_max_exposure_pct=0.5)
        assert config.cfm_max_exposure_pct == 0.5

    def test_set_cfm_max_position_size_pct(self):
        """Can set CFM max position size."""
        config = RiskConfig(cfm_max_position_size_pct=0.10)
        assert config.cfm_max_position_size_pct == 0.10

    def test_set_cfm_leverage_max_per_symbol(self):
        """Can set CFM per-symbol leverage limits."""
        limits = {"BTC-20DEC30-CDE": 10, "ETH-20DEC30-CDE": 5}
        config = RiskConfig(cfm_leverage_max_per_symbol=limits)
        assert config.cfm_leverage_max_per_symbol == limits

    def test_set_cfm_max_notional_per_symbol(self):
        """Can set CFM per-symbol notional limits."""
        limits = {
            "BTC-20DEC30-CDE": Decimal("100000"),
            "ETH-20DEC30-CDE": Decimal("50000"),
        }
        config = RiskConfig(cfm_max_notional_per_symbol=limits)
        assert config.cfm_max_notional_per_symbol == limits

    def test_full_cfm_risk_config(self):
        """Full CFM risk configuration works together."""
        config = RiskConfig(
            cfm_max_leverage=3,
            cfm_min_liquidation_buffer_pct=0.25,
            cfm_max_exposure_pct=0.6,
            cfm_max_position_size_pct=0.15,
            cfm_leverage_max_per_symbol={"BTC-20DEC30-CDE": 5},
            cfm_day_leverage_max_per_symbol={"BTC-20DEC30-CDE": 10},
            cfm_night_leverage_max_per_symbol={"BTC-20DEC30-CDE": 3},
        )
        assert config.cfm_max_leverage == 3
        assert config.cfm_min_liquidation_buffer_pct == 0.25
        assert config.cfm_max_exposure_pct == 0.6
        assert config.cfm_max_position_size_pct == 0.15
        assert config.cfm_leverage_max_per_symbol == {"BTC-20DEC30-CDE": 5}
        assert config.cfm_day_leverage_max_per_symbol == {"BTC-20DEC30-CDE": 10}
        assert config.cfm_night_leverage_max_per_symbol == {"BTC-20DEC30-CDE": 3}


class TestCFMRiskConfigFromEnv:
    """Tests for loading CFM risk config from environment variables."""

    def test_from_env_cfm_max_leverage(self, monkeypatch):
        """Loads CFM max leverage from env."""
        monkeypatch.setenv("CFM_MAX_LEVERAGE", "10")
        config = RiskConfig.from_env()
        assert config.cfm_max_leverage == 10

    def test_from_env_cfm_min_liquidation_buffer_pct(self, monkeypatch):
        """Loads CFM min liquidation buffer from env."""
        monkeypatch.setenv("CFM_MIN_LIQUIDATION_BUFFER_PCT", "0.25")
        config = RiskConfig.from_env()
        assert config.cfm_min_liquidation_buffer_pct == 0.25

    def test_from_env_cfm_max_exposure_pct(self, monkeypatch):
        """Loads CFM max exposure from env."""
        monkeypatch.setenv("CFM_MAX_EXPOSURE_PCT", "0.6")
        config = RiskConfig.from_env()
        assert config.cfm_max_exposure_pct == 0.6

    def test_from_env_cfm_max_position_size_pct(self, monkeypatch):
        """Loads CFM max position size from env."""
        monkeypatch.setenv("CFM_MAX_POSITION_SIZE_PCT", "0.10")
        config = RiskConfig.from_env()
        assert config.cfm_max_position_size_pct == 0.10

    def test_from_env_defaults(self):
        """Uses defaults when env vars not set."""
        config = RiskConfig.from_env()
        assert config.cfm_max_leverage == 5
        assert config.cfm_min_liquidation_buffer_pct == 0.15
        assert config.cfm_max_exposure_pct == 0.8
        assert config.cfm_max_position_size_pct == 0.25


class TestCFMRiskConfigFromJson:
    """Tests for loading CFM risk config from JSON."""

    def test_from_json_cfm_values(self):
        """Loads CFM config values from JSON."""
        config_data = {
            "cfm_max_leverage": 10,
            "cfm_min_liquidation_buffer_pct": 0.20,
            "cfm_max_exposure_pct": 0.7,
            "cfm_max_position_size_pct": 0.30,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = RiskConfig.from_json(temp_path)
            assert config.cfm_max_leverage == 10
            assert config.cfm_min_liquidation_buffer_pct == 0.20
            assert config.cfm_max_exposure_pct == 0.7
            assert config.cfm_max_position_size_pct == 0.30
        finally:
            os.unlink(temp_path)

    def test_from_json_cfm_max_notional_per_symbol(self):
        """Loads CFM notional limits with Decimal conversion."""
        config_data = {
            "cfm_max_notional_per_symbol": {
                "BTC-20DEC30-CDE": 100000,
                "ETH-20DEC30-CDE": 50000,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = RiskConfig.from_json(temp_path)
            assert config.cfm_max_notional_per_symbol["BTC-20DEC30-CDE"] == Decimal("100000")
            assert config.cfm_max_notional_per_symbol["ETH-20DEC30-CDE"] == Decimal("50000")
        finally:
            os.unlink(temp_path)

    def test_from_json_cfm_leverage_per_symbol(self):
        """Loads CFM leverage limits from JSON."""
        config_data = {
            "cfm_leverage_max_per_symbol": {"BTC-20DEC30-CDE": 10, "ETH-20DEC30-CDE": 5},
            "cfm_day_leverage_max_per_symbol": {"BTC-20DEC30-CDE": 15},
            "cfm_night_leverage_max_per_symbol": {"BTC-20DEC30-CDE": 5},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = RiskConfig.from_json(temp_path)
            assert config.cfm_leverage_max_per_symbol == {
                "BTC-20DEC30-CDE": 10,
                "ETH-20DEC30-CDE": 5,
            }
            assert config.cfm_day_leverage_max_per_symbol == {"BTC-20DEC30-CDE": 15}
            assert config.cfm_night_leverage_max_per_symbol == {"BTC-20DEC30-CDE": 5}
        finally:
            os.unlink(temp_path)


class TestCFMRiskConfigToDict:
    """Tests for serializing CFM risk config to dict."""

    def test_to_dict_cfm_values(self):
        """CFM values serialize correctly."""
        config = RiskConfig(
            cfm_max_leverage=10,
            cfm_min_liquidation_buffer_pct=0.20,
            cfm_max_exposure_pct=0.7,
        )

        data = config.to_dict()

        assert data["cfm_max_leverage"] == 10
        assert data["cfm_min_liquidation_buffer_pct"] == 0.20
        assert data["cfm_max_exposure_pct"] == 0.7

    def test_to_dict_cfm_notional_decimals(self):
        """CFM notional Decimal values convert to strings."""
        config = RiskConfig(
            cfm_max_notional_per_symbol={
                "BTC-20DEC30-CDE": Decimal("100000"),
            }
        )

        data = config.to_dict()

        assert data["cfm_max_notional_per_symbol"]["BTC-20DEC30-CDE"] == "100000"
