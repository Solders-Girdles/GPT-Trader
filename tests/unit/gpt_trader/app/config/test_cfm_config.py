"""Tests for CFM configuration in gpt_trader.app.config.bot_config module."""

from gpt_trader.app.config.bot_config import BotConfig


class TestCFMConfigDefaults:
    """Test CFM configuration default values."""

    def test_trading_modes_default(self):
        """Default trading modes is spot only."""
        config = BotConfig()
        assert config.trading_modes == ["spot"]

    def test_cfm_enabled_default(self):
        """CFM is disabled by default."""
        config = BotConfig()
        assert config.cfm_enabled is False

    def test_cfm_max_leverage_default(self):
        """Default CFM max leverage is 5."""
        config = BotConfig()
        assert config.cfm_max_leverage == 5

    def test_cfm_symbols_default(self):
        """CFM symbols list is empty by default."""
        config = BotConfig()
        assert config.cfm_symbols == []

    def test_cfm_margin_window_default(self):
        """Default margin window is STANDARD."""
        config = BotConfig()
        assert config.cfm_margin_window == "STANDARD"


class TestTradingModeProperties:
    """Test trading mode convenience properties."""

    def test_is_spot_only_default(self):
        """Default config is spot-only."""
        config = BotConfig()
        assert config.is_spot_only is True
        assert config.is_cfm_only is False
        assert config.is_hybrid_mode is False

    def test_is_cfm_only(self):
        """CFM-only mode detection."""
        config = BotConfig(trading_modes=["cfm"])
        assert config.is_spot_only is False
        assert config.is_cfm_only is True
        assert config.is_hybrid_mode is False

    def test_is_hybrid_mode(self):
        """Hybrid mode detection (both spot and CFM)."""
        config = BotConfig(trading_modes=["spot", "cfm"])
        assert config.is_spot_only is False
        assert config.is_cfm_only is False
        assert config.is_hybrid_mode is True

    def test_hybrid_mode_order_independent(self):
        """Hybrid mode works regardless of order."""
        config = BotConfig(trading_modes=["cfm", "spot"])
        assert config.is_hybrid_mode is True

    def test_empty_trading_modes(self):
        """Empty trading modes results in no mode flags being True."""
        config = BotConfig(trading_modes=[])
        assert config.is_spot_only is False
        assert config.is_cfm_only is False
        assert config.is_hybrid_mode is False


class TestCFMConfigValues:
    """Test setting CFM configuration values."""

    def test_cfm_enabled(self):
        """Can enable CFM."""
        config = BotConfig(cfm_enabled=True)
        assert config.cfm_enabled is True

    def test_cfm_max_leverage(self):
        """Can set max leverage."""
        config = BotConfig(cfm_max_leverage=10)
        assert config.cfm_max_leverage == 10

    def test_cfm_symbols(self):
        """Can set CFM symbols."""
        symbols = ["BTC-20DEC30-CDE", "ETH-20DEC30-CDE"]
        config = BotConfig(cfm_symbols=symbols)
        assert config.cfm_symbols == symbols

    def test_cfm_margin_window_intraday(self):
        """Can set intraday margin window."""
        config = BotConfig(cfm_margin_window="INTRADAY_STANDARD")
        assert config.cfm_margin_window == "INTRADAY_STANDARD"

    def test_full_cfm_config(self):
        """Full CFM configuration works together."""
        config = BotConfig(
            trading_modes=["spot", "cfm"],
            cfm_enabled=True,
            cfm_max_leverage=3,
            cfm_symbols=["BTC-20DEC30-CDE"],
            cfm_margin_window="INTRADAY_PLUS",
        )
        assert config.is_hybrid_mode is True
        assert config.cfm_enabled is True
        assert config.cfm_max_leverage == 3
        assert config.cfm_symbols == ["BTC-20DEC30-CDE"]
        assert config.cfm_margin_window == "INTRADAY_PLUS"
