from bot.config import TradingConfig, get_config


def test_config_defaults() -> None:
    """Test that configuration loads with sensible defaults."""
    config = get_config()

    # Test logging config
    assert config.logging.level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    # Test trading config exists
    assert isinstance(config, TradingConfig)

    # Test financial config
    assert config.financial.capital.backtesting_capital > 0
    assert config.financial.capital.live_trading_capital > 0
