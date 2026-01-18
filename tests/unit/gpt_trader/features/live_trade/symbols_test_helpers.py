from __future__ import annotations

from gpt_trader.app.config import BotConfig
from gpt_trader.config.types import Profile


def make_bot_config(
    *,
    derivatives_enabled: bool = False,
    coinbase_default_quote: str = "USD",
    **kwargs: dict,
) -> BotConfig:
    """Create a BotConfig instance for testing."""
    return BotConfig(
        symbols=["BTC-USD"],
        profile=Profile.DEV,
        mock_broker=True,
        dry_run=True,
        runtime_root=kwargs.get("runtime_root", "/tmp"),
        coinbase_default_quote=coinbase_default_quote,
        derivatives_enabled=derivatives_enabled,
        perps_enable_streaming=kwargs.get("perps_enable_streaming", False),
        perps_stream_level=kwargs.get("perps_stream_level", 1),
        perps_paper_trading=kwargs.get("perps_paper_trading", False),
        perps_skip_startup_reconcile=kwargs.get("perps_skip_startup_reconcile", False),
        perps_position_fraction=kwargs.get("perps_position_fraction"),
        enable_order_preview=kwargs.get("enable_order_preview", False),
        spot_force_live=kwargs.get("spot_force_live", False),
        broker_hint=kwargs.get("broker_hint"),
        coinbase_sandbox_enabled=kwargs.get("coinbase_sandbox_enabled", False),
        coinbase_api_mode=kwargs.get("coinbase_api_mode", "advanced"),
        risk_config_path=kwargs.get("risk_config_path"),
        coinbase_intx_portfolio_uuid=kwargs.get("coinbase_intx_portfolio_uuid"),
    )


def make_bot_config_extended(
    *,
    derivatives_enabled: bool = False,
    coinbase_default_quote: str = "USD",
    coinbase_us_futures_enabled: bool = False,
    coinbase_intx_perpetuals_enabled: bool = False,
    coinbase_derivatives_type: str = "",
    **kwargs: dict,
) -> BotConfig:
    """Create BotConfig instance with extended options."""
    return BotConfig(
        symbols=["BTC-USD"],
        profile=Profile.DEV,
        mock_broker=True,
        dry_run=True,
        runtime_root=kwargs.get("runtime_root", "/tmp"),
        coinbase_default_quote=coinbase_default_quote,
        derivatives_enabled=derivatives_enabled,
        coinbase_us_futures_enabled=coinbase_us_futures_enabled,
        coinbase_intx_perpetuals_enabled=coinbase_intx_perpetuals_enabled,
        coinbase_derivatives_type=coinbase_derivatives_type,
        perps_enable_streaming=kwargs.get("perps_enable_streaming", False),
        perps_stream_level=kwargs.get("perps_stream_level", 1),
        perps_paper_trading=kwargs.get("perps_paper_trading", False),
        perps_skip_startup_reconcile=kwargs.get("perps_skip_startup_reconcile", False),
        perps_position_fraction=kwargs.get("perps_position_fraction"),
        enable_order_preview=kwargs.get("enable_order_preview", False),
        spot_force_live=kwargs.get("spot_force_live", False),
        broker_hint=kwargs.get("broker_hint"),
        coinbase_sandbox_enabled=kwargs.get("coinbase_sandbox_enabled", False),
        coinbase_api_mode=kwargs.get("coinbase_api_mode", "advanced"),
        risk_config_path=kwargs.get("risk_config_path"),
        coinbase_intx_portfolio_uuid=kwargs.get("coinbase_intx_portfolio_uuid"),
    )
