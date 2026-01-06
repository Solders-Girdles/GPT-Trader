import pytest

from gpt_trader.app.config import BotConfig


@pytest.fixture
def bot_config_factory():
    """Factory for creating BotConfig instances for testing.

    Provides sensible defaults for all configuration fields used in unit tests.
    """

    def _factory(
        symbols: list[str] | None = None,
        profile: object = None,
        dry_run: bool = True,
        mock_broker: bool = True,
        runtime_root: str = "/mock/runtime",
        event_store_root_override: str | None = None,
        coinbase_default_quote: str = "USD",
        coinbase_sandbox_enabled: bool = False,
        coinbase_api_mode: str = "advanced",
        derivatives_enabled: bool = False,
        perps_enable_streaming: bool = False,
        perps_stream_level: int = 1,
        perps_paper_trading: bool = False,
        perps_skip_startup_reconcile: bool = False,
        perps_position_fraction: float | None = None,
        enable_order_preview: bool = False,
        spot_force_live: bool = False,
        broker_hint: str | None = None,
        risk_config_path: str | None = None,
        coinbase_intx_portfolio_uuid: str | None = None,
        **kwargs,
    ) -> BotConfig:
        return BotConfig(
            symbols=symbols or ["BTC-USD", "ETH-USD"],
            profile=profile,
            dry_run=dry_run,
            mock_broker=mock_broker,
            runtime_root=runtime_root,
            event_store_root_override=event_store_root_override,
            coinbase_default_quote=coinbase_default_quote,
            coinbase_sandbox_enabled=coinbase_sandbox_enabled,
            coinbase_api_mode=coinbase_api_mode,
            derivatives_enabled=derivatives_enabled,
            perps_enable_streaming=perps_enable_streaming,
            perps_stream_level=perps_stream_level,
            perps_paper_trading=perps_paper_trading,
            perps_skip_startup_reconcile=perps_skip_startup_reconcile,
            perps_position_fraction=perps_position_fraction,
            enable_order_preview=enable_order_preview,
            spot_force_live=spot_force_live,
            broker_hint=broker_hint,
            risk_config_path=risk_config_path,
            coinbase_intx_portfolio_uuid=coinbase_intx_portfolio_uuid,
            **kwargs,
        )

    return _factory
