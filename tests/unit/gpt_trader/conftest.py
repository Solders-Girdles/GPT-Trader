from pathlib import Path

import pytest

from gpt_trader.config.runtime_settings import RuntimeSettings


@pytest.fixture
def runtime_settings_factory():
    """Factory for creating RuntimeSettings instances for testing."""

    def _factory(
        env_overrides: dict[str, str] | None = None,
        runtime_root: Path | None = None,
        event_store_root_override: Path | None = None,
        **kwargs,
    ) -> RuntimeSettings:
        # Create a basic RuntimeSettings instance, allowing overrides
        settings = RuntimeSettings(
            raw_env=env_overrides or {},
            runtime_root=runtime_root or Path("/mock/runtime"),
            event_store_root_override=event_store_root_override,
            # Populate other fields with sensible defaults or from kwargs
            coinbase_default_quote="USD",
            coinbase_default_quote_overridden=False,
            coinbase_enable_derivatives=False,
            coinbase_enable_derivatives_overridden=False,
            perps_enable_streaming=False,
            perps_stream_level=1,
            perps_paper_trading=False,
            perps_force_mock=False,
            perps_skip_startup_reconcile=False,
            perps_position_fraction=None,
            order_preview_enabled=None,
            spot_force_live=False,
            broker_hint=None,
            coinbase_sandbox_enabled=False,
            coinbase_api_mode="advanced",
            risk_config_path=None,
            coinbase_intx_portfolio_uuid=None,
            **kwargs,
        )
        return settings

    return _factory
