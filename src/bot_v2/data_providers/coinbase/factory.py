"""Factory helpers for constructing Coinbase data providers."""

from __future__ import annotations

from bot_v2.data_providers import DataProvider
from bot_v2.orchestration.runtime_settings import load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_provider")

TRUTHY = {"1", "true", "yes", "on"}


def create_coinbase_provider(
    use_real_data: bool | None = None,
    enable_streaming: bool | None = None,
) -> DataProvider:
    """
    Factory function returning either the real Coinbase provider or the mock provider.
    """
    runtime_settings = load_runtime_settings()
    raw_env = runtime_settings.raw_env

    if use_real_data is None:
        use_real_data = raw_env.get("COINBASE_USE_REAL_DATA", "0").strip().lower() in TRUTHY

    if not use_real_data:
        from bot_v2.data_providers import MockProvider

        logger.info(
            "Using MockProvider for Coinbase data",
            operation="provider_factory",
            status="mock",
        )
        return MockProvider()

    if enable_streaming is None:
        enable_streaming = raw_env.get("COINBASE_ENABLE_STREAMING", "0").strip().lower() in TRUTHY

    logger.info(
        "Creating CoinbaseDataProvider",
        operation="provider_factory",
        streaming=bool(enable_streaming),
    )
    from bot_v2.data_providers.coinbase_provider import CoinbaseDataProvider

    return CoinbaseDataProvider(enable_streaming=bool(enable_streaming))


__all__ = ["create_coinbase_provider"]
