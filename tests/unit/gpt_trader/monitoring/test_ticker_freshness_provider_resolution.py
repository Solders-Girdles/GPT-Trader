"""Unit tests for ticker freshness provider resolution."""

from __future__ import annotations

import pytest

from gpt_trader.monitoring.health_checks import _resolve_ticker_freshness_provider


class StubProvider:
    def is_stale(self, symbol: str) -> bool:
        return False


class ProviderSource:
    def __init__(self, provider):
        self._provider = provider

    def get_ticker_freshness_provider(self):
        return self._provider


class ProviderSourceReturningNone:
    def __init__(self, cache):
        self.ticker_cache = cache

    def get_ticker_freshness_provider(self):
        return None


class ProviderSourceRaising:
    def __init__(self, cache):
        self._ticker_cache = cache

    def get_ticker_freshness_provider(self):
        raise RuntimeError("boom")


class DynamicIsStaleService:
    def __getattr__(self, name: str):
        if name == "is_stale":
            return lambda symbol: False
        raise AttributeError(name)


class NotAProvider:
    pass


def test_resolves_provider_instance():
    provider = StubProvider()

    resolved = _resolve_ticker_freshness_provider(provider)

    assert resolved is provider


def test_resolves_provider_source():
    provider = StubProvider()
    service = ProviderSource(provider)

    resolved = _resolve_ticker_freshness_provider(service)

    assert resolved is provider


def test_provider_source_returning_none_skips_attribute_fallback():
    cache = StubProvider()
    service = ProviderSourceReturningNone(cache)

    resolved = _resolve_ticker_freshness_provider(service)

    assert resolved is None


def test_provider_source_raising_skips_attribute_fallback():
    cache = StubProvider()
    service = ProviderSourceRaising(cache)

    resolved = _resolve_ticker_freshness_provider(service)

    assert resolved is None


@pytest.mark.parametrize("attribute_name", ["ticker_cache", "_ticker_cache"])
def test_resolves_legacy_ticker_cache_attribute(attribute_name: str):
    cache = StubProvider()

    class LegacyService:
        pass

    service = LegacyService()
    setattr(service, attribute_name, cache)

    resolved = _resolve_ticker_freshness_provider(service)

    assert resolved is cache


def test_resolves_callable_is_stale_attribute():
    service = DynamicIsStaleService()

    resolved = _resolve_ticker_freshness_provider(service)

    assert resolved is service


def test_provider_source_returning_non_provider_object():
    service = ProviderSource(NotAProvider())

    resolved = _resolve_ticker_freshness_provider(service)

    assert resolved is None
