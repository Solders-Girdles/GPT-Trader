from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore


@dataclass(frozen=True)
class CDPCredentials:
    api_key: str
    private_key: str
    skip_reason: str | None = None


@pytest.fixture(autouse=True)
def fast_retry_sleep(fake_clock, monkeypatch):
    """Auto-use deterministic clock so retry loops advance instantly."""
    monkeypatch.setattr("time.sleep", fake_clock.sleep)
    return fake_clock


@pytest.fixture(autouse=True)
def fast_retry_env(monkeypatch, request):
    """Ensure Coinbase client retries run with zero delay for faster tests."""
    if "test_coinbase_system.py" in request.node.nodeid:
        monkeypatch.delenv("COINBASE_FAST_RETRY", raising=False)
        yield
        return
    monkeypatch.setenv("COINBASE_FAST_RETRY", "1")
    yield


@pytest.fixture
def coinbase_cdp_credentials() -> CDPCredentials:
    """Provide Coinbase CDP API credentials or skip when unavailable."""

    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY")

    if not api_key or not private_key:
        skip_reason = "COINBASE_PROD_CDP_* credentials not set"
        pytest.skip(skip_reason)

    return CDPCredentials(api_key=api_key, private_key=private_key)


@pytest.fixture
def fake_clock():
    """Provides a controllable time source for testing."""
    import time

    class FakeClock:
        def __init__(self):
            self._current = time.time()

        def time(self):
            return self._current

        def sleep(self, duration):
            self._current += duration

    return FakeClock()
