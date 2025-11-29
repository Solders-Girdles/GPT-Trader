"""Tests for the shared INTX portfolio service."""

from __future__ import annotations

import pytest

from gpt_trader.orchestration.intx_portfolio_service import IntxPortfolioService


class StubAccountManager:
    def __init__(self, *, supports_intx: bool = True) -> None:
        self._supports_intx = supports_intx
        self.intx_portfolio_uuid: str | None = None
        self.calls: list[tuple[str, bool]] = []

    def supports_intx(self) -> bool:
        return self._supports_intx

    def get_intx_portfolio_uuid(self, *, refresh: bool = False) -> str | None:
        self.calls.append(("get", refresh))
        if refresh:
            return self.intx_portfolio_uuid
        return self.intx_portfolio_uuid

    def invalidate_intx_cache(self) -> None:
        self.calls.append(("invalidate", False))


def test_service_honours_config_override() -> None:
    manager = StubAccountManager()
    mock_config = type("MockConfig", (), {"coinbase_intx_portfolio_uuid": "pf-config"})()
    service = IntxPortfolioService(account_manager=manager, config=mock_config)

    assert manager.intx_portfolio_uuid == "pf-config"
    assert service.get_portfolio_uuid() == "pf-config"


def test_service_attempts_refresh_when_initial_lookup_empty() -> None:
    class RefreshingManager(StubAccountManager):
        def __init__(self) -> None:
            super().__init__()
            self.intx_portfolio_uuid = None
            self._attempts = 0

        def get_intx_portfolio_uuid(self, *, refresh: bool = False) -> str | None:
            self.calls.append(("get", refresh))
            if refresh or self._attempts > 0:
                return "pf-discovered"
            self._attempts += 1
            return None

    manager = RefreshingManager()
    service = IntxPortfolioService(account_manager=manager)

    assert service.get_portfolio_uuid() == "pf-discovered"
    assert ("get", False) in manager.calls
    assert ("get", True) in manager.calls


def test_service_invalidate_delegates_to_account_manager() -> None:
    manager = StubAccountManager()
    service = IntxPortfolioService(account_manager=manager)

    service.invalidate()

    assert ("invalidate", False) in manager.calls


def test_service_resolve_or_raise_errors_when_unavailable() -> None:
    manager = StubAccountManager(supports_intx=False)
    service = IntxPortfolioService(account_manager=manager)

    with pytest.raises(RuntimeError):
        service.resolve_or_raise()
