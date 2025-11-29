"""Shared service for resolving Coinbase INTX portfolio identifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing convenience
    from gpt_trader.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
    from gpt_trader.orchestration.configuration import BotConfig


class IntxPortfolioService:
    """Provides a cached INTX portfolio UUID for orchestration components."""

    def __init__(
        self,
        *,
        account_manager: CoinbaseAccountManager,
        config: BotConfig | None = None,
    ) -> None:
        self._account_manager = account_manager
        self._config = config
        override_uuid = (
            getattr(self._config, "coinbase_intx_portfolio_uuid", None)
            if self._config is not None
            else None
        )
        if override_uuid and not getattr(self._account_manager, "intx_portfolio_uuid", None):
            self._account_manager.intx_portfolio_uuid = override_uuid  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def get_portfolio_uuid(self, *, refresh: bool = False) -> str | None:
        """Return the currently resolved INTX portfolio UUID."""
        if not self._account_manager.supports_intx():
            return None
        uuid = self._account_manager.get_intx_portfolio_uuid(refresh=refresh)
        if uuid:
            return uuid
        if refresh:
            return None
        # Attempt a forced refresh if the first pass failed.
        return self._account_manager.get_intx_portfolio_uuid(refresh=True)

    def invalidate(self) -> None:
        """Clear the cached INTX portfolio UUID."""
        if hasattr(self._account_manager, "invalidate_intx_cache"):
            self._account_manager.invalidate_intx_cache()

    def resolve_or_raise(self) -> str:
        """Return a portfolio UUID or raise if none can be resolved."""
        uuid = self.get_portfolio_uuid(refresh=False)
        if uuid is not None:
            return uuid
        uuid = self.get_portfolio_uuid(refresh=True)
        if uuid is not None:
            return uuid
        raise RuntimeError(
            "Unable to resolve INTX portfolio UUID; no entitlement or override found."
        )

    def supports_intx(self) -> bool:
        return self._account_manager.supports_intx()

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Diagnostics snapshot for debugging."""
        uuid = self.get_portfolio_uuid(refresh=False)
        return {
            "supports_intx": self.supports_intx(),
            "portfolio_uuid": uuid,
            "override_uuid": (
                getattr(self._config, "coinbase_intx_portfolio_uuid", None)
                if self._config is not None
                else None
            ),
        }
