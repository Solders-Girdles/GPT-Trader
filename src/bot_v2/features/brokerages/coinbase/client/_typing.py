"""Type helpers for Coinbase REST client mixins."""

from __future__ import annotations

from typing import Any, Protocol


class CoinbaseClientProtocol(Protocol):
    """Protocol describing the minimal surface required by the mixins."""

    api_mode: str

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request and return the decoded JSON payload."""

    def _get_endpoint_path(self, endpoint_name: str, **kwargs: str) -> str:
        """Resolve an endpoint alias into a concrete path."""

    def _build_path_with_params(
        self,
        path: str,
        params: dict[str, Any] | None,
    ) -> str:
        """Append query parameters to the provided path."""

    def get_products(self) -> dict[str, Any]:
        """Return the raw product listing payload."""

    def get_product_book(self, product_id: str, level: int = 2) -> dict[str, Any]:
        """Optional helper for market mixins; implemented by the combined client."""


__all__ = ["CoinbaseClientProtocol"]
