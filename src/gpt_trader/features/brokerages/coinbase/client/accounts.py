"""Account and system endpoints for Coinbase REST client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.client._typing import CoinbaseClientProtocol


class AccountClientMixin:
    """Methods related to accounts, system info, and limits."""

    def get_accounts(self: CoinbaseClientProtocol) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path("accounts"))

    def get_account(self: CoinbaseClientProtocol, account_uuid: str) -> dict[str, Any]:
        path = self._get_endpoint_path(
            "account", account_uuid=account_uuid, account_id=account_uuid
        )
        return self._request("GET", path)

    def get_time(self: CoinbaseClientProtocol) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path("time"))

    def get_key_permissions(self: CoinbaseClientProtocol) -> dict[str, Any]:
        path = self._get_endpoint_path("key_permissions")
        return self._request("GET", path)

    def get_fees(self: CoinbaseClientProtocol) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path("fees"))

    def get_limits(self: CoinbaseClientProtocol) -> dict[str, Any]:
        path = self._get_endpoint_path("limits")
        return self._request("GET", path)

    def get_transaction_summary(self: CoinbaseClientProtocol) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "get_transaction_summary not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path("transaction_summary")
        return self._request("GET", path)


__all__ = ["AccountClientMixin"]
