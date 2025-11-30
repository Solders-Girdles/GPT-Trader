"""Portfolio, convert, derivatives, and treasury endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.client._typing import CoinbaseClientProtocol


class PortfolioClientMixin:
    """Aggregates portfolio, treasury, and derivatives endpoints."""

    def list_portfolios(self: CoinbaseClientProtocol) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError("list_portfolios requires advanced mode.")
        return self._request("GET", self._get_endpoint_path("portfolios"))

    def get_portfolio(self: CoinbaseClientProtocol, portfolio_uuid: str) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError("get_portfolio requires advanced mode.")
        path = self._get_endpoint_path("portfolio", portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    def get_portfolio_breakdown(
        self: CoinbaseClientProtocol, portfolio_uuid: str
    ) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError("get_portfolio_breakdown requires advanced mode.")
        path = self._get_endpoint_path("portfolio_breakdown", portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    def move_funds(self: CoinbaseClientProtocol, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError("move_funds requires advanced mode.")
        path = self._get_endpoint_path("move_funds")
        return self._request("POST", path, payload)

    def convert_quote(self: CoinbaseClientProtocol, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._get_endpoint_path("convert_quote")
        return self._request("POST", path, payload)

    def get_convert_trade(self: CoinbaseClientProtocol, trade_id: str) -> dict[str, Any]:
        path = self._get_endpoint_path("convert_trade", trade_id=trade_id)
        return self._request("GET", path)

    def commit_convert_trade(
        self: CoinbaseClientProtocol,
        trade_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "commit_convert_trade not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path("convert_trade", trade_id=trade_id)
        return self._request("POST", path, payload or {})

    def list_payment_methods(self: CoinbaseClientProtocol) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError("list_payment_methods requires advanced mode.")
        return self._request("GET", self._get_endpoint_path("payment_methods"))

    def get_payment_method(self: CoinbaseClientProtocol, payment_method_id: str) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError("get_payment_method requires advanced mode.")
        path = self._get_endpoint_path("payment_method", payment_method_id=payment_method_id)
        return self._request("GET", path)

    # ------------------------------------------------------------------
    # Derivatives & positions
    # ------------------------------------------------------------------
    def list_cfm_positions(self: CoinbaseClientProtocol) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "list_cfm_positions not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced with derivatives enabled."
            )
        path = self._get_endpoint_path("cfm_positions")
        return self._request("GET", path)

    # Renamed to avoid collision with CoinbaseClient.list_positions (which returns domain objects)
    def list_positions_raw(self: CoinbaseClientProtocol) -> dict[str, Any]:
        return self.list_cfm_positions()

    def get_cfm_position(self: CoinbaseClientProtocol, product_id: str) -> dict[str, Any]:
        if self.api_mode == "exchange":
            raise InvalidRequestError(
                "get_cfm_position not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced with derivatives enabled."
            )
        path = self._get_endpoint_path("cfm_position", product_id=product_id)
        return self._request("GET", path)

    def intx_allocate(self: CoinbaseClientProtocol, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._get_endpoint_path("intx_allocate")
        return self._request("POST", path, payload)

    def intx_balances(self: CoinbaseClientProtocol, portfolio_uuid: str) -> dict[str, Any]:
        path = self._get_endpoint_path("intx_balances", portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    def intx_portfolio(self: CoinbaseClientProtocol, portfolio_uuid: str) -> dict[str, Any]:
        path = self._get_endpoint_path("intx_portfolio", portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    # Alias for intx_portfolio
    def get_intx_portfolio(self: CoinbaseClientProtocol, portfolio_uuid: str) -> dict[str, Any]:
        return self.intx_portfolio(portfolio_uuid)

    def intx_positions(self: CoinbaseClientProtocol, portfolio_uuid: str) -> dict[str, Any]:
        path = self._get_endpoint_path("intx_positions", portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    # Alias for intx_positions
    def list_intx_positions(self: CoinbaseClientProtocol, portfolio_uuid: str) -> dict[str, Any]:
        return self.intx_positions(portfolio_uuid)

    def intx_position(
        self: CoinbaseClientProtocol, portfolio_uuid: str, symbol: str
    ) -> dict[str, Any]:
        path = self._get_endpoint_path(
            "intx_position", portfolio_uuid=portfolio_uuid, symbol=symbol
        )
        return self._request("GET", path)

    # Alias for intx_position
    def get_intx_position(
        self: CoinbaseClientProtocol, portfolio_uuid: str, symbol: str
    ) -> dict[str, Any]:
        return self.intx_position(portfolio_uuid, symbol)

    def intx_multi_asset_collateral(self: CoinbaseClientProtocol) -> dict[str, Any]:
        path = self._get_endpoint_path("intx_multi_asset_collateral")
        return self._request("GET", path)

    # Alias for intx_multi_asset_collateral
    def get_intx_multi_asset_collateral(self: CoinbaseClientProtocol) -> dict[str, Any]:
        return self.intx_multi_asset_collateral()

    # ------------------------------------------------------------------
    # Clearing & Treasury (CFM)
    # ------------------------------------------------------------------
    def cfm_balance_summary(self: CoinbaseClientProtocol) -> dict[str, Any]:
        path = self._get_endpoint_path("cfm_balance_summary")
        return self._request("GET", path)

    def cfm_positions(self: CoinbaseClientProtocol) -> dict[str, Any]:
        path = self._get_endpoint_path("cfm_positions")
        return self._request("GET", path)

    def cfm_position(self: CoinbaseClientProtocol, product_id: str) -> dict[str, Any]:
        path = self._get_endpoint_path("cfm_position", product_id=product_id)
        return self._request("GET", path)

    def cfm_sweeps(self: CoinbaseClientProtocol) -> dict[str, Any]:
        path = self._get_endpoint_path("cfm_sweeps")
        return self._request("GET", path)

    def cfm_sweeps_schedule(self: CoinbaseClientProtocol) -> dict[str, Any]:
        path = self._get_endpoint_path("cfm_schedule_sweep")
        return self._request("GET", path)

    def cfm_intraday_current_margin_window(self: CoinbaseClientProtocol) -> dict[str, Any]:
        path = self._get_endpoint_path("cfm_intraday_current_margin_window")
        return self._request("GET", path)

    def cfm_intraday_margin_setting(
        self: CoinbaseClientProtocol, payload: dict[str, Any]
    ) -> dict[str, Any]:
        path = self._get_endpoint_path("cfm_intraday_margin_setting")
        return self._request("POST", path, payload)


__all__ = ["PortfolioClientMixin"]
