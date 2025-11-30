"""Portfolio management service for Coinbase REST API.

This service handles balance and position management with explicit
dependencies injected via constructor, replacing the PortfolioRestMixin.

Implements the PositionProvider protocol for use by OrderService.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.models import to_position
from gpt_trader.features.brokerages.core.interfaces import Balance, InvalidRequestError, Position
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_portfolio")


class PortfolioService:
    """Handles balance and position management.

    Implements PositionProvider protocol for use by OrderService.

    Dependencies:
        client: CoinbaseClient for API calls
        endpoints: CoinbaseEndpoints for mode detection
        event_store: EventStore for telemetry
    """

    def __init__(
        self,
        *,
        client: CoinbaseClient,
        endpoints: CoinbaseEndpoints,
        event_store: EventStore,
    ) -> None:
        self._client = client
        self._endpoints = endpoints
        self._event_store = event_store

    def list_balances(self) -> list[Balance]:
        """List all balances."""
        balances = []
        try:
            response = self._client.get_accounts()
            if isinstance(response, list):
                accounts = response
            else:
                accounts = response.get("accounts", [])

            for acc in accounts:
                try:
                    currency = acc.get("currency")

                    # Handle different payload shapes (advanced vs exchange)
                    avail_val = (
                        acc.get("available_balance", {}).get("value") or acc.get("available") or "0"
                    )
                    available = Decimal(str(avail_val))

                    hold_val = "0"
                    if isinstance(acc.get("hold"), dict):
                        hold_val = acc.get("hold", {}).get("value") or "0"
                    else:
                        hold_val = acc.get("hold") or "0"
                    hold = Decimal(str(hold_val))

                    total_val = "0"
                    if isinstance(acc.get("balance"), dict):
                        total_val = acc.get("balance", {}).get("value")
                    else:
                        total_val = acc.get("balance") or acc.get("total_balance")

                    if total_val:
                        total = Decimal(str(total_val))
                    else:
                        total = available + hold

                    balances.append(
                        Balance(asset=currency, total=total, available=available, hold=hold)
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to parse account balance entry",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        operation="list_balances",
                        account_id=acc.get("uuid", "unknown"),
                    )
                    continue
        except Exception as exc:
            logger.error(
                "Failed to fetch accounts from broker",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="list_balances",
            )
        return balances

    def get_portfolio_balances(self) -> list[Balance]:
        """Get portfolio balances. Fallback to list_balances if not available."""
        return self.list_balances()

    def list_positions(self) -> list[Position]:
        """List all open positions.

        This method satisfies the PositionProvider protocol.
        """
        positions = []
        try:
            if self._endpoints.supports_derivatives():
                # Client returns list[Position] now
                positions = self._client.list_positions()
        except Exception as exc:
            logger.error(
                "Failed to list positions from broker",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="list_positions",
            )
        return positions

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        try:
            if self._endpoints.supports_derivatives():
                # For get_position, client mixin returns dict (get_cfm_position)
                # But client doesn't override it to return Position object?
                # Let's use the raw mixin method exposed via client
                response = self._client.get_cfm_position(product_id=symbol)
                return to_position(response)
        except Exception as exc:
            logger.error(
                "Failed to get position from broker",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_position",
                symbol=symbol,
            )
        return None

    def intx_allocate(self, amount_dict: dict[str, Any]) -> dict[str, Any]:
        """Allocate funds to/from INTX portfolio."""
        if self._endpoints.mode != "advanced":
            raise InvalidRequestError("INTX allocation requires advanced mode")

        try:
            response = self._client.intx_allocate(amount_dict)

            # Normalize decimals in response
            if "allocated_amount" in response:
                response["allocated_amount"] = Decimal(str(response["allocated_amount"]))
            if "source_amount" in response:
                response["source_amount"] = Decimal(str(response["source_amount"]))
            if "target_amount" in response:
                response["target_amount"] = Decimal(str(response["target_amount"]))

            self._event_store.append_metric(
                metrics={"event_type": "intx_allocation", "response": response}
            )
            return cast(dict[str, Any], response)
        except Exception as e:
            raise e

    def get_intx_balances(self, portfolio_id: str) -> list[dict[str, Any]]:
        """Get INTX portfolio balances."""
        if self._endpoints.mode != "advanced":
            return []  # Or raise, test expects empty if not advanced

        try:
            response = self._client.get_intx_portfolio(portfolio_id)
            balances = response.get("balances", [])
            for b in balances:
                if "amount" in b:
                    b["amount"] = Decimal(str(b["amount"]))
                if "hold" in b:
                    b["hold"] = Decimal(str(b["hold"]))

            self._event_store.append_metric(
                metrics={
                    "event_type": "intx_balances",
                    "data": {"portfolio_id": portfolio_id, "count": len(balances)},
                }
            )
            return cast(list[dict[str, Any]], balances)
        except Exception as exc:
            logger.error(
                "Failed to get INTX balances",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_intx_balances",
                portfolio_id=portfolio_id,
            )
            return []

    def get_intx_portfolio(self, portfolio_id: str) -> dict[str, Any]:
        """Get INTX portfolio details."""
        if self._endpoints.mode != "advanced":
            return {}
        try:
            response = self._client.get_intx_portfolio(portfolio_id)
            if "portfolio_value" in response:
                response["portfolio_value"] = Decimal(str(response["portfolio_value"]))
            return cast(dict[str, Any], response)
        except Exception as exc:
            logger.error(
                "Failed to get INTX portfolio",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_intx_portfolio",
                portfolio_id=portfolio_id,
            )
            return {}

    def list_intx_positions(self, portfolio_id: str) -> list[Position]:
        """List INTX positions."""
        if self._endpoints.mode != "advanced":
            return []
        try:
            response = self._client.list_intx_positions(portfolio_id)
            positions = []
            for p in response.get("positions", []):
                positions.append(to_position(p))
            return positions
        except Exception as exc:
            logger.error(
                "Failed to list INTX positions",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="list_intx_positions",
                portfolio_id=portfolio_id,
            )
            return []

    def get_intx_position(self, portfolio_id: str, symbol: str) -> Position | None:
        """Get a single INTX position."""
        if self._endpoints.mode != "advanced":
            return None
        try:
            response = self._client.get_intx_position(portfolio_id, symbol)
            return to_position(response)
        except Exception as exc:
            logger.error(
                "Failed to get INTX position",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_intx_position",
                portfolio_id=portfolio_id,
                symbol=symbol,
            )
            return None

    def get_intx_multi_asset_collateral(self) -> dict[str, Any]:
        """Get INTX multi-asset collateral details."""
        if self._endpoints.mode != "advanced":
            return {}
        try:
            response = self._client.get_intx_multi_asset_collateral()
            if "total_usd_value" in response:
                response["total_usd_value"] = Decimal(str(response["total_usd_value"]))
            self._event_store.append_metric(
                metrics={"event_type": "intx_multi_asset_collateral", "data": response}
            )
            return cast(dict[str, Any], response)
        except Exception as exc:
            logger.error(
                "Failed to get INTX multi-asset collateral",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_intx_multi_asset_collateral",
            )
            return {}

    def get_cfm_balance_summary(self) -> dict[str, Any]:
        """Get CFM balance summary."""
        if not self._endpoints.supports_derivatives():
            return {}

        response = self._client.cfm_balance_summary()
        summary = response.get("balance_summary", {})

        # Convert numeric fields
        for key in ["total_balance", "available_balance", "portfolio_value", "available_margin"]:
            if key in summary:
                summary[key] = Decimal(str(summary[key]))

        self._event_store.append_metric(
            metrics={
                "event_type": "cfm_balance_summary",
                "summary": {k: str(v) for k, v in summary.items()},
            }
        )
        return cast(dict[str, Any], summary)

    def list_cfm_sweeps(self) -> list[dict[str, Any]]:
        """List CFM sweeps."""
        if not self._endpoints.supports_derivatives():
            return []

        response = self._client.cfm_sweeps()
        sweeps = response.get("sweeps", [])

        processed_sweeps = []
        for s in sweeps:
            if "amount" in s:
                s["amount"] = Decimal(str(s["amount"]))
            processed_sweeps.append(s)

        self._event_store.append_metric(
            metrics={"event_type": "cfm_sweeps", "count": len(processed_sweeps)}
        )
        return processed_sweeps

    def get_cfm_sweeps_schedule(self) -> dict[str, Any]:
        """Get CFM sweeps schedule."""
        if not self._endpoints.supports_derivatives():
            return {}

        try:
            response = self._client.cfm_sweeps_schedule()
            return cast(dict[str, Any], response.get("schedule", {}))
        except Exception as exc:
            logger.error(
                "Failed to get CFM sweeps schedule",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_cfm_sweeps_schedule",
            )
            return {}

    def get_cfm_margin_window(self) -> dict[str, Any]:
        """Get current CFM margin window."""
        if not self._endpoints.supports_derivatives():
            return {}

        try:
            response = self._client.cfm_intraday_current_margin_window()
            return cast(dict[str, Any], response)
        except Exception as exc:
            logger.error(
                "Failed to get CFM margin window",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_cfm_margin_window",
            )
            return {}

    def update_cfm_margin_window(
        self,
        margin_window: str,
        effective_time: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update CFM margin window."""
        if not self._endpoints.supports_derivatives():
            raise InvalidRequestError("Derivatives not supported")

        payload = {"margin_window": margin_window}
        if effective_time:
            payload["effective_time"] = effective_time

        response = self._client.cfm_intraday_margin_setting(payload)

        if "leverage" in response:
            response["leverage"] = Decimal(str(response["leverage"]))

        self._event_store.append_metric(
            metrics={
                "event_type": "cfm_margin_setting",
                "margin_window": margin_window,
                "response": {k: str(v) for k, v in response.items()},
            }
        )
        return cast(dict[str, Any], response)
