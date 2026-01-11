"""Portfolio management service for Coinbase REST API.

This service handles balance and position management with explicit
dependencies injected via constructor, replacing the PortfolioRestMixin.

Implements the PositionProvider protocol for use by OrderService.
Supports unified portfolio view across spot and CFM futures.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, cast

from gpt_trader.core import Balance, InvalidRequestError, Position
from gpt_trader.core.account import CFMBalance, UnifiedBalance
from gpt_trader.core.account import Position as CorePosition
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.models import to_position
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.telemetry import emit_metric

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
        logger.debug("PortfolioService.list_balances: Fetching accounts from Coinbase API")
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

        # Log final balance parsing summary
        if balances:
            logger.debug(
                f"PortfolioService.list_balances: Successfully parsed {len(balances)} balances"
            )
        else:
            logger.warning(
                "PortfolioService.list_balances: No balances parsed from API response. "
                "Check API portfolio selection or account permissions."
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

            emit_metric(
                self._event_store,
                "coinbase_portfolio",
                {"event_type": "intx_allocation", "response": response},
                logger=logger,
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

            emit_metric(
                self._event_store,
                "coinbase_portfolio",
                {
                    "event_type": "intx_balances",
                    "data": {"portfolio_id": portfolio_id, "count": len(balances)},
                },
                logger=logger,
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
            emit_metric(
                self._event_store,
                "coinbase_portfolio",
                {"event_type": "intx_multi_asset_collateral", "data": response},
                logger=logger,
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

        emit_metric(
            self._event_store,
            "coinbase_portfolio",
            {
                "event_type": "cfm_balance_summary",
                "summary": {k: str(v) for k, v in summary.items()},
            },
            logger=logger,
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

        emit_metric(
            self._event_store,
            "coinbase_portfolio",
            {"event_type": "cfm_sweeps", "count": len(processed_sweeps)},
            logger=logger,
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

        emit_metric(
            self._event_store,
            "coinbase_portfolio",
            {
                "event_type": "cfm_margin_setting",
                "margin_window": margin_window,
                "response": {k: str(v) for k, v in response.items()},
            },
            logger=logger,
        )
        return cast(dict[str, Any], response)

    # -------------------------------------------------------------------------
    # Unified Portfolio Methods (Spot + CFM)
    # -------------------------------------------------------------------------

    def has_cfm_access(self) -> bool:
        """Check if CFM futures access is available."""
        if not self._endpoints.supports_derivatives():
            return False
        try:
            response = self._client.cfm_balance_summary()
            return "balance_summary" in response
        except Exception:
            return False

    def list_cfm_positions(self) -> list[CorePosition]:
        """List CFM futures positions as core Position objects."""
        if not self._endpoints.supports_derivatives():
            return []

        try:
            response = self._client.cfm_positions()
            positions = []

            for p in response.get("positions", []):
                # Parse expiration time
                expiry = None
                if p.get("expiration_time"):
                    try:
                        expiry = datetime.fromisoformat(p["expiration_time"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass

                pos = CorePosition(
                    symbol=p.get("product_id", ""),
                    quantity=Decimal(str(p.get("number_of_contracts", "0"))),
                    entry_price=Decimal(str(p.get("avg_entry_price", "0"))),
                    mark_price=Decimal(str(p.get("current_price", "0"))),
                    unrealized_pnl=Decimal(str(p.get("unrealized_pnl", "0"))),
                    realized_pnl=Decimal(str(p.get("daily_realized_pnl", "0"))),
                    side=p.get("side", "LONG").lower(),
                    leverage=None,  # CFM doesn't expose per-position leverage
                    liquidation_price=None,  # Would need separate calculation
                    product_type="FUTURE",
                    contract_expiry=expiry,
                )
                positions.append(pos)

            return positions
        except Exception as exc:
            logger.error(
                "Failed to list CFM positions",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="list_cfm_positions",
            )
            return []

    def list_spot_positions_as_core(self) -> list[CorePosition]:
        """List spot positions as core Position objects.

        Note: Spot positions are derived from non-zero balances.
        """
        positions = []
        try:
            balances = self.list_balances()
            for bal in balances:
                if bal.asset == "USD" or bal.total == 0:
                    continue
                # Spot "positions" are just holdings - no entry price tracking
                # Mark price would need to be fetched separately
                positions.append(
                    CorePosition(
                        symbol=f"{bal.asset}-USD",
                        quantity=bal.total,
                        entry_price=Decimal("0"),  # Unknown for spot
                        mark_price=Decimal("0"),  # Would need price fetch
                        unrealized_pnl=Decimal("0"),
                        realized_pnl=Decimal("0"),
                        side="long",
                        leverage=1,
                        product_type="SPOT",
                    )
                )
        except Exception as exc:
            logger.error(
                "Failed to list spot positions",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="list_spot_positions_as_core",
            )
        return positions

    def list_all_positions(self) -> list[CorePosition]:
        """List all positions across spot and CFM.

        Returns unified list of CorePosition objects with product_type
        indicating whether each is SPOT or FUTURE.
        """
        positions: list[CorePosition] = []

        # Add spot positions (from balances)
        positions.extend(self.list_spot_positions_as_core())

        # Add CFM positions if available
        if self._endpoints.supports_derivatives():
            positions.extend(self.list_cfm_positions())

        logger.debug(
            "Listed all positions",
            spot_count=len([p for p in positions if p.product_type == "SPOT"]),
            cfm_count=len([p for p in positions if p.product_type == "FUTURE"]),
        )

        return positions

    def get_cfm_balance(self) -> CFMBalance | None:
        """Get CFM balance as a structured CFMBalance object."""
        if not self._endpoints.supports_derivatives():
            return None

        try:
            response = self._client.cfm_balance_summary()
            summary = response.get("balance_summary", {})

            if not summary:
                return None

            def get_value(field: str) -> Decimal:
                """Extract decimal value from nested dict or direct value."""
                val = summary.get(field, {})
                if isinstance(val, dict):
                    return Decimal(str(val.get("value", "0")))
                return Decimal(str(val or "0"))

            return CFMBalance(
                futures_buying_power=get_value("futures_buying_power"),
                total_usd_balance=get_value("total_usd_balance"),
                available_margin=get_value("available_margin"),
                initial_margin=get_value("initial_margin"),
                unrealized_pnl=get_value("unrealized_pnl"),
                daily_realized_pnl=get_value("daily_realized_pnl"),
                liquidation_threshold=get_value("liquidation_threshold"),
                liquidation_buffer_amount=get_value("liquidation_buffer_amount"),
                liquidation_buffer_percentage=float(
                    summary.get("liquidation_buffer_percentage", "0")
                ),
            )
        except Exception as exc:
            logger.error(
                "Failed to get CFM balance",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="get_cfm_balance",
            )
            return None

    def get_unified_balance(self) -> UnifiedBalance:
        """Get combined balance across spot and CFM.

        Returns UnifiedBalance with totals from both trading venues.
        """
        # Get spot balance (USD available)
        spot_balance = Decimal("0")
        try:
            balances = self.list_balances()
            for bal in balances:
                if bal.asset == "USD":
                    spot_balance = bal.available
                    break
        except Exception:
            pass

        # Get CFM balance
        cfm_balance = Decimal("0")
        cfm_available_margin = Decimal("0")
        cfm_buying_power = Decimal("0")

        cfm = self.get_cfm_balance()
        if cfm:
            cfm_balance = cfm.total_usd_balance
            cfm_available_margin = cfm.available_margin
            cfm_buying_power = cfm.futures_buying_power

        return UnifiedBalance(
            spot_balance=spot_balance,
            cfm_balance=cfm_balance,
            cfm_available_margin=cfm_available_margin,
            cfm_buying_power=cfm_buying_power,
            total_equity=spot_balance + cfm_balance,
        )
