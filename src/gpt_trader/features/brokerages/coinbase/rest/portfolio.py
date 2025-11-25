"""
Portfolio management mixin for Coinbase REST service.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from gpt_trader.features.brokerages.coinbase.models import to_position
from gpt_trader.features.brokerages.core.interfaces import Balance, InvalidRequestError, Position


class PortfolioRestMixin:
    """Mixin for portfolio management operations."""

    def list_balances(self) -> List[Balance]:
        """List all balances."""
        balances = []
        try:
            response = self.client.get_accounts()
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
                except Exception:
                    continue
        except Exception:
            pass
        return balances

    def get_portfolio_balances(self) -> List[Balance]:
        """Get portfolio balances. Fallback to list_balances if not available."""
        return self.list_balances()

    def list_positions(self) -> List[Position]:
        """List all open positions."""
        positions = []
        try:
            if self.endpoints.supports_derivatives():
                response = self.client.list_positions()
                raw_positions = response.get("positions", [])
                for p in raw_positions:
                    positions.append(to_position(p))
        except Exception:
            pass
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        try:
            if self.endpoints.supports_derivatives():
                response = self.client.get_position(product_id=symbol)
                return to_position(response)
        except Exception:
            pass
        return None

    def intx_allocate(self, amount_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate funds to/from INTX portfolio."""
        if self.endpoints.mode != "advanced":
            raise InvalidRequestError("INTX allocation requires advanced mode")

        try:
            response = self.client.intx_allocate(amount_dict)

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
            return response
        except Exception as e:
            raise e

    def get_intx_balances(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get INTX portfolio balances."""
        if self.endpoints.mode != "advanced":
            return []  # Or raise, test expects empty if not advanced

        try:
            response = self.client.get_intx_portfolio(portfolio_id)
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
            return balances
        except Exception:
            return []

    def get_intx_portfolio(self, portfolio_id: str) -> Dict[str, Any]:
        """Get INTX portfolio details."""
        if self.endpoints.mode != "advanced":
            return {}
        try:
            response = self.client.get_intx_portfolio(portfolio_id)
            if "portfolio_value" in response:
                response["portfolio_value"] = Decimal(str(response["portfolio_value"]))
            return response
        except Exception:
            return {}

    def list_intx_positions(self, portfolio_id: str) -> List[Position]:
        """List INTX positions."""
        if self.endpoints.mode != "advanced":
            return []
        try:
            response = self.client.list_intx_positions(portfolio_id)
            positions = []
            for p in response.get("positions", []):
                positions.append(to_position(p))
            return positions
        except Exception:
            return []

    def get_intx_position(self, portfolio_id: str, symbol: str) -> Optional[Position]:
        """Get a single INTX position."""
        if self.endpoints.mode != "advanced":
            return None
        try:
            response = self.client.get_intx_position(portfolio_id, symbol)
            return to_position(response)
        except Exception:
            return None

    def get_intx_multi_asset_collateral(self) -> Dict[str, Any]:
        """Get INTX multi-asset collateral details."""
        if self.endpoints.mode != "advanced":
            return {}
        try:
            response = self.client.get_intx_multi_asset_collateral()
            if "total_usd_value" in response:
                response["total_usd_value"] = Decimal(str(response["total_usd_value"]))
            self._event_store.append_metric(
                metrics={"event_type": "intx_multi_asset_collateral", "data": response}
            )
            return response
        except Exception:
            return {}

    def get_cfm_balance_summary(self) -> Dict[str, Any]:
        """Get CFM balance summary."""
        if not self.endpoints.supports_derivatives():
            return {}

        response = self.client.cfm_balance_summary()
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
        return summary

    def list_cfm_sweeps(self) -> List[Dict[str, Any]]:
        """List CFM sweeps."""
        if not self.endpoints.supports_derivatives():
            return []

        response = self.client.cfm_sweeps()
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

    def get_cfm_sweeps_schedule(self) -> Dict[str, Any]:
        """Get CFM sweeps schedule."""
        if not self.endpoints.supports_derivatives():
            return {}

        try:
            response = self.client.cfm_sweeps_schedule()
            return response.get("schedule", {})
        except Exception:
            return {}

    def get_cfm_margin_window(self) -> Dict[str, Any]:
        """Get current CFM margin window."""
        if not self.endpoints.supports_derivatives():
            return {}

        try:
            response = self.client.cfm_intraday_current_margin_window()
            return response
        except Exception:
            return {}

    def update_cfm_margin_window(
        self,
        margin_window: str,
        effective_time: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update CFM margin window."""
        if not self.endpoints.supports_derivatives():
            raise InvalidRequestError("Derivatives not supported")

        payload = {"margin_window": margin_window}
        if effective_time:
            payload["effective_time"] = effective_time

        response = self.client.cfm_intraday_margin_setting(payload)

        if "leverage" in response:
            response["leverage"] = Decimal(str(response["leverage"]))

        self._event_store.append_metric(
            metrics={
                "event_type": "cfm_margin_setting",
                "margin_window": margin_window,
                "response": {k: str(v) for k, v in response.items()},
            }
        )
        return response
