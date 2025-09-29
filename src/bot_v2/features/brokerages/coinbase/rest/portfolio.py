"""Account, balance, and portfolio helpers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from bot_v2.utilities.quantities import quantity_from

from ...core.interfaces import Balance, Position
from .base import logger


class PortfolioRestMixin:
    """Account surface area exposed by the Coinbase REST service."""

    def list_balances(self) -> list[Balance]:
        raw = self.client.get_accounts() or {}
        accounts = raw if isinstance(raw, list) else raw.get("accounts") or raw.get("data") or []
        balances: list[Balance] = []

        for account in accounts:
            try:
                currency = account.get("currency") or account.get("asset") or ""

                available_balance = account.get("available_balance", {})
                if isinstance(available_balance, dict) and available_balance:
                    available = available_balance.get("value", "0")
                else:
                    fallback = account.get("available")
                    available = (
                        str(fallback) if fallback is not None else str(available_balance or "0")
                    )

                hold_data = account.get("hold", {})
                if isinstance(hold_data, dict) and hold_data:
                    hold = hold_data.get("value", "0")
                else:
                    hold = str(hold_data) if hold_data else str(account.get("hold", "0"))

                total_field = account.get("balance") or account.get("total_balance")
                if isinstance(total_field, dict):
                    total = total_field.get("value", "0")
                elif total_field is not None:
                    total = str(total_field)
                else:
                    total = str(Decimal(str(available)) + Decimal(str(hold)))

                total_decimal = Decimal(str(total))
                available_decimal = Decimal(str(available))
                hold_decimal = Decimal(str(hold))

                if total_decimal > 0 or currency in {"USD", "USDC", "EUR", "GBP"}:
                    balances.append(
                        Balance(
                            asset=str(currency),
                            total=total_decimal,
                            available=available_decimal,
                            hold=hold_decimal,
                        )
                    )
            except (ValueError, InvalidOperation) as exc:
                logger.warning(
                    "Could not parse balance for %s: %s", account.get("currency", "unknown"), exc
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Error processing account %s: %s", account.get("currency", "unknown"), exc
                )
                continue

        return balances

    def get_portfolio_balances(self) -> list[Balance]:
        try:
            accounts_data = self.client.get_accounts() or {}
            accounts = accounts_data.get("accounts", [])
            if not accounts:
                return self.list_balances()
            portfolio_id = accounts[0].get("retail_portfolio_id")
            if not portfolio_id:
                return self.list_balances()
            breakdown = self.client.get_portfolio_breakdown(portfolio_id)
            if not breakdown:
                return self.list_balances()
        except Exception as exc:
            logger.warning(
                "Could not get portfolio breakdown, falling back to account balances: %s", exc
            )
            return self.list_balances()

        balances: list[Balance] = []
        breakdown_data = breakdown.get("breakdown", {})
        for position in breakdown_data.get("spot_positions", []):
            asset = position.get("asset", "")
            balance_crypto = position.get("total_balance_crypto", 0)
            if isinstance(balance_crypto, dict):
                amount = Decimal(str(balance_crypto.get("value", 0)))
            else:
                amount = Decimal(str(balance_crypto or 0))
            hold_crypto = position.get("hold", 0)
            if isinstance(hold_crypto, dict):
                hold = Decimal(str(hold_crypto.get("value", 0)))
            else:
                hold = Decimal(str(hold_crypto or 0))
            available = amount - hold
            if amount > 0 or asset in {"USD", "USDC", "EUR", "GBP"}:
                balances.append(
                    Balance(
                        asset=asset,
                        total=amount,
                        available=available,
                        hold=hold,
                    )
                )
        return balances

    def get_key_permissions(self) -> dict[str, Any]:
        data = self.client.get_key_permissions() or {}
        return data.get("key_permissions") or data

    def get_fee_schedule(self) -> dict[str, Any]:
        return self.client.get_fees() or {}

    def get_account_limits(self) -> dict[str, Any]:
        data = self.client.get_limits() or {}
        return data.get("limits") or data

    def get_transaction_summary(self) -> dict[str, Any]:
        return self.client.get_transaction_summary() or {}

    def list_payment_methods(self) -> list[dict[str, Any]]:
        data = self.client.list_payment_methods() or {}
        return list(data.get("payment_methods") or data.get("data") or [])

    def get_payment_method(self, payment_method_id: str) -> dict[str, Any]:
        data = self.client.get_payment_method(payment_method_id) or {}
        return data.get("payment_method") or data

    def list_portfolios(self) -> list[dict[str, Any]]:
        data = self.client.list_portfolios() or {}
        return list(data.get("portfolios") or data.get("data") or [])

    def get_portfolio(self, portfolio_uuid: str) -> dict[str, Any]:
        data = self.client.get_portfolio(portfolio_uuid) or {}
        return data.get("portfolio") or data

    def get_portfolio_breakdown(self, portfolio_uuid: str) -> dict[str, Any]:
        data = self.client.get_portfolio_breakdown(portfolio_uuid) or {}
        return data.get("breakdown") or data

    def move_portfolio_funds(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.move_funds(payload) or {}

    def create_convert_quote(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.convert_quote(payload) or {}

    def commit_convert_trade(
        self, trade_id: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.client.commit_convert_trade(trade_id, payload or {}) or {}

    def get_convert_trade(self, trade_id: str) -> dict[str, Any]:
        return self.client.get_convert_trade(trade_id) or {}

    def list_positions(self) -> list[Position]:
        if not self.endpoints.supports_derivatives():
            return []
        try:
            data = self.client.list_positions() or {}
        except Exception as exc:
            logger.error("Failed to list positions: %s", exc)
            return []
        items = data.get("positions") or data.get("data") or []
        mapped: list[Position] = []
        for item in items:
            position = self._map_position(item)
            if position is not None:
                mapped.append(position)
        return mapped

    def get_position(self, symbol: str) -> Position | None:
        if not self.endpoints.supports_derivatives():
            return None
        try:
            data = self.client.get(self.endpoints.get_position(symbol))
        except Exception as exc:
            logger.debug("No position found for %s: %s", symbol, exc)
            return None
        return self._map_position(data)

    def _map_position(self, data: dict[str, Any]) -> Position | None:
        try:
            quantity_value = quantity_from(data, default=None)
            if quantity_value is None:
                fallback_size = data.get("size") or data.get("contracts") or "0"
                quantity_value = Decimal(str(fallback_size))
            quantity = Decimal(str(quantity_value))
            entry_price = Decimal(
                str(data.get("entry_price") or data.get("avg_entry_price") or "0")
            )
            mark_price = Decimal(
                str(
                    data.get("mark_price")
                    or data.get("index_price")
                    or data.get("current_price")
                    or entry_price
                )
            )
            unrealized = Decimal(
                str(data.get("unrealized_pnl") or data.get("unrealizedPnl") or "0")
            )
            realized = Decimal(str(data.get("realized_pnl") or data.get("realizedPnl") or "0"))
            leverage = data.get("leverage")
            side = str(data.get("side", "")).lower()
            if not side:
                side = "long" if quantity >= 0 else "short"
            return Position(
                symbol=data.get("product_id", ""),
                quantity=abs(quantity),
                entry_price=entry_price,
                mark_price=mark_price,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
                leverage=int(leverage) if leverage is not None else None,
                side=side,
            )
        except Exception as exc:
            logger.error("Failed to map position payload: %s", exc)
            return None


__all__ = ["PortfolioRestMixin"]
