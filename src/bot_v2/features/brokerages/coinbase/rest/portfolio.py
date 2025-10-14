"""Account, balance, and portfolio helpers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Literal, cast

from bot_v2.features.brokerages.coinbase.models import normalize_symbol
from bot_v2.features.brokerages.coinbase.rest.base import logger
from bot_v2.features.brokerages.core.interfaces import Balance, InvalidRequestError, Position
from bot_v2.utilities.quantities import quantity_from
from bot_v2.utilities.telemetry import emit_metric

if TYPE_CHECKING:
    from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
    from bot_v2.features.brokerages.coinbase.endpoints import CoinbaseEndpoints


class PortfolioRestMixin:
    """Account surface area exposed by the Coinbase REST service."""

    client: CoinbaseClient
    endpoints: CoinbaseEndpoints
    _intx_portfolio_cache: str | None = None
    _intx_balances_seed: list[dict[str, Any]] | None = None

    @staticmethod
    def _normalize_numeric(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError):
                return value
        if isinstance(value, str):
            try:
                return Decimal(value)
            except (InvalidOperation, ValueError):
                return value
        if isinstance(value, dict):
            return {key: PortfolioRestMixin._normalize_numeric(val) for key, val in value.items()}
        if isinstance(value, list):
            return [PortfolioRestMixin._normalize_numeric(item) for item in value]
        return value

    @staticmethod
    def _decimals_to_str(value: Any) -> Any:
        if isinstance(value, Decimal):
            return format(value, "f")
        if isinstance(value, dict):
            return {key: PortfolioRestMixin._decimals_to_str(val) for key, val in value.items()}
        if isinstance(value, list):
            return [PortfolioRestMixin._decimals_to_str(item) for item in value]
        return value

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

    # ------------------------------------------------------------------
    # Institutional (INTX)
    # ------------------------------------------------------------------
    def intx_allocate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._intx_supported():
            raise InvalidRequestError(
                "INTX endpoints require advanced mode with institutional access."
            )
        try:
            response = self.client.intx_allocate(payload) or {}
        except Exception as exc:
            logger.error("Failed to allocate INTX collateral: %s", exc)
            raise

        normalized = cast(dict[str, Any], self._normalize_numeric(response))
        emit_metric(
            getattr(self, "_event_store", None),
            "coinbase_perps",
            {
                "event_type": "intx_allocate",
                "timestamp": datetime.utcnow().isoformat(),
                "request": self._decimals_to_str(payload),
                "response": self._decimals_to_str(normalized),
            },
            logger=logger,
        )
        return normalized

    def get_intx_balances(self, portfolio_uuid: str) -> list[dict[str, Any]]:
        if not self._intx_supported():
            return []
        try:
            payload = self.client.intx_balances(portfolio_uuid) or {}
        except Exception as exc:
            logger.error("Failed to fetch INTX balances for %s: %s", portfolio_uuid, exc)
            return []
        raw_balances = payload.get("balances") or payload.get("data") or []
        balances: list[dict[str, Any]] = []
        for entry in raw_balances:
            if isinstance(entry, dict):
                balances.append(cast(dict[str, Any], self._normalize_numeric(entry)))
        if balances:
            emit_metric(
                getattr(self, "_event_store", None),
                "coinbase_perps",
                {
                    "event_type": "intx_balances",
                    "timestamp": datetime.utcnow().isoformat(),
                    "portfolio_uuid": portfolio_uuid,
                    "balances": self._decimals_to_str(balances),
                },
                logger=logger,
            )
        return balances

    def get_intx_portfolio(self, portfolio_uuid: str) -> dict[str, Any]:
        if not self._intx_supported():
            return {}
        try:
            payload = self.client.intx_portfolio(portfolio_uuid) or {}
        except Exception as exc:
            logger.error("Failed to fetch INTX portfolio %s: %s", portfolio_uuid, exc)
            return {}
        raw_portfolio = payload.get("portfolio") or payload
        return cast(dict[str, Any], self._normalize_numeric(raw_portfolio))

    def list_intx_positions(self, portfolio_uuid: str) -> list[dict[str, Any]]:
        if not self._intx_supported():
            return []
        try:
            payload = self.client.intx_positions(portfolio_uuid) or {}
        except Exception as exc:
            logger.error("Failed to fetch INTX positions for %s: %s", portfolio_uuid, exc)
            return []
        raw_positions = payload.get("positions") or payload.get("data") or []
        positions: list[dict[str, Any]] = []
        for entry in raw_positions:
            if isinstance(entry, dict):
                positions.append(cast(dict[str, Any], self._normalize_numeric(entry)))
        return positions

    def get_intx_position(self, portfolio_uuid: str, symbol: str) -> dict[str, Any]:
        if not self._intx_supported():
            return {}
        try:
            payload = self.client.intx_position(portfolio_uuid, symbol) or {}
        except Exception as exc:
            logger.debug("No INTX position for %s/%s: %s", portfolio_uuid, symbol, exc)
            return {}
        return cast(dict[str, Any], self._normalize_numeric(payload))

    def get_intx_multi_asset_collateral(self) -> dict[str, Any]:
        if not self._intx_supported():
            return {}
        try:
            payload = self.client.intx_multi_asset_collateral() or {}
        except Exception as exc:
            logger.error("Failed to fetch INTX multi-asset collateral: %s", exc)
            return {}
        normalized = cast(dict[str, Any], self._normalize_numeric(payload))
        emit_metric(
            getattr(self, "_event_store", None),
            "coinbase_perps",
            {
                "event_type": "intx_multi_asset_collateral",
                "timestamp": datetime.utcnow().isoformat(),
                "collateral": self._decimals_to_str(normalized),
            },
            logger=logger,
        )
        return normalized

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
            product_id = normalize_symbol(symbol)
            data = self.client.get_position(product_id) or {}
        except Exception as exc:
            logger.debug("No position found for %s: %s", symbol, exc)
            return None
        return self._map_position(data)

    # ------------------------------------------------------------------
    # CFM telemetry helpers
    # ------------------------------------------------------------------
    def get_cfm_balance_summary(self) -> dict[str, Any]:
        if not self.endpoints.supports_derivatives():
            return {}
        try:
            payload = self.client.cfm_balance_summary() or {}
        except Exception as exc:
            logger.error("Failed to fetch CFM balance summary: %s", exc)
            return {}

        raw_summary = payload.get("balance_summary") or payload
        summary = cast(dict[str, Any], self._normalize_numeric(raw_summary))
        timestamp = (
            summary.get("timestamp") or summary.get("as_of") or datetime.utcnow().isoformat()
        )

        emit_metric(
            getattr(self, "_event_store", None),
            "coinbase_perps",
            {
                "event_type": "cfm_balance_summary",
                "timestamp": timestamp,
                "summary": self._decimals_to_str(summary),
            },
            logger=logger,
        )
        return summary

    def list_cfm_sweeps(self) -> list[dict[str, Any]]:
        if not self.endpoints.supports_derivatives():
            return []
        try:
            payload = self.client.cfm_sweeps() or {}
        except Exception as exc:
            logger.error("Failed to fetch CFM sweeps: %s", exc)
            return []
        sweeps_raw = payload.get("sweeps") or payload.get("data") or []
        normalized: list[dict[str, Any]] = []
        for entry in sweeps_raw:
            if isinstance(entry, dict):
                normalized.append(cast(dict[str, Any], self._normalize_numeric(entry)))
        if normalized:
            emit_metric(
                getattr(self, "_event_store", None),
                "coinbase_perps",
                {
                    "event_type": "cfm_sweeps",
                    "timestamp": datetime.utcnow().isoformat(),
                    "count": len(normalized),
                    "sweeps": self._decimals_to_str(normalized),
                },
                logger=logger,
            )
        return normalized

    def get_cfm_sweeps_schedule(self) -> dict[str, Any]:
        if not self.endpoints.supports_derivatives():
            return {}
        try:
            payload = self.client.cfm_sweeps_schedule() or {}
        except Exception as exc:
            logger.error("Failed to fetch CFM sweep schedule: %s", exc)
            return {}
        schedule_raw = payload.get("schedule") or payload
        return cast(dict[str, Any], self._normalize_numeric(schedule_raw))

    def get_cfm_margin_window(self) -> dict[str, Any]:
        if not self.endpoints.supports_derivatives():
            return {}
        try:
            payload = self.client.cfm_intraday_current_margin_window() or {}
        except Exception as exc:
            logger.error("Failed to fetch CFM margin window: %s", exc)
            return {}
        return cast(dict[str, Any], self._normalize_numeric(payload))

    def update_cfm_margin_window(
        self,
        margin_window: str,
        *,
        effective_time: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.endpoints.supports_derivatives():
            raise InvalidRequestError("Derivatives mode disabled; cannot update CFM margin window.")
        request_payload: dict[str, Any] = {"margin_window": margin_window}
        if effective_time:
            request_payload["effective_time"] = effective_time
        if extra_payload:
            request_payload.update(extra_payload)
        try:
            response = self.client.cfm_intraday_margin_setting(request_payload) or {}
        except Exception as exc:
            logger.error("Failed to update CFM margin window: %s", exc)
            raise
        normalized = cast(dict[str, Any], self._normalize_numeric(response))
        emit_metric(
            getattr(self, "_event_store", None),
            "coinbase_perps",
            {
                "event_type": "cfm_margin_setting",
                "timestamp": datetime.utcnow().isoformat(),
                "margin_window": margin_window,
                "response": self._decimals_to_str(normalized),
            },
            logger=logger,
        )
        return normalized

    # ------------------------------------------------------------------
    # INTX helpers
    # ------------------------------------------------------------------
    def _intx_supported(self) -> bool:
        mode = getattr(self.endpoints, "mode", None)
        if mode == "advanced":
            return True
        support_cb = getattr(self.endpoints, "supports_intx", None)
        if callable(support_cb):
            try:
                return bool(support_cb())
            except Exception:  # pragma: no cover - defensive
                return False
        return False

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
            typed_side: Literal["long", "short"]
            if side in {"long", "short"}:
                typed_side = cast(Literal["long", "short"], side)
            else:
                typed_side = cast(Literal["long", "short"], "long" if quantity >= 0 else "short")
            return Position(
                symbol=data.get("product_id", ""),
                quantity=abs(quantity),
                entry_price=entry_price,
                mark_price=mark_price,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
                leverage=int(leverage) if leverage is not None else None,
                side=typed_side,
            )
        except Exception as exc:
            logger.error("Failed to map position payload: %s", exc)
            return None


__all__ = ["PortfolioRestMixin"]
