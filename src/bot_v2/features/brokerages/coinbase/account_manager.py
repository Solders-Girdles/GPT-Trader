"""Utility helpers for Coinbase account management actions.

Provides convenience wrappers around adapter endpoints so higher-level
workflows (telemetry, treasury automation) can remain thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.monitoring.system import LogLevel, get_logger
from bot_v2.utilities import emit_metric as emit_metric_util

logger = get_logger()


@dataclass
class CoinbaseAccountManager:
    """High-level helper for Coinbase account/treasury actions."""

    broker: CoinbaseBrokerage
    event_store: Any | None = None  # accept EventStore-like interface

    def snapshot(self, emit_metric: bool = True) -> dict[str, Any]:
        """Collect account-centric information in a single payload."""
        out: dict[str, Any] = {}
        try:
            out["key_permissions"] = self.broker.get_key_permissions()
        except Exception as exc:
            logger.log_event(
                LogLevel.WARNING, "account_snapshot", "key permissions unavailable", error=str(exc)
            )
            out["key_permissions"] = {}
        try:
            out["fee_schedule"] = self.broker.get_fee_schedule()
        except Exception as exc:
            logger.log_event(
                LogLevel.WARNING,
                "account_snapshot",
                "fee schedule unavailable",
                error=str(exc),
            )
            out["fee_schedule"] = {}
            out["fee_schedule_unavailable"] = True
        try:
            out["limits"] = self.broker.get_account_limits()
        except Exception as exc:
            logger.log_event(
                LogLevel.WARNING,
                "account_snapshot",
                "account limits unavailable",
                error=str(exc),
            )
            out["limits"] = {}
            out["limits_unavailable"] = True
        try:
            out["transaction_summary"] = self.broker.get_transaction_summary()
        except Exception:
            out["transaction_summary"] = {}
        try:
            out["payment_methods"] = self.broker.list_payment_methods()
        except Exception:
            out["payment_methods"] = []
        try:
            out["portfolios"] = self.broker.list_portfolios()
        except Exception:
            out["portfolios"] = []

        if emit_metric:
            emit_metric_util(
                self.event_store,
                bot_id="coinbase_account",
                payload={"event_type": "account_manager_snapshot", **out},
                logger=logger,
            )
        return out

    def convert(
        self,
        payload: dict[str, Any],
        commit: bool = True,
        commit_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create (and optionally commit) a convert quote."""
        quote = self.broker.create_convert_quote(payload)
        if not commit:
            return quote

        trade_id = quote.get("trade_id") or quote.get("id") or quote.get("quote_id")
        if not trade_id:
            raise ValueError("convert quote did not include trade identifier")

        commit_result = self.broker.commit_convert_trade(trade_id, commit_payload or {})
        if self.event_store is not None:
            emit_metric_util(
                self.event_store,
                "coinbase_account",
                {
                    "event_type": "convert_commit",
                    "trade_id": trade_id,
                    "quote": quote,
                    "result": commit_result,
                },
                logger=logger,
            )
        return commit_result

    def move_funds(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Move funds between Coinbase portfolios."""
        result = self.broker.move_portfolio_funds(payload)
        if self.event_store is not None:
            emit_metric_util(
                self.event_store,
                "coinbase_account",
                {"event_type": "portfolio_move", **payload},
                logger=logger,
            )
        return result
