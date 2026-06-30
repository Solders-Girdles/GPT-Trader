"""
Manages Coinbase account state, positions, balances, and CFM-specific features.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.telemetry import emit_metric

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.test_helpers import CoinbaseBrokerage

    from gpt_trader.persistence.event_store import EventStore

logger = get_logger(__name__, component="coinbase_account")


class CoinbaseAccountManager:
    _FRESHNESS_FRESH = "fresh"
    _FRESHNESS_ERROR = "error"
    _FRESHNESS_UNAVAILABLE = "unavailable"

    def __init__(self, broker: CoinbaseBrokerage, event_store: EventStore):
        self.broker = broker
        self._event_store = event_store

    def snapshot(self) -> dict[str, Any]:
        snapshot_data: dict[str, Any] = {}
        freshness_data: dict[str, dict[str, Any]] = {}

        snapshot_probes: list[tuple[str, str]] = [
            ("key_permissions", "get_key_permissions"),
            ("fee_schedule", "get_fee_schedule"),
            ("limits", "get_account_limits"),
            ("transaction_summary", "get_transaction_summary"),
            ("payment_methods", "list_payment_methods"),
            ("portfolios", "list_portfolios"),
            ("cfm_balance_summary", "get_cfm_balance_summary"),
            ("cfm_sweeps", "list_cfm_sweeps"),
            ("cfm_sweeps_schedule", "get_cfm_sweeps_schedule"),
            ("cfm_margin_window", "get_cfm_margin_window"),
        ]

        for key, probe_name in snapshot_probes:
            result, metadata = self._execute_snapshot_probe(key, probe_name)
            snapshot_data[key] = result
            freshness_data[key] = metadata

        snapshot_data["freshness"] = freshness_data

        emit_metric(
            self._event_store,
            "account_manager",
            {"event_type": "account_manager_snapshot", "data": snapshot_data},
            logger=logger,
        )

        return snapshot_data

    def _execute_snapshot_probe(self, key: str, probe_name: str) -> tuple[Any, dict[str, Any]]:
        try:
            probe = getattr(self.broker, probe_name)
            if callable(probe):
                result = probe()
                return result, self._freshness_entry(self._FRESHNESS_FRESH)
        except Exception as error:  # noqa: BLE001 - optional probes should not abort snapshot
            return self._snapshot_probe_failure(key, error)

        return self._snapshot_probe_failure(key, TypeError(f"{probe_name} is not callable"))

    def _snapshot_probe_failure(
        self, key: str, error: Exception
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        logger.warning("Failed to collect %s: %s", key, error, exc_info=error)
        return (
            self._snapshot_error_payload(error),
            self._freshness_entry(
                self._FRESHNESS_ERROR,
                error_code=self._error_code_from_exception(error),
            ),
        )

    @staticmethod
    def _snapshot_error_payload(error: Exception) -> dict[str, Any]:
        return {
            "error": {
                "message": str(error),
                "type": type(error).__name__,
            }
        }

    def _freshness_entry(self, status: str, *, error_code: str | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "status": status,
            "fetched_at": self._current_timestamp(),
        }
        if error_code:
            entry["error_code"] = error_code
        return entry

    def _current_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _error_code_from_exception(error: Exception) -> str:
        return getattr(error, "error_code", type(error).__name__)

    def convert(self, payload: dict[str, Any], commit: bool = False) -> dict[str, Any]:
        quote = self.broker.create_convert_quote(payload)
        if commit:
            result = self.broker.commit_convert_trade(quote["trade_id"], payload)
            emit_metric(
                self._event_store,
                "account_manager",
                {"event_type": "convert_commit", "data": result},
                logger=logger,
            )
            return cast(dict[str, Any], result)
        return cast(dict[str, Any], quote)

    def move_funds(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.broker.move_portfolio_funds(payload)
        emit_metric(
            self._event_store,
            "account_manager",
            {"event_type": "portfolio_move", "data": result},
            logger=logger,
        )
        return cast(dict[str, Any], result)
