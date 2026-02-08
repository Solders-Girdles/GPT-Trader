"""
Manages Coinbase account state, positions, balances, and CFM/INTX specific features.
"""

from __future__ import annotations

from collections.abc import Callable
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

        intx_available = self.broker.supports_intx()
        snapshot_data["intx_available"] = intx_available
        freshness_data["intx_available"] = self._freshness_entry(
            self._FRESHNESS_FRESH if intx_available else self._FRESHNESS_UNAVAILABLE,
            error_code=None if intx_available else "INTX_NOT_SUPPORTED",
        )

        if not intx_available:
            self._record_intx_fallback(
                snapshot_data,
                freshness_data,
                reason="intx_not_supported",
                status=self._FRESHNESS_UNAVAILABLE,
                error_code="INTX_NOT_SUPPORTED",
            )
        else:
            try:
                intx_portfolio_uuid = self.broker.resolve_intx_portfolio()
                balances, balances_meta, resolved_uuid = self._fetch_intx_balances_with_retry(
                    intx_portfolio_uuid
                )
                snapshot_data["intx_balances"] = balances
                freshness_data["intx_balances"] = balances_meta

                target_uuid = resolved_uuid
                if not target_uuid:
                    snapshot_data["intx_available"] = False
                    self._record_intx_fallback(
                        snapshot_data,
                        freshness_data,
                        reason="intx_portfolio_not_found",
                        status=self._FRESHNESS_UNAVAILABLE,
                        error_code="INTX_PORTFOLIO_NOT_FOUND",
                    )
                else:
                    snapshot_data["intx_portfolio_uuid"] = target_uuid
                    positions, positions_meta = self._fetch_intx_section(
                        "intx_positions",
                        lambda: self.broker.list_intx_positions(target_uuid),
                        [],
                    )
                    snapshot_data["intx_positions"] = positions
                    freshness_data["intx_positions"] = positions_meta

                    collateral, collateral_meta = self._fetch_intx_section(
                        "intx_collateral",
                        self.broker.get_intx_multi_asset_collateral,
                        {},
                    )
                    snapshot_data["intx_collateral"] = collateral
                    freshness_data["intx_collateral"] = collateral_meta
            except Exception as e:
                logger.warning(f"Failed to get INTX data: {e}")
                self._record_intx_fallback(
                    snapshot_data,
                    freshness_data,
                    reason=str(e),
                    status=self._FRESHNESS_ERROR,
                    error_code=self._error_code_from_exception(e),
                )

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
            if not callable(probe):
                raise TypeError(f"{probe_name} is not callable")
            result = probe()
            return result, self._freshness_entry(self._FRESHNESS_FRESH)
        except Exception as error:
            logger.warning("Failed to collect %s: %s", key, error)
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

    def _fetch_intx_balances_with_retry(
        self, portfolio_uuid: str | None
    ) -> tuple[list[Any], dict[str, Any], str | None]:
        if not portfolio_uuid:
            return (
                [],
                self._freshness_entry(
                    self._FRESHNESS_UNAVAILABLE,
                    error_code="INTX_PORTFOLIO_NOT_FOUND",
                ),
                None,
            )
        try:
            balances = self.broker.get_intx_balances(portfolio_uuid)
            return balances, self._freshness_entry(self._FRESHNESS_FRESH), portfolio_uuid
        except Exception as error:
            logger.warning("Failed to collect intx_balances: %s", error)
            refreshed_uuid = self.broker.resolve_intx_portfolio(refresh=True)
            if refreshed_uuid:
                try:
                    balances = self.broker.get_intx_balances(refreshed_uuid)
                    return balances, self._freshness_entry(self._FRESHNESS_FRESH), refreshed_uuid
                except Exception as retry_error:
                    logger.warning("Retry failed for intx_balances: %s", retry_error)
                    return (
                        [],
                        self._freshness_entry(
                            self._FRESHNESS_ERROR,
                            error_code=self._error_code_from_exception(retry_error),
                        ),
                        refreshed_uuid,
                    )
            return (
                [],
                self._freshness_entry(
                    self._FRESHNESS_ERROR,
                    error_code=self._error_code_from_exception(error),
                ),
                None,
            )

    def _fetch_intx_section(
        self, key: str, fetcher: Callable[[], Any], default: Any
    ) -> tuple[Any, dict[str, Any]]:
        try:
            value = fetcher()
            return value, self._freshness_entry(self._FRESHNESS_FRESH)
        except Exception as error:
            logger.warning("Failed to collect %s: %s", key, error)
            return (
                default,
                self._freshness_entry(
                    self._FRESHNESS_ERROR,
                    error_code=self._error_code_from_exception(error),
                ),
            )

    def _record_intx_fallback(
        self,
        snapshot_data: dict[str, Any],
        freshness_data: dict[str, dict[str, Any]],
        *,
        reason: str,
        status: str,
        error_code: str,
    ) -> None:
        snapshot_data["intx_unavailable_reason"] = reason
        snapshot_data["intx_balances"] = []
        snapshot_data["intx_positions"] = []
        snapshot_data["intx_collateral"] = {}
        freshness_data["intx_available"] = self._freshness_entry(status, error_code=error_code)
        for section in ("intx_balances", "intx_positions", "intx_collateral"):
            freshness_data[section] = self._freshness_entry(status, error_code=error_code)

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

    def supports_intx(self) -> bool:
        """Check if INTX is supported by the broker."""
        return bool(self.broker.supports_intx())

    def get_intx_portfolio_uuid(self, *, refresh: bool = False) -> str | None:
        """Get the INTX portfolio UUID, with optional refresh."""
        return cast(str | None, self.broker.resolve_intx_portfolio(refresh=refresh))

    def invalidate_intx_cache(self) -> None:
        """Invalidate the cached INTX portfolio UUID."""
        if hasattr(self.broker, "invalidate_intx_cache"):
            self.broker.invalidate_intx_cache()
