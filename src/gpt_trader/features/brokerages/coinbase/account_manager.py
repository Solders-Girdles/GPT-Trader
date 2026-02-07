"""
Manages Coinbase account state, positions, balances, and CFM/INTX specific features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.telemetry import emit_metric

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.test_helpers import CoinbaseBrokerage

    from gpt_trader.persistence.event_store import EventStore

logger = get_logger(__name__, component="coinbase_account")


class CoinbaseAccountManager:
    def __init__(self, broker: CoinbaseBrokerage, event_store: EventStore):
        self.broker = broker
        self._event_store = event_store

    def snapshot(self) -> dict[str, Any]:
        snapshot_data: dict[str, Any] = {}

        snapshot_probes: list[tuple[str, Callable[[], Any]]] = [
            ("key_permissions", self.broker.get_key_permissions),
            ("fee_schedule", self.broker.get_fee_schedule),
            ("limits", self.broker.get_account_limits),
            ("transaction_summary", self.broker.get_transaction_summary),
            ("payment_methods", self.broker.list_payment_methods),
            ("portfolios", self.broker.list_portfolios),
            ("cfm_balance_summary", self.broker.get_cfm_balance_summary),
            ("cfm_sweeps", self.broker.list_cfm_sweeps),
            ("cfm_sweeps_schedule", self.broker.get_cfm_sweeps_schedule),
            ("cfm_margin_window", self.broker.get_cfm_margin_window),
        ]

        for key, probe in snapshot_probes:
            snapshot_data[key] = self._execute_snapshot_probe(key, probe)

        # INTX Status
        snapshot_data["intx_available"] = self.broker.supports_intx()
        if not snapshot_data["intx_available"]:
            snapshot_data["intx_unavailable_reason"] = "intx_not_supported"
            snapshot_data["intx_balances"] = []
            snapshot_data["intx_positions"] = []
            snapshot_data["intx_collateral"] = {}
        else:
            try:
                # Resolve INTX portfolio
                intx_portfolio_uuid = self.broker.resolve_intx_portfolio()

                # Try to get balances, if fails with specific error, refresh portfolio
                try:
                    if intx_portfolio_uuid:
                        snapshot_data["intx_balances"] = self.broker.get_intx_balances(
                            intx_portfolio_uuid
                        )
                except Exception:
                    # Retry with refresh
                    intx_portfolio_uuid = self.broker.resolve_intx_portfolio(refresh=True)
                    if intx_portfolio_uuid:
                        snapshot_data["intx_balances"] = self.broker.get_intx_balances(
                            intx_portfolio_uuid
                        )

                if not intx_portfolio_uuid:
                    snapshot_data["intx_available"] = False
                    snapshot_data["intx_unavailable_reason"] = "intx_portfolio_not_found"
                    snapshot_data["intx_balances"] = []
                    snapshot_data["intx_positions"] = []
                    snapshot_data["intx_collateral"] = {}
                else:
                    snapshot_data["intx_portfolio_uuid"] = intx_portfolio_uuid
                    # If balances not set yet (no error or recovered)
                    if "intx_balances" not in snapshot_data:
                        snapshot_data["intx_balances"] = self.broker.get_intx_balances(
                            intx_portfolio_uuid
                        )

                    snapshot_data["intx_positions"] = self.broker.list_intx_positions(
                        intx_portfolio_uuid
                    )
                    snapshot_data["intx_collateral"] = self.broker.get_intx_multi_asset_collateral()
            except Exception as e:
                logger.warning(f"Failed to get INTX data: {e}")
                # Don't mark as unavailable just because data fetch failed (could be temporary)
                # snapshot_data["intx_available"] = False
                snapshot_data["intx_unavailable_reason"] = str(e)
                snapshot_data["intx_balances"] = []
                snapshot_data["intx_positions"] = []
                snapshot_data["intx_collateral"] = {}

        emit_metric(
            self._event_store,
            "account_manager",
            {"event_type": "account_manager_snapshot", "data": snapshot_data},
            logger=logger,
        )

        return snapshot_data

    def _execute_snapshot_probe(self, key: str, probe: Callable[[], Any]) -> Any:
        try:
            return probe()
        except Exception as error:
            logger.warning("Failed to collect %s: %s", key, error)
            return self._snapshot_error_payload(error)

    @staticmethod
    def _snapshot_error_payload(error: Exception) -> dict[str, Any]:
        return {
            "error": {
                "message": str(error),
                "type": type(error).__name__,
            }
        }

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
