"""
Manages Coinbase account state, positions, balances, and CFM/INTX specific features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger

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

        # Key Permissions
        try:
            permissions = self.broker.get_key_permissions()
            snapshot_data["key_permissions"] = permissions
        except Exception as e:
            logger.warning(f"Failed to get key permissions: {e}")
            snapshot_data["key_permissions"] = {"error": str(e)}

        # Fee Schedule
        try:
            fee_schedule = self.broker.get_fee_schedule()
            snapshot_data["fee_schedule"] = fee_schedule
        except Exception as e:
            logger.warning(f"Failed to get fee schedule: {e}")
            snapshot_data["fee_schedule"] = {"error": str(e)}

        # Account Limits
        try:
            limits = self.broker.get_account_limits()
            snapshot_data["limits"] = limits
        except Exception as e:
            logger.warning(f"Failed to get account limits: {e}")
            snapshot_data["limits"] = {"error": str(e)}

        # Transaction Summary
        try:
            transaction_summary = self.broker.get_transaction_summary()
            snapshot_data["transaction_summary"] = transaction_summary
        except Exception as e:
            logger.warning(f"Failed to get transaction summary: {e}")
            snapshot_data["transaction_summary"] = {"error": str(e)}

        # Payment Methods
        try:
            payment_methods = self.broker.list_payment_methods()
            snapshot_data["payment_methods"] = payment_methods
        except Exception as e:
            logger.warning(f"Failed to list payment methods: {e}")
            snapshot_data["payment_methods"] = {"error": str(e)}

        # Portfolios
        try:
            portfolios = self.broker.list_portfolios()
            snapshot_data["portfolios"] = portfolios
        except Exception as e:
            logger.warning(f"Failed to list portfolios: {e}")
            snapshot_data["portfolios"] = {"error": str(e)}

        # CFM Summary
        try:
            cfm_summary = self.broker.get_cfm_balance_summary()
            snapshot_data["cfm_balance_summary"] = cfm_summary
        except Exception as e:
            logger.warning(f"Failed to get CFM balance summary: {e}")
            snapshot_data["cfm_balance_summary"] = {"error": str(e)}

        # CFM Sweeps
        try:
            cfm_sweeps = self.broker.list_cfm_sweeps()
            snapshot_data["cfm_sweeps"] = cfm_sweeps
        except Exception as e:
            logger.warning(f"Failed to list CFM sweeps: {e}")
            snapshot_data["cfm_sweeps"] = {"error": str(e)}

        # CFM Sweeps Schedule
        try:
            cfm_sweeps_schedule = self.broker.get_cfm_sweeps_schedule()
            snapshot_data["cfm_sweeps_schedule"] = cfm_sweeps_schedule
        except Exception as e:
            logger.warning(f"Failed to get CFM sweeps schedule: {e}")
            snapshot_data["cfm_sweeps_schedule"] = {"error": str(e)}

        # CFM Margin Window
        try:
            cfm_margin_window = self.broker.get_cfm_margin_window()
            snapshot_data["cfm_margin_window"] = cfm_margin_window
        except Exception as e:
            logger.warning(f"Failed to get CFM margin window: {e}")
            snapshot_data["cfm_margin_window"] = {"error": str(e)}

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

        self._event_store.append_metric(
            bot_id="account_manager",
            metrics={"event_type": "account_manager_snapshot", "data": snapshot_data},
        )

        return snapshot_data

    def convert(self, payload: dict[str, Any], commit: bool = False) -> dict[str, Any]:
        quote = self.broker.create_convert_quote(payload)
        if commit:
            result = self.broker.commit_convert_trade(quote["trade_id"], payload)
            self._event_store.append_metric(
                bot_id="account_manager", metrics={"event_type": "convert_commit", "data": result}
            )
            return result
        return quote

    def move_funds(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.broker.move_portfolio_funds(payload)
        self._event_store.append_metric(
            bot_id="account_manager", metrics={"event_type": "portfolio_move", "data": result}
        )
        return result

    def supports_intx(self) -> bool:
        """Check if INTX is supported by the broker."""
        return self.broker.supports_intx()

    def get_intx_portfolio_uuid(self, *, refresh: bool = False) -> str | None:
        """Get the INTX portfolio UUID, with optional refresh."""
        return self.broker.resolve_intx_portfolio(refresh=refresh)

    def invalidate_intx_cache(self) -> None:
        """Invalidate the cached INTX portfolio UUID."""
        if hasattr(self.broker, "invalidate_intx_cache"):
            self.broker.invalidate_intx_cache()
