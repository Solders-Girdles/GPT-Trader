"""Utility helpers for Coinbase account management actions.

Provides convenience wrappers around adapter endpoints so higher-level
workflows (telemetry, treasury automation) can remain thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bot_v2.monitoring.system import LogLevel, get_logger
from bot_v2.orchestration.runtime_settings import RuntimeSettings, get_runtime_settings
from bot_v2.utilities import emit_metric as emit_metric_util

logger = get_logger()

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage


@dataclass
class CoinbaseAccountManager:
    """High-level helper for Coinbase account/treasury actions."""

    broker: CoinbaseBrokerage
    event_store: Any | None = None  # accept EventStore-like interface
    runtime_settings: RuntimeSettings | None = None
    intx_portfolio_uuid: str | None = None
    _intx_uuid_cache: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.runtime_settings is None:
            try:
                self.runtime_settings = get_runtime_settings()
            except Exception:  # pragma: no cover - defensive
                self.runtime_settings = None
        if self.intx_portfolio_uuid is None and self.runtime_settings is not None:
            self.intx_portfolio_uuid = getattr(
                self.runtime_settings, "coinbase_intx_portfolio_uuid", None
            )

    def _supports_intx(self) -> bool:
        broker_support = getattr(self.broker, "supports_intx", None)
        if callable(broker_support):
            try:
                return bool(broker_support())
            except Exception:  # pragma: no cover - defensive
                return False
        return False

    def supports_intx(self) -> bool:
        return self._supports_intx()

    def _resolve_intx_portfolio_uuid(self, *, refresh: bool = False) -> str | None:
        if not self._supports_intx():
            return None
        if not refresh and self._intx_uuid_cache:
            return self._intx_uuid_cache

        preferred = self.intx_portfolio_uuid
        resolver = getattr(self.broker, "resolve_intx_portfolio", None)
        resolved: str | None = None
        if callable(resolver):
            try:
                resolved = resolver(preferred_uuid=preferred, refresh=refresh)
            except Exception as exc:
                logger.log_event(
                    LogLevel.DEBUG,
                    "account_snapshot",
                    "intx resolve failed",
                    error=str(exc),
                )
                resolved = None
            if resolved is None and not preferred and not refresh:
                # Fallback discovery on refresh attempt
                try:
                    resolved = resolver(preferred_uuid=None, refresh=True)
                except Exception as exc:
                    logger.log_event(
                        LogLevel.DEBUG,
                        "account_snapshot",
                        "intx refresh discovery failed",
                        error=str(exc),
                    )
                    resolved = None
        else:
            resolved = preferred

        if resolved:
            self._intx_uuid_cache = resolved
        elif refresh:
            self._intx_uuid_cache = None
        return resolved

    def get_intx_portfolio_uuid(self, *, refresh: bool = False) -> str | None:
        return self._resolve_intx_portfolio_uuid(refresh=refresh)

    def invalidate_intx_cache(self) -> None:
        self._intx_uuid_cache = None

    def _populate_intx_snapshot(
        self,
        target_uuid: str,
        out: dict[str, Any],
    ) -> tuple[bool, str | None]:
        try:
            balances = self.broker.get_intx_balances(portfolio_uuid=target_uuid)
            portfolio = self.broker.get_intx_portfolio(portfolio_uuid=target_uuid)
            positions = self.broker.list_intx_positions(portfolio_uuid=target_uuid)
            collateral = self.broker.get_intx_collateral()
        except Exception as exc:
            logger.log_event(
                LogLevel.DEBUG,
                "account_snapshot",
                "intx telemetry unavailable",
                error=str(exc),
            )
            self._intx_uuid_cache = None
            return False, str(exc)

        out["intx_balances"] = balances
        out["intx_portfolio"] = portfolio
        out["intx_positions"] = positions
        out["intx_collateral"] = collateral
        return True, None

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
        try:
            out["cfm_balance_summary"] = self.broker.get_cfm_balance_summary()
        except Exception as exc:
            logger.log_event(
                LogLevel.DEBUG,
                "account_snapshot",
                "cfm balance summary unavailable",
                error=str(exc),
            )
            out["cfm_balance_summary"] = {}
        try:
            out["cfm_sweeps"] = self.broker.list_cfm_sweeps()
        except Exception as exc:
            logger.log_event(
                LogLevel.DEBUG, "account_snapshot", "cfm sweeps unavailable", error=str(exc)
            )
            out["cfm_sweeps"] = []
        try:
            out["cfm_sweeps_schedule"] = self.broker.get_cfm_sweeps_schedule()
        except Exception as exc:
            logger.log_event(
                LogLevel.DEBUG,
                "account_snapshot",
                "cfm sweeps schedule unavailable",
                error=str(exc),
            )
            out["cfm_sweeps_schedule"] = {}
        try:
            out["cfm_margin_window"] = self.broker.get_cfm_margin_window()
        except Exception as exc:
            logger.log_event(
                LogLevel.DEBUG,
                "account_snapshot",
                "cfm margin window unavailable",
                error=str(exc),
            )
            out["cfm_margin_window"] = {}

        intx_available = False
        intx_reason: str | None = None
        if self._supports_intx():
            intx_uuid = self.get_intx_portfolio_uuid()
            if intx_uuid:
                success, reason = self._populate_intx_snapshot(intx_uuid, out)
                if not success:
                    refreshed_uuid = self.get_intx_portfolio_uuid(refresh=True)
                    if refreshed_uuid and refreshed_uuid != intx_uuid:
                        success, reason = self._populate_intx_snapshot(refreshed_uuid, out)
                        if success:
                            intx_uuid = refreshed_uuid
                intx_available = success
                if success:
                    out["intx_portfolio_uuid"] = intx_uuid
                else:
                    intx_reason = reason or "intx_snapshot_failed"
            else:
                intx_reason = "intx_portfolio_not_found"
        else:
            intx_reason = "intx_not_supported"

        out["intx_available"] = intx_available
        if not intx_available:
            out.setdefault("intx_balances", [])
            out.setdefault("intx_portfolio", {})
            out.setdefault("intx_positions", [])
            out.setdefault("intx_collateral", {})
            if intx_reason:
                out["intx_unavailable_reason"] = intx_reason

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
