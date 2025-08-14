from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from ..exec.base import Account, Broker, Position
from ..logging import get_logger
from ..portfolio.allocator import PortfolioRules

logger = get_logger("live_portfolio")


@dataclass
class PortfolioState:
    """Current state of the live portfolio."""

    account: Account
    positions: dict[str, Position] = field(default_factory=dict)
    total_market_value: float = 0.0
    total_unrealized_pl: float = 0.0
    total_unrealized_plpc: float = 0.0
    available_cash: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def update_from_account(self, account: Account) -> None:
        """Update portfolio state from account information."""
        self.account = account
        self.available_cash = account.cash
        self.total_market_value = account.portfolio_value
        self.timestamp = datetime.now()

    def update_positions(self, positions: list[Position]) -> None:
        """Update portfolio state from position list."""
        self.positions = {pos.symbol: pos for pos in positions}
        self.total_unrealized_pl = sum(pos.unrealized_pl for pos in positions)
        self.total_unrealized_plpc = sum(pos.unrealized_plpc for pos in positions)
        self.timestamp = datetime.now()


class LivePortfolioManager:
    """Manages live portfolio state and risk monitoring."""

    def __init__(self, broker: Broker, rules: PortfolioRules) -> None:
        """Initialize the portfolio manager."""
        self.broker = broker
        self.rules = rules
        self.state = PortfolioState(
            Account(
                id="",
                account_number="",
                status="",
                crypto_status="",
                currency="",
                buying_power=0.0,
                regt_buying_power=0.0,
                daytrading_buying_power=0.0,
                non_marginable_buying_power=0.0,
                cash=0.0,
                accrued_fees=0.0,
                pending_transfer_out=0.0,
                pending_transfer_in=0.0,
                portfolio_value=0.0,
                pattern_day_trader=False,
                trading_blocked=False,
                transfers_blocked=False,
                account_blocked=False,
                created_at=datetime.now(),
                trade_suspended_by_user=False,
                multiplier="",
                shorting_enabled=False,
                equity=0.0,
                last_equity=0.0,
                long_market_value=0.0,
                short_market_value=0.0,
                initial_margin=0.0,
                maintenance_margin=0.0,
                last_maintenance_margin=0.0,
                sma=0.0,
                daytrade_count=0,
            )
        )
        self.position_history: list[PortfolioState] = []
        self.max_history_size = 1000

        logger.info("Live portfolio manager initialized")

    async def refresh_state(self) -> None:
        """Refresh portfolio state from broker."""
        try:
            # Get account information
            account = self.broker.get_account()
            self.state.update_from_account(account)

            # Get current positions
            positions = self.broker.get_positions()
            self.state.update_positions(positions)

            # Store in history
            self.position_history.append(self.state)
            if len(self.position_history) > self.max_history_size:
                self.position_history.pop(0)

            logger.debug(
                f"Portfolio refreshed: {len(positions)} positions, ${self.state.total_market_value:.2f} value"
            )

        except Exception as e:
            logger.error(f"Failed to refresh portfolio state: {e}")
            raise

    def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol."""
        return self.state.positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all current positions."""
        return self.state.positions.copy()

    def get_position_count(self) -> int:
        """Get number of current positions."""
        return len(self.state.positions)

    def get_total_market_value(self) -> float:
        """Get total market value of portfolio."""
        return self.state.total_market_value

    def get_available_cash(self) -> float:
        """Get available cash."""
        return self.state.available_cash

    def get_unrealized_pl(self) -> float:
        """Get total unrealized P&L."""
        return self.state.total_unrealized_pl

    def get_unrealized_plpc(self) -> float:
        """Get total unrealized P&L percentage."""
        return self.state.total_unrealized_plpc

    def check_risk_limits(self) -> dict[str, bool]:
        """Check various risk limits and return status."""
        limits = {}

        # Check position count limit
        limits["position_count_ok"] = self.get_position_count() <= self.rules.max_positions

        # Check gross exposure limit
        if self.state.total_market_value > 0:
            gross_exposure_pct = self.state.total_market_value / self.state.account.portfolio_value
            limits["gross_exposure_ok"] = gross_exposure_pct <= self.rules.max_gross_exposure_pct
        else:
            limits["gross_exposure_ok"] = True

        # Check buying power
        limits["buying_power_ok"] = self.state.account.buying_power > 0

        # Check account status
        limits["account_active"] = (
            not self.state.account.account_blocked and not self.state.account.trading_blocked
        )

        return limits

    def get_risk_summary(self) -> dict[str, any]:
        """Get comprehensive risk summary."""
        risk_limits = self.check_risk_limits()

        return {
            "timestamp": self.state.timestamp,
            "portfolio_value": self.state.total_market_value,
            "available_cash": self.state.available_cash,
            "unrealized_pl": self.state.total_unrealized_pl,
            "unrealized_plpc": self.state.total_unrealized_plpc,
            "position_count": self.get_position_count(),
            "risk_limits": risk_limits,
            "all_limits_ok": all(risk_limits.values()),
        }

    def get_position_summary(self) -> pd.DataFrame:
        """Get position summary as DataFrame."""
        if not self.state.positions:
            return pd.DataFrame()

        data = []
        for symbol, position in self.state.positions.items():
            data.append(
                {
                    "symbol": symbol,
                    "qty": position.qty,
                    "avg_price": position.avg_price,
                    "current_price": position.current_price,
                    "market_value": position.market_value,
                    "unrealized_pl": position.unrealized_pl,
                    "unrealized_plpc": position.unrealized_plpc,
                    "timestamp": position.timestamp,
                }
            )

        return pd.DataFrame(data)

    def get_portfolio_history(self, hours: int = 24) -> pd.DataFrame:
        """Get portfolio history for the last N hours."""
        if not self.position_history:
            return pd.DataFrame()

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_states = [state for state in self.position_history if state.timestamp >= cutoff_time]

        if not recent_states:
            return pd.DataFrame()

        data = []
        for state in recent_states:
            data.append(
                {
                    "timestamp": state.timestamp,
                    "portfolio_value": state.total_market_value,
                    "available_cash": state.available_cash,
                    "unrealized_pl": state.total_unrealized_pl,
                    "unrealized_plpc": state.total_unrealized_plpc,
                    "position_count": len(state.positions),
                }
            )

        return pd.DataFrame(data)
