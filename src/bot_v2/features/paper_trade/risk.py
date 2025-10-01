"""
Local risk management for paper trading.

Complete isolation - no external dependencies.
"""

from bot_v2.features.paper_trade.types import AccountStatus
from bot_v2.types.trading import AccountSnapshot


class RiskManager:
    """Risk management for paper trading."""

    def __init__(
        self,
        max_position_size: float = 0.2,  # Max 20% per position
        max_daily_loss: float = 0.05,  # Max 5% daily loss
        max_drawdown: float = 0.15,  # Max 15% drawdown
        min_cash_reserve: float = 0.1,  # Keep 10% cash
    ) -> None:
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum position size as fraction of equity
            max_daily_loss: Maximum daily loss tolerance
            max_drawdown: Maximum drawdown tolerance
            min_cash_reserve: Minimum cash to maintain
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.min_cash_reserve = min_cash_reserve

        self.daily_pnl = 0
        self.peak_equity = 0
        self.initial_equity = 0
        self.last_equity = 0

    def initialize(self, initial_equity: float) -> None:
        """Initialize risk tracking."""
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.last_equity = initial_equity

    def check_trade(
        self, symbol: str, signal: int, price: float, account: AccountStatus | AccountSnapshot
    ) -> bool:
        """
        Check if trade meets risk criteria.

        Args:
            symbol: Stock symbol
            signal: Trade signal (1=buy, -1=sell)
            price: Current price
            account: Current account status

        Returns:
            True if trade is allowed
        """
        if isinstance(account, AccountSnapshot):
            account = AccountStatus.from_account_snapshot(account)

        # Update tracking
        if self.initial_equity == 0:
            self.initialize(account.total_equity)

        current_equity = account.total_equity

        # Update peak for drawdown calculation
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Check drawdown
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            return False

        # Check daily loss
        daily_loss = (self.last_equity - current_equity) / self.last_equity
        if daily_loss > self.max_daily_loss:
            return False

        # For buy signals, check position sizing
        if signal == 1:
            # Check cash reserve
            cash_after_trade = account.cash - (price * 100)  # Assume 100 shares
            min_cash_needed = account.total_equity * self.min_cash_reserve

            if cash_after_trade <= min_cash_needed:
                return False

            # Check position size limit
            position_value = price * 100
            max_position_value = account.total_equity * self.max_position_size

            if position_value > max_position_value:
                return False

        return True

    def update_daily_stats(self, current_equity: float) -> None:
        """
        Update daily statistics.

        Args:
            current_equity: Current total equity
        """
        self.daily_pnl = current_equity - self.last_equity
        self.last_equity = current_equity

    def reset_daily_stats(self, current_equity: float) -> None:
        """
        Reset daily statistics (call at start of new day).

        Args:
            current_equity: Current total equity
        """
        self.daily_pnl = 0
        self.last_equity = current_equity

    def get_risk_metrics(self, account: AccountStatus | AccountSnapshot) -> dict:
        """
        Get current risk metrics.

        Args:
            account: Current account status

        Returns:
            Dict of risk metrics
        """
        if isinstance(account, AccountSnapshot):
            account = AccountStatus.from_account_snapshot(account)

        current_equity = account.total_equity

        # Update peak for drawdown calculation
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate metrics
        current_drawdown = (
            (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        )
        daily_return = (
            (current_equity - self.last_equity) / self.last_equity if self.last_equity > 0 else 0
        )
        total_return = (
            (current_equity - self.initial_equity) / self.initial_equity
            if self.initial_equity > 0
            else 0
        )
        cash_percentage = account.cash / current_equity if current_equity > 0 else 0

        return {
            "current_drawdown": current_drawdown,
            "max_drawdown_limit": self.max_drawdown,
            "daily_return": daily_return,
            "max_daily_loss_limit": self.max_daily_loss,
            "total_return": total_return,
            "cash_percentage": cash_percentage,
            "total_equity": current_equity,
            "min_cash_reserve": self.min_cash_reserve,
            "risk_utilization": (
                current_drawdown / self.max_drawdown if self.max_drawdown > 0 else 0
            ),
        }
