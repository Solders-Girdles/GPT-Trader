"""
Local risk management for live trading.

Complete isolation - no external dependencies.
"""

from typing import Optional, Dict
from .types import AccountInfo, Position


class LiveRiskManager:
    """Risk management for live trading."""
    
    def __init__(
        self,
        max_position_size: float = 0.1,    # Max 10% per position
        max_daily_loss: float = 0.02,       # Max 2% daily loss
        max_total_risk: float = 0.06,       # Max 6% portfolio risk
        min_cash_reserve: float = 0.2,      # Keep 20% cash
        max_positions: int = 10             # Max concurrent positions
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum size per position as fraction of equity
            max_daily_loss: Maximum daily loss tolerance
            max_total_risk: Maximum total portfolio risk
            min_cash_reserve: Minimum cash to maintain
            max_positions: Maximum number of positions
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_total_risk = max_total_risk
        self.min_cash_reserve = min_cash_reserve
        self.max_positions = max_positions
        
        self.daily_pnl = 0.0
        self.start_of_day_equity = 0.0
        self.positions_count = 0
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        account: Optional[AccountInfo],
        price: Optional[float] = None
    ) -> bool:
        """
        Validate an order against risk rules.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            account: Current account info
            price: Expected price (optional)
            
        Returns:
            True if order passes risk checks
        """
        if not account:
            return False
        
        # Check daily loss limit
        if self.start_of_day_equity > 0:
            daily_loss = (self.start_of_day_equity - account.equity) / self.start_of_day_equity
            if daily_loss > self.max_daily_loss:
                print(f"Risk Check Failed: Daily loss limit exceeded ({daily_loss:.2%})")
                return False
        
        # For buy orders, check additional constraints
        if side.lower() == 'buy':
            # Estimate position value
            est_price = price or 100.0  # Use provided price or default
            position_value = quantity * est_price
            
            # Check position size limit
            max_position_value = account.equity * self.max_position_size
            if position_value > max_position_value:
                print(f"Risk Check Failed: Position too large (${position_value:.2f} > ${max_position_value:.2f})")
                return False
            
            # Check cash reserve
            cash_after = account.cash - position_value
            min_cash_needed = account.equity * self.min_cash_reserve
            if cash_after < min_cash_needed:
                print(f"Risk Check Failed: Insufficient cash reserve (${cash_after:.2f} < ${min_cash_needed:.2f})")
                return False
            
            # Check max positions
            if self.positions_count >= self.max_positions:
                print(f"Risk Check Failed: Maximum positions reached ({self.max_positions})")
                return False
        
        return True
    
    def update_positions(self, positions: Dict[str, Position]):
        """
        Update position tracking.
        
        Args:
            positions: Current positions
        """
        self.positions_count = len(positions)
    
    def reset_daily_tracking(self, current_equity: float):
        """
        Reset daily tracking (call at start of trading day).
        
        Args:
            current_equity: Current account equity
        """
        self.start_of_day_equity = current_equity
        self.daily_pnl = 0.0
    
    def calculate_position_size(
        self,
        account: AccountInfo,
        symbol: str,
        risk_per_trade: float = 0.01
    ) -> int:
        """
        Calculate appropriate position size.
        
        Args:
            account: Account information
            symbol: Stock symbol
            risk_per_trade: Risk per trade as fraction of equity
            
        Returns:
            Recommended position size in shares
        """
        # Risk-based position sizing
        risk_amount = account.equity * risk_per_trade
        
        # Simplified calculation (would use ATR or volatility in production)
        estimated_price = 100.0  # Would fetch real price
        stop_loss_distance = estimated_price * 0.02  # 2% stop loss
        
        shares = int(risk_amount / stop_loss_distance)
        
        # Apply position size limit
        max_shares = int((account.equity * self.max_position_size) / estimated_price)
        shares = min(shares, max_shares)
        
        # Ensure we have buying power
        max_affordable = int(account.buying_power / estimated_price)
        shares = min(shares, max_affordable)
        
        return shares
    
    def get_risk_metrics(self, account: AccountInfo) -> Dict:
        """
        Get current risk metrics.
        
        Args:
            account: Account information
            
        Returns:
            Dict of risk metrics
        """
        metrics = {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': 0.0,
            'positions_count': self.positions_count,
            'cash_percentage': account.cash / account.equity if account.equity > 0 else 0,
            'margin_usage': account.margin_used / account.equity if account.equity > 0 else 0,
            'risk_capacity_used': 0.0
        }
        
        # Calculate daily P&L percentage
        if self.start_of_day_equity > 0:
            metrics['daily_pnl_pct'] = (account.equity - self.start_of_day_equity) / self.start_of_day_equity
        
        # Calculate risk capacity usage
        total_risk = account.positions_value / account.equity if account.equity > 0 else 0
        metrics['risk_capacity_used'] = total_risk / self.max_total_risk if self.max_total_risk > 0 else 0
        
        return metrics