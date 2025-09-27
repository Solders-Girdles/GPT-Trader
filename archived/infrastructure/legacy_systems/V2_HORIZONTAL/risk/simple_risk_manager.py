"""
Simple risk manager for position and portfolio protection.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from core.interfaces import IRiskManager, ComponentConfig
from core.types import Order, Portfolio, Position, RiskMetrics
from core.events import Event, EventType, RiskEvent, get_event_bus


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size: float = 0.25  # Max 25% in single position
    max_portfolio_risk: float = 0.02  # Max 2% portfolio risk per trade
    max_daily_loss: float = 0.05  # Max 5% daily loss
    max_leverage: float = 1.0  # No leverage by default
    max_positions: int = 10  # Max concurrent positions
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    min_position_value: float = 100  # Minimum $100 position
    max_concentration: float = 0.4  # Max 40% in single sector/asset


class SimpleRiskManager(IRiskManager):
    """
    Simple risk manager that enforces basic position and portfolio limits.
    
    Features:
    - Position size limits
    - Stop-loss and take-profit
    - Portfolio exposure limits
    - Daily loss limits
    - Concentration limits
    """
    
    def __init__(self, config: ComponentConfig, limits: Optional[RiskLimits] = None):
        """
        Initialize the risk manager.
        
        Args:
            config: Component configuration
            limits: Risk limits (uses defaults if not provided)
        """
        super().__init__(config)
        self.limits = limits or RiskLimits()
        self._event_bus = get_event_bus()
        self._daily_pnl = 0.0
        self._daily_reset_time = None
        
    def initialize(self) -> None:
        """Initialize the risk manager."""
        self._daily_reset_time = datetime.now()
        self._daily_pnl = 0.0
    
    def shutdown(self) -> None:
        """Cleanup and shutdown."""
        pass
    
    def validate_order(self, order: Order, portfolio: Portfolio) -> Tuple[bool, str]:
        """
        Validate an order against risk rules.
        
        Args:
            order: Order to validate
            portfolio: Current portfolio state
            
        Returns:
            (is_valid, reason_if_rejected)
        """
        # Check daily loss limit
        if self._check_daily_loss_breach(portfolio):
            self._publish_risk_event("daily_loss_limit", {
                'current_loss': self._daily_pnl,
                'limit': self.limits.max_daily_loss
            })
            return False, "Daily loss limit reached"
        
        # Check max positions
        if portfolio.position_count >= self.limits.max_positions:
            return False, f"Maximum positions ({self.limits.max_positions}) reached"
        
        # Calculate order value
        order_value = order.quantity * (order.limit_price or order.stop_price or 0)
        
        # Check minimum position value
        if order_value < self.limits.min_position_value:
            return False, f"Position value ${order_value:.2f} below minimum ${self.limits.min_position_value}"
        
        # Check position size limit
        position_pct = order_value / portfolio.total_value
        if position_pct > self.limits.max_position_size:
            self._publish_risk_event("position_size_limit", {
                'position_pct': position_pct,
                'limit': self.limits.max_position_size
            })
            return False, f"Position size {position_pct:.1%} exceeds limit {self.limits.max_position_size:.1%}"
        
        # Check leverage
        total_exposure = sum(
            pos.quantity * pos.current_price 
            for pos in portfolio.positions.values() 
            if pos.status.value == "open"
        ) + order_value
        
        leverage = total_exposure / portfolio.total_value
        if leverage > self.limits.max_leverage:
            return False, f"Leverage {leverage:.2f} exceeds limit {self.limits.max_leverage}"
        
        # Check concentration
        if not self._check_concentration(order, portfolio):
            return False, "Concentration limit exceeded"
        
        # All checks passed
        return True, "Order validated"
    
    def calculate_position_size(
        self,
        signal_strength: float,
        portfolio_value: float,
        current_positions: Dict[str, float]
    ) -> float:
        """
        Calculate appropriate position size based on risk.
        
        Uses Kelly-inspired sizing with risk limits.
        
        Args:
            signal_strength: Signal strength/confidence [0, 1]
            portfolio_value: Total portfolio value
            current_positions: Current position values by symbol
            
        Returns:
            Position size in dollars
        """
        # Base size is a fraction of max position size based on signal strength
        base_size = portfolio_value * self.limits.max_position_size * signal_strength
        
        # Apply portfolio risk limit (2% risk per trade)
        risk_based_size = portfolio_value * self.limits.max_portfolio_risk / self.limits.stop_loss_pct
        
        # Take the minimum of base and risk-based sizing
        position_size = min(base_size, risk_based_size)
        
        # Ensure minimum position value
        if position_size < self.limits.min_position_value:
            return 0.0  # Don't trade if below minimum
        
        # Adjust for current exposure
        total_exposure = sum(current_positions.values())
        remaining_capacity = (portfolio_value * self.limits.max_leverage) - total_exposure
        
        # Cap at remaining capacity
        position_size = min(position_size, remaining_capacity)
        
        return max(0, position_size)
    
    def get_risk_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        """
        Calculate current risk metrics.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Dict of risk metrics
        """
        metrics = {}
        
        # Portfolio exposure
        total_exposure = sum(
            pos.quantity * pos.current_price 
            for pos in portfolio.positions.values() 
            if pos.status.value == "open"
        )
        
        metrics['total_exposure'] = total_exposure
        metrics['leverage'] = total_exposure / portfolio.total_value if portfolio.total_value > 0 else 0
        
        # Position concentration
        position_values = [
            pos.quantity * pos.current_price
            for pos in portfolio.positions.values()
            if pos.status.value == "open"
        ]
        
        if position_values:
            max_position = max(position_values)
            metrics['max_concentration'] = max_position / portfolio.total_value
        else:
            metrics['max_concentration'] = 0
        
        # Daily P&L
        metrics['daily_pnl'] = self._daily_pnl
        metrics['daily_pnl_pct'] = (self._daily_pnl / portfolio.initial_capital) * 100
        
        # Risk utilization
        metrics['position_count'] = portfolio.position_count
        metrics['position_limit_usage'] = portfolio.position_count / self.limits.max_positions
        
        # Value at Risk (simplified)
        if position_values:
            # Assume 2-sigma daily move
            daily_volatility = 0.02  # 2% daily vol assumption
            var_95 = total_exposure * daily_volatility * 1.65  # 95% confidence
            var_99 = total_exposure * daily_volatility * 2.33  # 99% confidence
            metrics['var_95'] = var_95
            metrics['var_99'] = var_99
        else:
            metrics['var_95'] = 0
            metrics['var_99'] = 0
        
        return metrics
    
    def should_close_position(self, position: Position, current_price: float) -> bool:
        """
        Determine if a position should be closed for risk reasons.
        
        Args:
            position: Current position
            current_price: Current market price
            
        Returns:
            True if position should be closed
        """
        # Update position's current price
        position.current_price = current_price
        
        # Check stop-loss
        if position.stop_loss and current_price <= position.stop_loss:
            self._publish_risk_event("stop_loss_triggered", {
                'symbol': position.symbol,
                'entry_price': position.entry_price,
                'stop_price': position.stop_loss,
                'current_price': current_price
            })
            return True
        
        # Check take-profit
        if position.take_profit and current_price >= position.take_profit:
            self._publish_risk_event("take_profit_triggered", {
                'symbol': position.symbol,
                'entry_price': position.entry_price,
                'target_price': position.take_profit,
                'current_price': current_price
            })
            return True
        
        # Check percentage-based stop-loss
        pnl_pct = position.return_pct / 100
        if pnl_pct <= -self.limits.stop_loss_pct:
            self._publish_risk_event("stop_loss_triggered", {
                'symbol': position.symbol,
                'loss_pct': pnl_pct,
                'limit': self.limits.stop_loss_pct
            })
            return True
        
        # Check percentage-based take-profit
        if pnl_pct >= self.limits.take_profit_pct:
            self._publish_risk_event("take_profit_triggered", {
                'symbol': position.symbol,
                'gain_pct': pnl_pct,
                'target': self.limits.take_profit_pct
            })
            return True
        
        return False
    
    def _check_daily_loss_breach(self, portfolio: Portfolio) -> bool:
        """Check if daily loss limit has been breached."""
        # Reset daily P&L if new day
        now = datetime.now()
        if self._daily_reset_time and now.date() != self._daily_reset_time.date():
            self._daily_pnl = 0
            self._daily_reset_time = now
        
        # Calculate current day's P&L (simplified)
        daily_return = portfolio.total_return / 100
        
        # Check against limit
        return daily_return <= -self.limits.max_daily_loss
    
    def _check_concentration(self, order: Order, portfolio: Portfolio) -> bool:
        """Check concentration limits."""
        # Calculate what concentration would be after order
        order_value = order.quantity * (order.limit_price or order.stop_price or 0)
        
        # Find existing position in same symbol
        existing_value = 0
        if order.symbol in portfolio.positions:
            pos = portfolio.positions[order.symbol]
            if pos.status.value == "open":
                existing_value = pos.quantity * pos.current_price
        
        total_position = existing_value + order_value
        concentration = total_position / portfolio.total_value
        
        return concentration <= self.limits.max_concentration
    
    def _publish_risk_event(self, risk_type: str, details: Dict[str, Any]) -> None:
        """Publish a risk event."""
        event = RiskEvent(
            risk_type=risk_type,
            details=details,
            source=self.name
        )
        self._event_bus.publish(event)
        self.notify(event)