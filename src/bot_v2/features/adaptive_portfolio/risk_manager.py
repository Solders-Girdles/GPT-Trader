"""
Adaptive risk management based on portfolio tier.

Provides tier-specific risk calculations and limits.
"""

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from typing import Dict, List, Optional, Tuple
import logging

from .types import (
    PortfolioConfig, TierConfig, PortfolioSnapshot, 
    PositionInfo, RiskProfile
)


class AdaptiveRiskManager:
    """Risk management that adapts to portfolio tier."""
    
    def __init__(self, config: PortfolioConfig):
        """Initialize with portfolio configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_risk_metrics(
        self, 
        portfolio_snapshot: PortfolioSnapshot, 
        tier_config: TierConfig
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for current tier.
        
        Args:
            portfolio_snapshot: Current portfolio state
            tier_config: Current tier configuration
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Basic risk metrics
        metrics['total_value'] = portfolio_snapshot.total_value
        metrics['cash_pct'] = portfolio_snapshot.cash / portfolio_snapshot.total_value * 100
        metrics['invested_pct'] = 100 - metrics['cash_pct']
        
        # Position concentration
        metrics['positions_count'] = portfolio_snapshot.positions_count
        metrics['largest_position_pct'] = portfolio_snapshot.largest_position_pct
        metrics['position_concentration_risk'] = self._calculate_concentration_risk(
            portfolio_snapshot.positions
        )
        
        # Daily risk
        metrics['daily_pnl_pct'] = portfolio_snapshot.daily_pnl_pct
        metrics['daily_risk_pct'] = abs(portfolio_snapshot.daily_pnl_pct)
        metrics['daily_risk_limit_pct'] = tier_config.risk.daily_limit_pct
        metrics['daily_risk_utilization_pct'] = (
            metrics['daily_risk_pct'] / metrics['daily_risk_limit_pct'] * 100
        )
        
        # Position sizing risk
        metrics['avg_position_size'] = self._calculate_avg_position_size(
            portfolio_snapshot.positions, portfolio_snapshot.total_value
        )
        metrics['position_size_variance'] = self._calculate_position_size_variance(
            portfolio_snapshot.positions, portfolio_snapshot.total_value
        )
        
        # Tier compliance
        metrics['tier_compliant'] = self._check_tier_compliance(
            portfolio_snapshot, tier_config
        )
        
        # Risk-adjusted metrics
        metrics['risk_adjusted_score'] = self._calculate_risk_adjusted_score(
            portfolio_snapshot, tier_config
        )
        
        # Drawdown risk (simplified - would need historical data)
        metrics['estimated_max_drawdown_pct'] = self._estimate_max_drawdown(
            portfolio_snapshot, tier_config
        )
        
        return metrics
    
    def check_position_size_limits(
        self, 
        position_value: float, 
        total_portfolio_value: float,
        tier_config: TierConfig
    ) -> Tuple[bool, str]:
        """
        Check if position size is within tier limits.
        
        Args:
            position_value: Value of the position
            total_portfolio_value: Total portfolio value
            tier_config: Current tier configuration
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check minimum position size
        if position_value < tier_config.min_position_size:
            return False, f"Position too small: ${position_value:,.0f} < ${tier_config.min_position_size:,.0f} minimum"
        
        # Check maximum position percentage (general rule - 25% max)
        position_pct = position_value / total_portfolio_value * 100
        max_position_pct = self.config.validation.get("max_position_size_pct", 25.0)
        
        if position_pct > max_position_pct:
            return False, f"Position too large: {position_pct:.1f}% > {max_position_pct}% maximum"
        
        # Check if it would exceed max positions for tier
        # (This would need current position count as input)
        
        return True, "Position size acceptable"
    
    def calculate_position_size(
        self, 
        total_portfolio_value: float,
        tier_config: TierConfig,
        confidence: float = 1.0,
        current_positions: int = 0
    ) -> float:
        """
        Calculate appropriate position size for tier and conditions.
        
        Args:
            total_portfolio_value: Total portfolio value
            tier_config: Current tier configuration
            confidence: Signal confidence (0.0 to 1.0)
            current_positions: Current number of positions
            
        Returns:
            Recommended position size in dollars
        """
        # Base position size from tier target
        target_positions = tier_config.positions.target_positions
        base_position_size = total_portfolio_value / target_positions
        
        # Adjust for confidence
        confidence_adjusted_size = base_position_size * confidence
        
        # Ensure minimum position size
        min_size = tier_config.min_position_size
        position_size = max(confidence_adjusted_size, min_size)
        
        # Ensure maximum position percentage
        max_position_pct = self.config.validation.get("max_position_size_pct", 25.0)
        max_size = total_portfolio_value * max_position_pct / 100
        position_size = min(position_size, max_size)
        
        # Conservative adjustment for small portfolios
        if total_portfolio_value < 5000:
            # Be more conservative with small portfolios
            position_size *= 0.8
        
        self.logger.info(
            f"Calculated position size: ${position_size:,.0f} "
            f"(confidence: {confidence:.2f}, tier: {tier_config.name})"
        )
        
        return position_size
    
    def check_trading_frequency_limits(
        self, 
        trades_this_week: int, 
        tier_config: TierConfig
    ) -> Tuple[bool, str]:
        """
        Check if trading frequency is within tier limits.
        
        Args:
            trades_this_week: Number of trades executed this week
            tier_config: Current tier configuration
            
        Returns:
            Tuple of (can_trade, reason)
        """
        max_trades = tier_config.trading.max_trades_per_week
        
        if trades_this_week >= max_trades:
            return False, f"Weekly trade limit reached: {trades_this_week}/{max_trades}"
        
        # PDT check for small accounts
        if tier_config.trading.pdt_compliant and trades_this_week >= 2:
            return False, "Approaching PDT limit (3 day trades per week)"
        
        return True, f"Can trade: {trades_this_week}/{max_trades} this week"
    
    def calculate_stop_loss_price(
        self, 
        entry_price: float, 
        tier_config: TierConfig,
        position_direction: str = "LONG"
    ) -> float:
        """
        Calculate appropriate stop loss price for tier.
        
        Args:
            entry_price: Entry price for position
            tier_config: Current tier configuration
            position_direction: "LONG" or "SHORT"
            
        Returns:
            Stop loss price
        """
        stop_loss_pct = tier_config.risk.position_stop_loss_pct
        
        if position_direction == "LONG":
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        else:  # SHORT
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
        
        return stop_loss_price
    
    def assess_portfolio_risk_level(
        self, 
        portfolio_snapshot: PortfolioSnapshot,
        tier_config: TierConfig
    ) -> Tuple[str, float, List[str]]:
        """
        Assess overall portfolio risk level.
        
        Args:
            portfolio_snapshot: Current portfolio state
            tier_config: Current tier configuration
            
        Returns:
            Tuple of (risk_level, risk_score, risk_factors)
        """
        risk_factors = []
        risk_score = 0
        
        # Daily risk utilization
        daily_risk_utilization = abs(portfolio_snapshot.daily_pnl_pct) / tier_config.risk.daily_limit_pct
        if daily_risk_utilization > 0.8:
            risk_factors.append("High daily risk utilization")
            risk_score += 3
        elif daily_risk_utilization > 0.5:
            risk_factors.append("Moderate daily risk utilization")
            risk_score += 1
        
        # Position concentration
        if portfolio_snapshot.largest_position_pct > 25:
            risk_factors.append("High position concentration")
            risk_score += 3
        elif portfolio_snapshot.largest_position_pct > 15:
            risk_factors.append("Moderate position concentration")
            risk_score += 1
        
        # Portfolio size vs tier
        total_value = portfolio_snapshot.total_value
        tier_min, tier_max = tier_config.range
        
        if total_value < tier_min * 1.1:  # Close to tier minimum
            risk_factors.append("Portfolio near tier minimum")
            risk_score += 2
        
        # Cash level
        cash_pct = portfolio_snapshot.cash / total_value * 100
        if cash_pct < 5:
            risk_factors.append("Very low cash reserves")
            risk_score += 2
        elif cash_pct > 50:
            risk_factors.append("High cash allocation (opportunity cost)")
            risk_score += 1
        
        # Position count vs tier
        if portfolio_snapshot.positions_count > tier_config.positions.max_positions:
            risk_factors.append("Too many positions for tier")
            risk_score += 3
        elif portfolio_snapshot.positions_count < tier_config.positions.min_positions:
            risk_factors.append("Too few positions (lack of diversification)")
            risk_score += 2
        
        # Determine risk level
        if risk_score >= 8:
            risk_level = "HIGH"
        elif risk_score >= 4:
            risk_level = "MEDIUM"
        elif risk_score >= 1:
            risk_level = "LOW"
        else:
            risk_level = "VERY_LOW"
        
        return risk_level, risk_score, risk_factors
    
    def _calculate_concentration_risk(self, positions: List[PositionInfo]) -> float:
        """Calculate portfolio concentration risk using Herfindahl index."""
        if not positions:
            return 0
        
        total_value = sum(pos.position_value for pos in positions)
        if total_value == 0:
            return 0
        
        # Calculate Herfindahl-Hirschman Index
        hhi = sum((pos.position_value / total_value) ** 2 for pos in positions)
        
        # Normalize to 0-100 scale (100 = maximum concentration)
        max_hhi = 1.0  # Single position
        min_hhi = 1 / len(positions)  # Equal weight
        
        if max_hhi == min_hhi:
            return 0
        
        concentration_risk = (hhi - min_hhi) / (max_hhi - min_hhi) * 100
        return concentration_risk
    
    def _calculate_avg_position_size(
        self, 
        positions: List[PositionInfo], 
        total_value: float
    ) -> float:
        """Calculate average position size as percentage of portfolio."""
        if not positions or total_value == 0:
            return 0
        
        return sum(pos.position_value for pos in positions) / len(positions) / total_value * 100
    
    def _calculate_position_size_variance(
        self, 
        positions: List[PositionInfo], 
        total_value: float
    ) -> float:
        """Calculate variance in position sizes."""
        if not positions or total_value == 0:
            return 0
        
        position_pcts = [pos.position_value / total_value * 100 for pos in positions]
        if len(position_pcts) <= 1:
            return 0
        
        # Calculate variance without numpy
        mean = sum(position_pcts) / len(position_pcts)
        variance = sum((x - mean) ** 2 for x in position_pcts) / len(position_pcts)
        return variance
    
    def _check_tier_compliance(
        self, 
        portfolio_snapshot: PortfolioSnapshot,
        tier_config: TierConfig
    ) -> bool:
        """Check if portfolio is compliant with tier requirements."""
        
        # Check position count
        if portfolio_snapshot.positions_count > tier_config.positions.max_positions:
            return False
        
        # Check position sizes
        for position in portfolio_snapshot.positions:
            if position.position_value < tier_config.min_position_size:
                return False
        
        # Check daily risk
        if abs(portfolio_snapshot.daily_pnl_pct) > tier_config.risk.daily_limit_pct:
            return False
        
        return True
    
    def _calculate_risk_adjusted_score(
        self, 
        portfolio_snapshot: PortfolioSnapshot,
        tier_config: TierConfig
    ) -> float:
        """Calculate overall risk-adjusted score (0-100)."""
        
        score = 50  # Start with neutral score
        
        # Adjust for daily performance relative to risk budget
        risk_utilization = abs(portfolio_snapshot.daily_pnl_pct) / tier_config.risk.daily_limit_pct
        if risk_utilization > 1:
            score -= 30  # Exceeding risk limits
        elif risk_utilization > 0.8:
            score -= 15  # High risk utilization
        elif risk_utilization < 0.2:
            score += 10  # Conservative risk usage
        
        # Adjust for diversification
        positions_count = portfolio_snapshot.positions_count
        target_positions = tier_config.positions.target_positions
        
        diversification_ratio = positions_count / target_positions
        if diversification_ratio > 1.2:
            score -= 10  # Over-diversified
        elif diversification_ratio < 0.8:
            score -= 15  # Under-diversified
        else:
            score += 10  # Good diversification
        
        # Adjust for concentration
        if portfolio_snapshot.largest_position_pct > 25:
            score -= 20
        elif portfolio_snapshot.largest_position_pct < 10:
            score += 10
        
        return max(0, min(100, score))
    
    def _estimate_max_drawdown(
        self, 
        portfolio_snapshot: PortfolioSnapshot,
        tier_config: TierConfig
    ) -> float:
        """Estimate potential maximum drawdown based on current risk."""
        
        # Simple estimation based on position sizes and tier risk
        base_drawdown = tier_config.risk.daily_limit_pct * 5  # 5 bad days
        
        # Adjust for concentration
        concentration_multiplier = 1 + portfolio_snapshot.largest_position_pct / 100
        
        # Adjust for number of positions
        diversification_factor = max(0.5, 1 - (portfolio_snapshot.positions_count - 1) * 0.1)
        
        estimated_drawdown = base_drawdown * concentration_multiplier * diversification_factor
        
        return min(estimated_drawdown, 50)  # Cap at 50%