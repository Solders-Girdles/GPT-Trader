"""
Week 2 strategy enhancement interfaces.

Provides configurable filters and guards for risk management:
1. Market condition filters (spread, depth, volume, RSI)
2. Risk guards (liquidation distance, slippage impact)
3. Strategy enhancements (RSI confirmation, volatility adaptive)
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class MarketConditionFilters:
    """Configurable filters for market conditions."""
    
    # Spread and depth filters
    max_spread_bps: Optional[Decimal] = None        # Max spread in basis points
    min_depth_l1: Optional[Decimal] = None          # Min L1 depth in notional
    min_depth_l10: Optional[Decimal] = None         # Min L10 depth in notional
    
    # Volume filters
    min_volume_1m: Optional[Decimal] = None         # Min 1-minute volume
    min_volume_5m: Optional[Decimal] = None         # Min 5-minute volume
    
    # RSI filter for entry confirmation
    rsi_oversold: Optional[Decimal] = Decimal('30') # RSI oversold threshold
    rsi_overbought: Optional[Decimal] = Decimal('70') # RSI overbought threshold
    require_rsi_confirmation: bool = False           # Whether to require RSI alignment
    
    def should_allow_long_entry(self, market_snapshot: Dict[str, Any], rsi: Optional[Decimal] = None) -> tuple[bool, str]:
        """Check if market conditions allow long entry."""
        
        # Check spread
        if self.max_spread_bps and market_snapshot.get('spread_bps', 0) > self.max_spread_bps:
            return False, f"Spread too wide: {market_snapshot.get('spread_bps')} > {self.max_spread_bps} bps"
        
        # Check depth
        if self.min_depth_l1 and market_snapshot.get('depth_l1', 0) < self.min_depth_l1:
            return False, f"L1 depth insufficient: {market_snapshot.get('depth_l1')} < {self.min_depth_l1}"
            
        if self.min_depth_l10 and market_snapshot.get('depth_l10', 0) < self.min_depth_l10:
            return False, f"L10 depth insufficient: {market_snapshot.get('depth_l10')} < {self.min_depth_l10}"
        
        # Check volume
        if self.min_volume_1m and market_snapshot.get('vol_1m', 0) < self.min_volume_1m:
            return False, f"1m volume too low: {market_snapshot.get('vol_1m')} < {self.min_volume_1m}"
            
        if self.min_volume_5m and market_snapshot.get('vol_5m', 0) < self.min_volume_5m:
            return False, f"5m volume too low: {market_snapshot.get('vol_5m')} < {self.min_volume_5m}"
        
        # Check RSI confirmation for longs (should be oversold or neutral)
        if self.require_rsi_confirmation and rsi is not None:
            if rsi > self.rsi_overbought:
                return False, f"RSI too high for long entry: {rsi} > {self.rsi_overbought}"
        
        return True, "Market conditions acceptable"
    
    def should_allow_short_entry(self, market_snapshot: Dict[str, Any], rsi: Optional[Decimal] = None) -> tuple[bool, str]:
        """Check if market conditions allow short entry."""
        
        # Same spread/depth/volume checks as longs
        long_ok, reason = self.should_allow_long_entry(market_snapshot, None)  # Skip RSI check
        if not long_ok:
            return False, reason
        
        # Check RSI confirmation for shorts (should be overbought or neutral)
        if self.require_rsi_confirmation and rsi is not None:
            if rsi < self.rsi_oversold:
                return False, f"RSI too low for short entry: {rsi} < {self.rsi_oversold}"
        
        return True, "Market conditions acceptable"


@dataclass
class RiskGuards:
    """Risk management guards for position sizing and safety."""
    
    # Liquidation distance guard
    min_liquidation_buffer_pct: Optional[Decimal] = Decimal('15')  # Min % buffer to liquidation
    
    # Slippage impact guard  
    max_slippage_impact_bps: Optional[Decimal] = Decimal('20')     # Max expected slippage in bps
    
    def check_liquidation_distance(
        self, 
        entry_price: Decimal,
        position_size: Decimal, 
        leverage: Decimal,
        account_equity: Decimal,
        maintenance_margin_rate: Decimal = Decimal('0.05')  # 5% default
    ) -> tuple[bool, str]:
        """Check if position maintains safe distance from liquidation."""
        
        if not self.min_liquidation_buffer_pct:
            return True, "Liquidation guard disabled"
        
        # Approximate liquidation calculation
        # Liq price ≈ entry_price * (1 - (1/leverage) + maintenance_margin_rate)
        liquidation_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)
        
        # Calculate distance to liquidation as % of entry price
        price_diff = abs(entry_price - liquidation_price)
        distance_pct = (price_diff / entry_price) * 100
        
        if distance_pct < self.min_liquidation_buffer_pct:
            return False, f"Too close to liquidation: {distance_pct:.1f}% < {self.min_liquidation_buffer_pct}%"
        
        return True, f"Safe liquidation distance: {distance_pct:.1f}%"
    
    def check_slippage_impact(
        self,
        order_size: Decimal,
        market_snapshot: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check if order size would cause excessive slippage."""
        
        if not self.max_slippage_impact_bps:
            return True, "Slippage guard disabled"
        
        # Simple impact estimation using depth
        l1_depth = market_snapshot.get('depth_l1', 0)
        l10_depth = market_snapshot.get('depth_l10', 0) 
        mid_price = market_snapshot.get('mid', 0)
        
        if not l1_depth or not mid_price:
            return False, "Insufficient market data for slippage calculation"
        
        # Estimate impact: if order > L1, use L10; if order > L10, reject
        if order_size > l10_depth:
            return False, f"Order too large: {order_size} > L10 depth {l10_depth}"
        
        # More realistic impact model with square root for larger orders
        if order_size <= l1_depth:
            estimated_impact_bps = (order_size / l1_depth) * Decimal('5')  # ~5 bps for full L1
        else:
            # Non-linear impact for orders beyond L1
            l1_impact = Decimal('5')
            excess = order_size - l1_depth
            excess_depth = l10_depth - l1_depth if l10_depth > l1_depth else l1_depth
            
            # Use square root for more realistic large order impact
            # Full L10 = ~20 bps impact (more realistic)
            excess_ratio = min(excess / excess_depth, Decimal('1'))
            additional_impact = excess_ratio ** Decimal('0.5') * Decimal('20')
            estimated_impact_bps = l1_impact + additional_impact
        
        if estimated_impact_bps > self.max_slippage_impact_bps:
            return False, f"Estimated slippage too high: {estimated_impact_bps:.1f} > {self.max_slippage_impact_bps} bps"
        
        return True, f"Acceptable slippage: {estimated_impact_bps:.1f} bps"


@dataclass 
class StrategyEnhancements:
    """Enhanced strategy logic with adaptive parameters."""
    
    # RSI parameters for MA crossover confirmation
    rsi_period: int = 14
    rsi_confirmation_enabled: bool = True
    
    # Volatility adaptive parameters
    volatility_lookback: int = 20
    volatility_scaling_enabled: bool = False
    min_volatility_percentile: Decimal = Decimal('25')  # Only trade above 25th percentile
    
    def calculate_rsi(self, prices: list[Decimal], period: int = None) -> Optional[Decimal]:
        """Calculate RSI for the given price series."""
        period = period or self.rsi_period
        
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [abs(min(change, 0)) for change in changes]
        
        # Calculate initial average gain/loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Use Wilder's smoothing for remaining periods
        for i in range(period, len(changes)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return Decimal('100')  # No losses = RSI 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return Decimal(str(rsi))
    
    def should_confirm_ma_crossover(
        self,
        ma_signal: str,  # "buy" or "sell"
        prices: list[Decimal],
        rsi: Optional[Decimal] = None
    ) -> tuple[bool, str]:
        """Check if RSI confirms MA crossover signal."""
        
        if not self.rsi_confirmation_enabled:
            return True, "RSI confirmation disabled"
        
        if rsi is None:
            rsi = self.calculate_rsi(prices)
            
        if rsi is None:
            return False, "Insufficient price data for RSI calculation"
        
        if ma_signal == "buy":
            # For buy signals, prefer RSI below 70 (not overbought)
            if rsi > 70:
                return False, f"RSI too high for buy: {rsi} > 70"
            return True, f"RSI confirms buy signal: {rsi}"
            
        elif ma_signal == "sell":
            # For sell signals, prefer RSI above 30 (not oversold) 
            if rsi < 30:
                return False, f"RSI too low for sell: {rsi} < 30"
            return True, f"RSI confirms sell signal: {rsi}"
        
        return False, f"Unknown MA signal: {ma_signal}"


def create_conservative_filters() -> MarketConditionFilters:
    """Create conservative market condition filters for risk-averse trading."""
    return MarketConditionFilters(
        max_spread_bps=Decimal('10'),       # Max 10 bps spread
        min_depth_l1=Decimal('50000'),      # Min $50k L1 depth
        min_depth_l10=Decimal('200000'),    # Min $200k L10 depth
        min_volume_1m=Decimal('100000'),    # Min $100k 1m volume
        require_rsi_confirmation=True
    )


def create_aggressive_filters() -> MarketConditionFilters:
    """Create aggressive market condition filters for higher-risk trading.""" 
    return MarketConditionFilters(
        max_spread_bps=Decimal('25'),       # Max 25 bps spread
        min_depth_l1=Decimal('20000'),      # Min $20k L1 depth
        min_depth_l10=Decimal('100000'),    # Min $100k L10 depth
        min_volume_1m=Decimal('50000'),     # Min $50k 1m volume
        require_rsi_confirmation=False      # No RSI confirmation required
    )


def create_standard_risk_guards() -> RiskGuards:
    """Create standard risk management guards."""
    return RiskGuards(
        min_liquidation_buffer_pct=Decimal('20'),    # 20% liquidation buffer
        max_slippage_impact_bps=Decimal('15')        # Max 15 bps slippage
    )