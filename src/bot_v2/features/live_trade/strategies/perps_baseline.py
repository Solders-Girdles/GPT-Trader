"""
Baseline perpetuals trading strategy with MA crossover and trailing stops.

Phase 6: Minimal, production-safe strategy for perps trading.
Uses RiskManager constraints and ProductCatalog rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import logging

from ..risk import LiveRiskManager
from ...brokerages.core.interfaces import Product, OrderType, TimeInForce, MarketType

logger = logging.getLogger(__name__)


class Action(Enum):
    """Trading action decisions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Decision:
    """Strategy decision output."""
    action: Action
    target_notional: Optional[Decimal] = None
    qty: Optional[Decimal] = None
    leverage: Optional[int] = None
    stop_params: Optional[Dict[str, Any]] = None
    reduce_only: bool = False
    reason: str = ""
    # Advanced optional fields
    order_type: Optional[OrderType] = None
    stop_trigger: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    time_in_force: Optional[TimeInForce] = None
    post_only: Optional[bool] = None


@dataclass
class StrategyConfig:
    """Configuration for baseline strategy."""
    # MA parameters
    short_ma_period: int = 5
    long_ma_period: int = 20
    
    # Position management
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01  # 1% trailing stop (fixed)
    # Simple, predictable sizing: percentage of equity per trade, with optional USD cap
    position_fraction: float = 0.05  # 5% of equity
    max_trade_usd: Optional[Decimal] = None  # cap notional if set
    
    # Feature flags
    enable_shorts: bool = False
    max_adds: int = 0  # Disable pyramiding by default
    disable_new_entries: bool = False
    # Advanced entries (deprecated/no-op in simplified baseline)
    use_stop_entry: bool = False
    use_post_only: bool = False
    prefer_maker_orders: bool = False
    
    # Funding-awareness (deprecated for baseline; retained for backward compat, not used)
    funding_bias_bps: float = 0.0
    funding_block_long_bps: float = 0.0
    funding_block_short_bps: float = 0.0

    # Crossover robustness (optional)
    # Epsilon tolerance in basis points for crossover detection (0 = strict)
    ma_cross_epsilon_bps: Decimal = Decimal('0')
    # Bars to confirm crossover persistence (0 = no confirmation)
    ma_cross_confirm_bars: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class BaselinePerpsStrategy:
    """
    Simple MA crossover strategy with trailing stops for perpetuals.
    
    - Enter long on bullish MA crossover
    - Enter short on bearish MA crossover (if enabled)
    - Exit on opposing crossover or trailing stop hit
    - Respects RiskManager constraints and reduce-only mode
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        risk_manager: Optional[LiveRiskManager] = None
    ):
        """
        Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration
            risk_manager: Risk manager for constraint checks
        """
        self.config = config or StrategyConfig()
        self.risk_manager = risk_manager
        
        # In-memory state
        self.mark_windows: Dict[str, List[Decimal]] = {}
        self.position_adds: Dict[str, int] = {}  # Track adds per symbol
        self.trailing_stops: Dict[str, Tuple[Decimal, Decimal]] = {}  # symbol -> (peak, stop_price)
        
        logger.info(f"BaselinePerpsStrategy initialized with config: {self.config}")
    
    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: Optional[Dict[str, Any]],
        recent_marks: Optional[List[Decimal]],
        equity: Decimal,
        product: Product
    ) -> Decision:
        """
        Generate trading decision based on strategy logic.
        
        Args:
            symbol: Trading symbol
            current_mark: Current mark price
            position_state: Current position (qty, side, entry) or None
            recent_marks: Recent mark prices for MA calculation
            equity: Account equity for sizing
            product: Product metadata for rules
            
        Returns:
            Decision with action and parameters
        """
        # Update mark window
        if symbol not in self.mark_windows:
            self.mark_windows[symbol] = []
        
        # Use provided recent marks or maintain our own
        if recent_marks:
            self.mark_windows[symbol] = list(recent_marks)
        self.mark_windows[symbol].append(current_mark)
        
        # Keep window size reasonable
        max_window = max(self.config.short_ma_period, self.config.long_ma_period) + 5
        if len(self.mark_windows[symbol]) > max_window:
            self.mark_windows[symbol] = self.mark_windows[symbol][-max_window:]
        
        marks = self.mark_windows[symbol]
        
        # Check for existing position
        has_position = position_state and position_state.get('qty', 0) != 0
        
        # Check if we're in reduce-only mode
        if self.risk_manager and self.risk_manager.is_reduce_only_mode():
            if has_position:
                # Can only close in reduce-only mode
                return self._create_close_decision(symbol, position_state, "Reduce-only mode active")
            else:
                return Decision(action=Action.HOLD, reason="Reduce-only mode - no new entries")
        
        # Check if new entries are disabled
        if self.config.disable_new_entries:
            if has_position:
                # Check for exit signals
                signal = self._calculate_signal(marks)
                pos_side = position_state.get('side', '').lower()
                
                if (pos_side == 'long' and signal == 'bearish') or \
                   (pos_side == 'short' and signal == 'bullish'):
                    return self._create_close_decision(symbol, position_state, f"Exit on {signal} signal")
            
            return Decision(action=Action.HOLD, reason="New entries disabled")
        
        # Calculate MA signal
        signal = self._calculate_signal(marks)

        # Funding awareness intentionally disabled in simplified baseline

        # Generate decision based on signal and position
        if not has_position:
            # No position - check for entry
            if signal == 'bullish':
                return self._create_entry_decision(
                    symbol, Action.BUY, equity, product, "Bullish MA crossover"
                )
            elif signal == 'bearish' and self.config.enable_shorts:
                return self._create_entry_decision(
                    symbol, Action.SELL, equity, product, "Bearish MA crossover"
                )
        else:
            # Has position - check for exit or add
            pos_side = position_state.get('side', '').lower()
            pos_qty = abs(Decimal(str(position_state.get('qty', 0))))
            
            # Check for exit signal
            if pos_side == 'long' and signal == 'bearish':
                return self._create_close_decision(symbol, position_state, "Exit long on bearish signal")
            elif pos_side == 'short' and signal == 'bullish':
                return self._create_close_decision(symbol, position_state, "Exit short on bullish signal")
            
            # Check trailing stop after signal check (fixed pct)
            stop_decision = self._check_trailing_stop(symbol, current_mark, position_state, override_stop_pct=None)
            if stop_decision.action == Action.CLOSE:
                return stop_decision
            
            # Check for adding to position (pyramiding)
            adds = self.position_adds.get(symbol, 0)
            if adds < self.config.max_adds:
                if pos_side == 'long' and signal == 'bullish':
                    # Could add to long, but keep simple for now
                    pass
                elif pos_side == 'short' and signal == 'bearish' and self.config.enable_shorts:
                    # Could add to short, but keep simple for now
                    pass
        
        return Decision(action=Action.HOLD, reason="No signal")
    
    def _calculate_signal(self, marks: List[Decimal]) -> str:
        """
        Calculate MA crossover signal (event-based).

        Returns 'bullish' or 'bearish' only on a fresh crossover event
        on the most recent bar, otherwise 'neutral'.
        """
        if len(marks) < self.config.long_ma_period:
            return 'neutral'

        # Calculate current MAs
        short_ma = sum(marks[-self.config.short_ma_period:]) / self.config.short_ma_period
        long_ma = sum(marks[-self.config.long_ma_period:]) / self.config.long_ma_period

        # Require at least one additional bar for previous MA state
        if len(marks) >= self.config.long_ma_period + 1:
            prev_marks = marks[:-1]
            prev_short = sum(prev_marks[-self.config.short_ma_period:]) / self.config.short_ma_period
            prev_long = sum(prev_marks[-self.config.long_ma_period:]) / self.config.long_ma_period

            # Epsilon tolerance based on current price in bps
            current_mark = marks[-1]
            try:
                eps = current_mark * (self.config.ma_cross_epsilon_bps / Decimal('10000'))
            except Exception:
                eps = Decimal('0')

            prev_diff = prev_short - prev_long
            cur_diff = short_ma - long_ma

            bullish_cross = (prev_diff <= eps) and (cur_diff > eps)
            bearish_cross = (prev_diff >= -eps) and (cur_diff < -eps)

            # Optional confirmation: require persistence of the crossed state
            if self.config.ma_cross_confirm_bars > 0 and (bullish_cross or bearish_cross):
                confirm_bars = self.config.ma_cross_confirm_bars
                if len(marks) >= self.config.long_ma_period + confirm_bars + 1:
                    confirmed = True
                    for i in range(1, confirm_bars + 1):
                        check_marks = marks[:-i]
                        check_short = sum(check_marks[-self.config.short_ma_period:]) / self.config.short_ma_period
                        check_long = sum(check_marks[-self.config.long_ma_period:]) / self.config.long_ma_period
                        check_diff = check_short - check_long
                        if bullish_cross and check_diff <= eps:
                            confirmed = False
                            break
                        if bearish_cross and check_diff >= -eps:
                            confirmed = False
                            break
                    bullish_cross = bullish_cross and confirmed
                    bearish_cross = bearish_cross and confirmed
                else:
                    # Not enough history yet to confirm
                    bullish_cross = False
                    bearish_cross = False

            if bullish_cross:
                return 'bullish'
            if bearish_cross:
                return 'bearish'

        # No fresh crossover event on this bar
        return 'neutral'
    
    def _check_trailing_stop(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: Dict[str, Any],
        override_stop_pct: Optional[float] = None
    ) -> Decision:
        """
        Check and update trailing stop.
        
        Returns:
            Decision to close if stop hit, otherwise hold
        """
        pos_side = position_state.get('side', '').lower()
        
        # Initialize trailing stop if needed
        if symbol not in self.trailing_stops:
            # Initialize with current price and calculate stop from there
            stop_pct = Decimal(str(override_stop_pct if override_stop_pct is not None else self.config.trailing_stop_pct))
            if pos_side == 'long':
                stop_price = current_mark * (Decimal('1') - stop_pct)
            else:
                stop_price = current_mark * (Decimal('1') + stop_pct)
            self.trailing_stops[symbol] = (current_mark, stop_price)
        
        peak, stop_price = self.trailing_stops[symbol]
        
        if pos_side == 'long':
            # Update peak if higher
            if current_mark > peak:
                peak = current_mark
                stop_pct = Decimal(str(override_stop_pct if override_stop_pct is not None else self.config.trailing_stop_pct))
                stop_price = peak * (Decimal('1') - stop_pct)
                self.trailing_stops[symbol] = (peak, stop_price)
            
            # Check if stop hit
            if current_mark <= stop_price:
                return self._create_close_decision(
                    symbol, position_state,
                    f"Trailing stop hit (peak: {peak}, stop: {stop_price}, current: {current_mark})"
                )
        
        elif pos_side == 'short':
            # Update peak (lowest for short)
            if current_mark < peak:
                peak = current_mark
                stop_pct = Decimal(str(override_stop_pct if override_stop_pct is not None else self.config.trailing_stop_pct))
                stop_price = peak * (Decimal('1') + stop_pct)
                self.trailing_stops[symbol] = (peak, stop_price)
            
            # Check if stop hit
            if current_mark >= stop_price:
                return self._create_close_decision(
                    symbol, position_state,
                    f"Trailing stop hit (peak: {peak}, stop: {stop_price}, current: {current_mark})"
                )
        
        return Decision(action=Action.HOLD, reason="Position within trailing stop")

    # Volatility-based trailing stop removed for simplicity
    
    def _create_entry_decision(
        self,
        symbol: str,
        action: Action,
        equity: Decimal,
        product: Product,
        reason: str
    ) -> Decision:
        """Create entry decision with leverage-based sizing.

        - Notional = equity * target_leverage
        - Order type left unset (defaults to market in execution)

        Risk and caps are enforced at execution time by the risk engine.
        """
        # Leverage-based notional requested by strategy
        fraction = Decimal(str(self.config.position_fraction or 0))
        if fraction <= Decimal('0'):
            fraction = Decimal('0.05')  # conservative default sizing
        target_notional = equity * fraction

        if self.config.max_trade_usd is not None:
            try:
                cap = Decimal(str(self.config.max_trade_usd))
                target_notional = min(target_notional, cap)
            except Exception:
                pass

        leverage_value: Optional[int] = None
        if product.market_type == MarketType.PERPETUAL:
            try:
                lv = Decimal(str(self.config.target_leverage))
                if lv > Decimal('1'):
                    target_notional = target_notional * lv
            except Exception:
                lv = Decimal('1')
            try:
                leverage_value = int(self.config.target_leverage)
            except Exception:
                leverage_value = None

        # Reset position tracking
        self.position_adds[symbol] = 0
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]

        return Decision(
            action=action,
            target_notional=target_notional,
            leverage=leverage_value,
            reason=reason
        )
    
    def _create_close_decision(
        self,
        symbol: str,
        position_state: Dict[str, Any],
        reason: str
    ) -> Decision:
        """Create close/exit decision."""
        # Clear position tracking
        if symbol in self.position_adds:
            del self.position_adds[symbol]
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]
        
        # Get position qty to close
        qty = abs(Decimal(str(position_state.get('qty', 0))))
        
        return Decision(
            action=Action.CLOSE,
            qty=qty,
            reduce_only=True,  # Always reduce-only for exits
            reason=reason
        )

    def update_marks(self, symbol: str, marks: List[Decimal]) -> None:
        """Seed or update the internal mark window for a symbol.

        This helper is used by some tests to pre-populate recent prices
        without invoking decide() first.
        """
        # Normalize and store a bounded window similar to decide()
        self.mark_windows[symbol] = list(marks)
        max_window = max(self.config.short_ma_period, self.config.long_ma_period) + 5
        if len(self.mark_windows[symbol]) > max_window:
            self.mark_windows[symbol] = self.mark_windows[symbol][-max_window:]
    
    def reset(self, symbol: Optional[str] = None):
        """
        Reset strategy state.
        
        Args:
            symbol: Reset specific symbol or all if None
        """
        if symbol:
            if symbol in self.mark_windows:
                del self.mark_windows[symbol]
            if symbol in self.position_adds:
                del self.position_adds[symbol]
            if symbol in self.trailing_stops:
                del self.trailing_stops[symbol]
        else:
            self.mark_windows.clear()
            self.position_adds.clear()
            self.trailing_stops.clear()


# Functional wrapper for simple usage
def create_baseline_strategy(
    config: Optional[Dict[str, Any]] = None,
    risk_manager: Optional[LiveRiskManager] = None
) -> BaselinePerpsStrategy:
    """
    Create a baseline perpetuals strategy.
    
    Args:
        config: Strategy configuration dict
        risk_manager: Risk manager instance
        
    Returns:
        Configured strategy instance
    """
    strategy_config = StrategyConfig.from_dict(config) if config else StrategyConfig()
    return BaselinePerpsStrategy(config=strategy_config, risk_manager=risk_manager)
