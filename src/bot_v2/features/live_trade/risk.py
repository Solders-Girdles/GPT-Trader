"""
Risk management for perpetuals live trading.

Phase 5: Leverage-aware sizing and runtime guards.
Complete isolation - no strategy logic, pure risk controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Callable
import logging

from ...config.live_trade_config import RiskConfig
from ...persistence.event_store import EventStore
from ..brokerages.core.interfaces import Product, MarketType
from .guard_errors import (
    RiskGuardTelemetryError,
    RiskGuardDataCorrupt,
    RiskGuardComputationError,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Risk validation failure with clear message."""
    pass


@dataclass
class RiskRuntimeState:
    reduce_only_mode: bool = False
    last_reduce_only_reason: Optional[str] = None
    last_reduce_only_at: Optional[datetime] = None


class LiveRiskManager:
    """Risk management for perpetuals live trading.
    
    Enforces leverage limits, liquidation buffers, exposure caps,
    and runtime guards for safe trading.
    """
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        event_store: Optional[EventStore] = None,
        risk_info_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        """
        Initialize risk manager with configuration.
        
        Args:
            config: Risk configuration (defaults loaded from env)
            event_store: Event store for risk metrics/events
        """
        self.config = config or RiskConfig.from_env()
        self.event_store = event_store or EventStore()
        self._risk_info_provider = risk_info_provider
        
        # Runtime state
        self.daily_pnl = Decimal("0")
        self.start_of_day_equity = Decimal("0")
        self.last_mark_update: Dict[str, datetime] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}  # Track for exposure
        # Circuit breaker state
        self._cb_last_trigger: Dict[str, datetime] = {}
        # Time provider for testability
        self._now_provider = lambda: datetime.utcnow()
        self._state = RiskRuntimeState(reduce_only_mode=bool(self.config.reduce_only_mode))
        self._state_listener: Optional[Callable[[RiskRuntimeState], None]] = None

        # Log configuration
        if hasattr(self.config, 'to_dict'):
            logger.info(f"LiveRiskManager initialized with config: {self.config.to_dict()}")
        else:
            logger.info("LiveRiskManager initialized with a non-standard config object")

    # ========== Pre-trade Checks (sync) ==========

    def set_state_listener(self, listener: Optional[Callable[[RiskRuntimeState], None]]) -> None:
        self._state_listener = listener

    def is_reduce_only_mode(self) -> bool:
        return self._state.reduce_only_mode or bool(getattr(self.config, 'reduce_only_mode', False))

    def set_reduce_only_mode(self, enabled: bool, reason: str = "") -> None:
        """Toggle reduce-only mode without mutating the shared config object."""
        if self._state.reduce_only_mode == enabled and bool(getattr(self.config, 'reduce_only_mode', False)) == enabled:
            return

        self._state.reduce_only_mode = enabled
        if enabled:
            self._state.last_reduce_only_reason = reason or "unspecified"
            self._state.last_reduce_only_at = self._now()
        else:
            self._state.last_reduce_only_reason = None
            self._state.last_reduce_only_at = None

        # Mirror change onto config so legacy access patterns remain valid
        try:
            self.config.reduce_only_mode = enabled
        except Exception as exc:
            logger.debug("Failed to mirror reduce-only state onto config: %s", exc, exc_info=True)

        try:
            self.event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    'event_type': 'reduce_only_mode_changed',
                    'enabled': enabled,
                    'reason': reason or "unspecified",
                    'timestamp': self._state.last_reduce_only_at.isoformat() if self._state.last_reduce_only_at else None,
                }
            )
        except Exception as exc:
            logger.warning("Failed to persist reduce-only mode change: %s", exc)

        if self._state_listener:
            try:
                self._state_listener(self._state)
            except Exception:
                logger.exception("Reduce-only state listener failed")
    
    def pre_trade_validate(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        qty: Decimal,
        price: Decimal,
        product: Product,
        equity: Decimal,
        current_positions: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Validate order against all risk limits before placement.
        
        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            qty: Order quantity
            price: Expected execution price
            product: Product metadata
            equity: Current account equity
            current_positions: Current positions for exposure calc
        
        Raises:
            ValidationError: If any risk check fails
        """
        # Kill switch check
        if self.config.kill_switch_enabled:
            try:
                # Persist event via EventStore instead of cross-slice logger
                self.event_store.append_metric(
                    bot_id="risk_engine",
                    metrics={
                        'event_type': 'kill_switch',
                        'message': 'Kill switch enabled - trading halted',
                        'component': 'risk_manager',
                    }
                )
            except Exception:
                logger.exception("Failed to record kill switch metric")
            raise ValidationError("Kill switch enabled - all trading halted")
        
        # Reduce-only mode check
        if self.is_reduce_only_mode():
            if not self._is_reducing_position(symbol, side, current_positions):
                try:
                    self.event_store.append_metric(
                        bot_id="risk_engine",
                        metrics={
                            'event_type': 'reduce_only_block',
                            'symbol': symbol,
                            'message': f'Blocked increase for {symbol} (reduce-only)',
                            'component': 'risk_manager',
                        }
                    )
                except Exception:
                    logger.exception("Failed to record reduce-only block metric for %s", symbol)
                raise ValidationError(
                    f"Reduce-only mode active - cannot increase position for {symbol}"
                )
        
        # Run all validation checks
        self.validate_leverage(symbol, qty, price, product, equity)
        self.validate_liquidation_buffer(symbol, qty, price, product, equity)
        self.validate_exposure_limits(symbol, qty * price, equity, current_positions)
        
        # Project post-trade liquidation buffer using MMR/liq data if enabled
        if self.config.enable_pre_trade_liq_projection and product.market_type == MarketType.PERPETUAL:
            projected = self._project_liquidation_distance(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                equity=equity,
                current_positions=current_positions or {}
            )
            if projected < Decimal(str(self.config.min_liquidation_buffer_pct)):
                raise ValidationError(
                    f"Projected liquidation buffer {projected:.2%} < "
                    f"{self.config.min_liquidation_buffer_pct:.2%} for {symbol}"
                )
        
        # Optional slippage guard if mark price available
        if symbol in self.last_mark_update:
            self.validate_slippage_guard(symbol, side, qty, price, price)  # Use price as mark for now
    
    def _now(self) -> datetime:
        return self._now_provider()

    def _is_daytime(self, now: Optional[datetime] = None) -> Optional[bool]:
        """Determine if current time is within configured daytime window (UTC).

        Returns True/False if window configured, otherwise None.
        """
        start_s = getattr(self.config, 'daytime_start_utc', None)
        end_s = getattr(self.config, 'daytime_end_utc', None)
        if not start_s or not end_s:
            return None
        try:
            from datetime import time as dtime
            start_h, start_m = map(int, start_s.split(":"))
            end_h, end_m = map(int, end_s.split(":"))
            start = dtime(start_h, start_m)
            end = dtime(end_h, end_m)
            now = now or self._now()
            t = now.time()
            if start <= end:
                return start <= t < end
            # window spans midnight
            return t >= start or t < end
        except Exception as exc:
            logger.debug("Failed to evaluate daytime window: %s", exc, exc_info=True)
            return None

    def _effective_symbol_leverage_cap(self, symbol: str) -> int:
        """Compute effective per-symbol leverage cap using day/night schedule and provider data."""
        # Base: per-symbol or global
        cap = self.config.leverage_max_per_symbol.get(symbol, self.config.max_leverage)

        # Apply day/night overrides if configured
        is_day = self._is_daytime()
        try:
            if is_day is True and symbol in getattr(self.config, 'day_leverage_max_per_symbol', {}):
                cap = min(cap, int(self.config.day_leverage_max_per_symbol[symbol]))
            elif is_day is False and symbol in getattr(self.config, 'night_leverage_max_per_symbol', {}):
                cap = min(cap, int(self.config.night_leverage_max_per_symbol[symbol]))
        except Exception as exc:
            logger.debug("Failed to apply day/night leverage override for %s: %s", symbol, exc, exc_info=True)

        # Provider may return a stricter cap (if available)
        if self._risk_info_provider:
            try:
                info = self._risk_info_provider(symbol) or {}
                prov_cap = info.get('max_leverage') or info.get('leverage_cap')
                if prov_cap is not None:
                    cap = min(cap, int(prov_cap))
            except Exception as exc:
                logger.debug("Risk info provider failed for leverage cap on %s: %s", symbol, exc, exc_info=True)
        return int(cap)

    def _effective_mmr(self, symbol: str) -> Decimal:
        """Compute effective maintenance margin rate using provider and day/night schedule."""
        # Provider preferred if available
        if self._risk_info_provider:
            try:
                info = self._risk_info_provider(symbol) or {}
                raw = info.get('maintenance_margin_rate') or info.get('mmr')
                if raw is not None:
                    return Decimal(str(raw))
            except Exception as exc:
                logger.debug("Risk info provider failed for MMR on %s: %s", symbol, exc, exc_info=True)

        # Day/night schedule fallback
        is_day = self._is_daytime()
        try:
            if is_day is True and symbol in getattr(self.config, 'day_mmr_per_symbol', {}):
                return Decimal(str(self.config.day_mmr_per_symbol[symbol]))
            if is_day is False and symbol in getattr(self.config, 'night_mmr_per_symbol', {}):
                return Decimal(str(self.config.night_mmr_per_symbol[symbol]))
        except Exception as exc:
            logger.debug("Failed to apply day/night MMR override for %s: %s", symbol, exc, exc_info=True)

        # Fallback to default
        return Decimal(str(self.config.default_maintenance_margin_rate))

    def validate_leverage(
        self,
        symbol: str,
        qty: Decimal,
        price: Decimal,
        product: Product,
        equity: Decimal
    ) -> None:
        """
        Validate that order doesn't exceed leverage limits.
        
        Raises:
            ValidationError: If leverage exceeds caps
        """
        if product.market_type != MarketType.PERPETUAL:
            return  # Only check leverage for perpetuals
        
        # Calculate notional value
        notional = qty * price
        
        # Calculate implied leverage
        target_leverage = notional / equity if equity > 0 else float('inf')
        
        # Check effective per-symbol cap (day/night schedule, provider, global)
        symbol_cap = self._effective_symbol_leverage_cap(symbol)

        # Always report the effective symbol-specific cap when exceeded. Tests
        # expect messaging like "exceeds ETH-PERP cap of 10x" even when the
        # per-symbol cap falls back to the global value.
        if target_leverage > symbol_cap:
            raise ValidationError(
                f"Leverage {target_leverage:.1f}x exceeds {symbol} cap of {symbol_cap}x "
                f"(notional: {notional}, equity: {equity})"
            )

        # Global cap as a final safeguard when a per-symbol cap is looser
        # than the global limit.
        if target_leverage > self.config.max_leverage:
            raise ValidationError(
                f"Leverage {target_leverage:.1f}x exceeds global cap of {self.config.max_leverage}x"
            )
    
    def validate_liquidation_buffer(
        self,
        symbol: str,
        qty: Decimal,
        price: Decimal,
        product: Product,
        equity: Decimal
    ) -> None:
        """
        Ensure adequate buffer from liquidation after trade.
        
        Raises:
            ValidationError: If buffer would be too low
        """
        if product.market_type != MarketType.PERPETUAL:
            return
        
        # Simplified liquidation buffer check
        # In reality, would need liquidation price from exchange
        notional = qty * price
        
        # Estimate margin required (inverse of max leverage)
        max_leverage = self._effective_symbol_leverage_cap(symbol)
        margin_required = notional / max_leverage if max_leverage > 0 else notional
        
        # Check if we'd have enough buffer after this trade
        remaining_equity = equity - margin_required
        buffer_pct = remaining_equity / equity if equity > 0 else 0
        
        if buffer_pct < self.config.min_liquidation_buffer_pct:
            raise ValidationError(
                f"Insufficient liquidation buffer: {buffer_pct:.1%} < "
                f"{self.config.min_liquidation_buffer_pct:.1%} required "
                f"(margin needed: {margin_required}, equity: {equity})"
            )
    
    def validate_exposure_limits(
        self,
        symbol: str,
        notional: Decimal,
        equity: Decimal,
        current_positions: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Validate per-symbol and total exposure limits.
        
        Raises:
            ValidationError: If exposure exceeds caps
        """
        # Per-symbol exposure check
        # Enforce explicit per-symbol notional caps if configured
        max_notional_cap = getattr(self.config, 'max_notional_per_symbol', {}).get(symbol)
        if max_notional_cap is not None and notional > max_notional_cap:
            raise ValidationError(
                f"Symbol notional {notional} exceeds cap {max_notional_cap} for {symbol}"
            )

        symbol_exposure_pct = notional / equity if equity > 0 else float('inf')
        
        if symbol_exposure_pct > self.config.max_position_pct_per_symbol:
            raise ValidationError(
                f"Symbol exposure {symbol_exposure_pct:.1%} exceeds cap of "
                f"{self.config.max_position_pct_per_symbol:.1%} for {symbol}"
            )
        
        # Total portfolio exposure check
        total_exposure = notional
        if current_positions:
            for pos_symbol, pos_data in current_positions.items():
                if pos_symbol != symbol and isinstance(pos_data, dict):
                    if 'notional' in pos_data:
                        try:
                            pos_notional = abs(Decimal(str(pos_data.get('notional', 0))))
                        except Exception:
                            pos_notional = Decimal('0')
                    else:
                        pos_notional = abs(
                            Decimal(str(pos_data.get('qty', 0))) *
                            Decimal(str(pos_data.get('mark', pos_data.get('price', 0))))
                        )
                    total_exposure += pos_notional
        
        total_exposure_pct = total_exposure / equity if equity > 0 else float('inf')
        
        if total_exposure_pct > self.config.max_exposure_pct:
            raise ValidationError(
                f"Total exposure {total_exposure_pct:.1%} would exceed cap of "
                f"{self.config.max_exposure_pct:.1%} (new notional: {notional})"
            )
    
    def validate_slippage_guard(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        expected_price: Decimal,
        mark_or_quote: Decimal
    ) -> None:
        """
        Optional slippage guard based on spread.
        
        Raises:
            ValidationError: If expected slippage exceeds threshold
        """
        if self.config.slippage_guard_bps <= 0:
            return  # Disabled
        
        # Calculate expected slippage
        if side.lower() == "buy":
            # Buying above mark is slippage
            slippage = (expected_price - mark_or_quote) / mark_or_quote if mark_or_quote > 0 else 0
        else:
            # Selling below mark is slippage
            slippage = (mark_or_quote - expected_price) / mark_or_quote if mark_or_quote > 0 else 0
        
        slippage_bps = slippage * 10000  # Convert to basis points
        
        if slippage_bps > self.config.slippage_guard_bps:
            raise ValidationError(
                f"Expected slippage {slippage_bps:.0f} bps exceeds guard of "
                f"{self.config.slippage_guard_bps} bps for {symbol} "
                f"(price: {expected_price}, mark: {mark_or_quote})"
            )
    
    # ========== Runtime Guards (async/lightweight) ==========
    
    def track_daily_pnl(
        self,
        current_equity: Decimal,
        positions_pnl: Dict[str, Dict[str, Decimal]]
    ) -> bool:
        """
        Track daily PnL and trigger reduce-only if breaching limit.
        
        Args:
            current_equity: Current account equity
            positions_pnl: PnL data from Phase 4 states
        
        Returns:
            True if reduce-only mode was triggered
        """
        # Initialize start of day if needed
        if self.start_of_day_equity == 0:
            self.start_of_day_equity = current_equity
            return False
        
        # Calculate total PnL
        total_realized = Decimal("0")
        total_unrealized = Decimal("0")
        
        for symbol, pnl_data in positions_pnl.items():
            total_realized += pnl_data.get('realized_pnl', Decimal("0"))
            total_unrealized += pnl_data.get('unrealized_pnl', Decimal("0"))
        
        self.daily_pnl = total_realized + total_unrealized
        
        # Check daily loss limit
        daily_loss_abs = -self.daily_pnl  # Negative PnL = loss
        
        if daily_loss_abs > self.config.daily_loss_limit:
            # Trip reduce-only mode
            self.set_reduce_only_mode(True, reason="daily_loss_limit")
            
            # Log risk event
            self._log_risk_event(
                "daily_loss_breach",
                {
                    "daily_pnl": str(self.daily_pnl),
                    "daily_loss_abs": str(daily_loss_abs),
                    "limit": str(self.config.daily_loss_limit),
                    "action": "reduce_only_mode_enabled"
                },
                guard="daily_loss",
            )
            
            logger.warning(
                f"Daily loss limit breached: ${daily_loss_abs} > ${self.config.daily_loss_limit} - Enabling reduce-only mode"
            )
            
            return True
        
        return False
    
    def check_liquidation_buffer(
        self,
        symbol: str,
        position_data: Dict[str, Any],
        equity: Decimal
    ) -> bool:
        """
        Monitor liquidation buffer for position.

        Returns:
            True if reduce-only was set for this symbol
        """
        guard_name = "liquidation_buffer"

        try:
            qty = Decimal(str(position_data.get('qty', 0)))
            mark = Decimal(str(position_data.get('mark', 0)))
        except Exception as exc:
            raise RiskGuardDataCorrupt(
                guard=guard_name,
                message="Invalid position data for liquidation buffer",
                details={'symbol': symbol, 'position_data': position_data},
                original=exc,
            ) from exc
        # Optional real liquidation price from exchange (preferred when available)
        liq_raw = position_data.get('liquidation_price') or position_data.get('liq_price')
        liquidation_price: Optional[Decimal] = None
        try:
            if liq_raw is not None:
                liquidation_price = Decimal(str(liq_raw))
        except Exception as exc:
            raise RiskGuardDataCorrupt(
                guard=guard_name,
                message="Invalid liquidation price",
                details={'symbol': symbol, 'liquidation_price': liq_raw},
                original=exc,
            ) from exc

        if qty == 0 or mark == 0:
            return False

        notional = abs(qty * mark)

        # If we have a true liquidation price, compute distance-to-liquidation as buffer
        if liquidation_price is not None and mark > 0:
            try:
                # Distance from current mark to liquidation, normalized by mark
                buffer_pct = abs(mark - liquidation_price) / mark
            except Exception as exc:
                raise RiskGuardComputationError(
                    guard=guard_name,
                    message="Failed to compute buffer using liquidation price",
                    details={'symbol': symbol},
                    original=exc,
                ) from exc
        else:
            # Fallback: estimate via leverage (conservative)
            max_leverage = self.config.leverage_max_per_symbol.get(
                symbol,
                self.config.max_leverage
            )
            try:
                margin_used = notional / max_leverage if max_leverage > 0 else notional
                buffer_pct = (equity - margin_used) / equity if equity > 0 else 0
            except Exception as exc:
                raise RiskGuardComputationError(
                    guard=guard_name,
                    message="Failed to compute buffer using leverage fallback",
                    details={'symbol': symbol},
                    original=exc,
                ) from exc
        
        if buffer_pct < self.config.min_liquidation_buffer_pct:
            # Set reduce-only for this symbol
            if symbol not in self.positions:
                self.positions[symbol] = {}
            self.positions[symbol]['reduce_only'] = True
            
            # Log risk event
            self._log_risk_event(
                "liquidation_buffer_breach",
                {
                    "symbol": symbol,
                    "buffer_pct": str(buffer_pct),
                    "limit": str(self.config.min_liquidation_buffer_pct),
                    "action": f"reduce_only_enabled_for_{symbol}"
                },
                guard=guard_name,
            )
            
            logger.warning(
                f"Liquidation buffer breach for {symbol}: {buffer_pct:.2%} < "
                f"{self.config.min_liquidation_buffer_pct:.2%} - Setting reduce-only"
            )
            
            return True
        
        return False

    # ========== Projection Helpers ==========
    def set_risk_info_provider(self, provider: Callable[[str], Dict[str, Any]]) -> None:
        """Set a provider callable that returns exchange risk info for a symbol.

        The provider should return a dict with best-effort keys like
        'maintenance_margin_rate' or 'liquidation_price' for the CURRENT position.
        """
        self._risk_info_provider = provider

    def _project_liquidation_distance(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        price: Decimal,
        equity: Decimal,
        current_positions: Dict[str, Any]
    ) -> Decimal:
        """Project post-trade liquidation buffer fraction (best-effort).

        This uses maintenance margin rate when available (preferred) and falls back
        to leverage-based estimation. Returns a fractional buffer where 0.15 means 15%.
        """
        guard_name = "liquidation_projection"
        try:
            # Determine post-trade absolute quantity and side
            pos = current_positions.get(symbol) or {}
            cur_qty = abs(Decimal(str(pos.get('qty', 0))))
            cur_side = str(pos.get('side', '')).lower()
            order_side = side.lower()

            if cur_qty == 0:
                new_qty = abs(qty)
            else:
                if (cur_side == 'long' and order_side == 'buy') or (cur_side == 'short' and order_side == 'sell'):
                    new_qty = cur_qty + abs(qty)
                else:
                    # Reducing or flipping
                    reduce_qty = min(cur_qty, abs(qty))
                    residual = cur_qty - reduce_qty
                    new_qty = residual if residual > 0 else (abs(qty) - reduce_qty)

            if new_qty <= 0:
                return Decimal('1')  # Reducing-only → safe

            notional = Decimal(str(new_qty)) * Decimal(str(price))
            if equity <= 0 or notional <= 0:
                return Decimal('0')

            # Determine maintenance margin rate (provider preferred; else day/night schedule; else default)
            mmr = self._effective_mmr(symbol)

            # Equity buffer after accounting for maintenance requirement
            buffer = (Decimal(str(equity)) - (mmr * notional)) / Decimal(str(equity))
            if buffer < Decimal('0'):
                buffer = Decimal('0')
            if buffer > Decimal('1'):
                buffer = Decimal('1')
            return buffer
        except Exception as exc:
            raise RiskGuardComputationError(
                guard=guard_name,
                message="Failed to project liquidation distance",
                details={'symbol': symbol},
                original=exc,
            ) from exc
    
    def check_mark_staleness(self, symbol: str, mark_timestamp: Optional[datetime] = None) -> bool:
        """
        Check if mark price is stale.
        
        Returns:
            True if mark is stale and new orders should be halted
        """
        if mark_timestamp:
            self.last_mark_update[symbol] = mark_timestamp
        
        if symbol not in self.last_mark_update:
            return False  # No mark data yet
        
        age = datetime.utcnow() - self.last_mark_update[symbol]

        # Soft-warn if slightly stale; halt only if severely stale (>2x threshold)
        soft_limit = timedelta(seconds=self.config.max_mark_staleness_seconds)
        hard_limit = timedelta(seconds=self.config.max_mark_staleness_seconds * 2)

        if age > hard_limit:
            self._log_risk_event(
                "stale_mark_price",
                {
                    "symbol": symbol,
                    "age_seconds": str(age.total_seconds()),
                    "limit_seconds": str(self.config.max_mark_staleness_seconds),
                    "action": "halt_new_orders"
                },
                guard="mark_staleness",
            )
            logger.warning(
                f"Stale mark price for {symbol}: {age.total_seconds():.0f}s > hard limit {hard_limit.total_seconds():.0f}s - Halting new orders"
            )
            return True
        elif age > soft_limit:
            logger.info(
                f"Mark slightly stale for {symbol}: {age.total_seconds():.0f}s > {soft_limit.total_seconds():.0f}s - continuing"
            )
            return False

        return False
    
    def append_risk_metrics(self, equity: Decimal, positions: Dict[str, Any]) -> None:
        """
        Append periodic risk metrics snapshot to EventStore.
        
        Args:
            equity: Current account equity
            positions: Current positions with PnL data
        """
        # Calculate aggregate metrics
        total_notional = Decimal("0")
        max_leverage = Decimal("0")
        
        for symbol, pos_data in positions.items():
            qty = abs(Decimal(str(pos_data.get('qty', 0))))
            mark = Decimal(str(pos_data.get('mark', 0)))
            
            if qty > 0 and mark > 0:
                notional = qty * mark
                total_notional += notional
                
                # Track max leverage
                leverage = notional / equity if equity > 0 else 0
                max_leverage = max(max_leverage, leverage)
        
        # Calculate metrics
        exposure_pct = total_notional / equity if equity > 0 else 0
        daily_pnl_pct = self.daily_pnl / self.start_of_day_equity if self.start_of_day_equity > 0 else 0
        
        logger.debug(
            f"Risk snapshot: equity={equity} notional={total_notional} exposure={exposure_pct:.3f} "
            f"max_lev={max_leverage:.2f} daily_pnl={self.daily_pnl} daily_pnl_pct={daily_pnl_pct:.4f} "
            f"reduce_only={self.is_reduce_only_mode()} kill={self.config.kill_switch_enabled}"
        )

        try:
            # Persist metrics in EventStore for observability
            self.event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "timestamp": datetime.utcnow().isoformat(),
                    "equity": str(equity),
                    "total_notional": str(total_notional),
                    "exposure_pct": str(exposure_pct),
                    "max_leverage": str(max_leverage),
                    "daily_pnl": str(self.daily_pnl),
                    "daily_pnl_pct": str(daily_pnl_pct),
                    "reduce_only": str(self.is_reduce_only_mode()),
                    "kill_switch": str(self.config.kill_switch_enabled),
                }
            )
        except Exception as exc:
            raise RiskGuardTelemetryError(
                guard="risk_metrics",
                message="Failed to persist risk snapshot metric",
                details={
                    "equity": str(equity),
                    "total_notional": str(total_notional),
                    "exposure_pct": str(exposure_pct),
                },
                original=exc,
            ) from exc

    # ========== Correlation / Concentration ==========
    def check_correlation_risk(self, positions: Dict[str, Any]) -> bool:
        """Check portfolio correlation and concentration risk.

        Returns True if concentration exceeds a simple HHI threshold or if a high
        correlation condition is detected (best‑effort, requires historical data).
        """
        try:
            symbols = list(positions.keys())
            if len(symbols) < 2:
                return False

            # Concentration via Herfindahl–Hirschman Index (HHI)
            notional_vals: List[Decimal] = []
            for sym, p in positions.items():
                qty = abs(Decimal(str(p.get('qty', 0))))
                mark = Decimal(str(p.get('mark', 0)))
                notional_vals.append(qty * mark)
            total = sum(notional_vals) if notional_vals else Decimal('0')
            if total <= 0:
                return False
            hhi = sum((v / total) ** 2 for v in notional_vals)
            if hhi > Decimal('0.4'):
                self._log_risk_event("concentration_risk", {"hhi": str(hhi)}, guard="correlation_risk")
                logger.warning(f"Concentration risk detected (HHI={hhi:.3f})")
                return True

            # Correlation: requires rolling returns; not available by default.
            # This is a placeholder; full implementation can derive returns from EventStore metrics.
            return False
        except Exception as exc:
            raise RiskGuardComputationError(
                guard="correlation_risk",
                message="Failed to evaluate correlation risk",
                details={'symbols': list(positions.keys())},
                original=exc,
            ) from exc

    def check_volatility_circuit_breaker(self, symbol: str, recent_marks: List[Decimal]) -> Dict[str, Any]:
        """Check rolling volatility and trigger progressive circuit breakers.

        Returns a dict: {triggered: bool, action: Optional[str], volatility: float}
        """
        try:
            if not self.config.enable_volatility_circuit_breaker:
                return {"triggered": False}
            window = int(getattr(self.config, 'volatility_window_periods', 20))
            if len(recent_marks) < window:
                return {"triggered": False}

            # Build returns over window
            rets: List[float] = []
            for a, b in zip(recent_marks[-window:-1], recent_marks[-window+1:]):
                if a and a > 0:
                    try:
                        rets.append(float((b - a) / a))
                    except Exception:
                        continue
            if len(rets) < max(10, window // 2):
                return {"triggered": False}

            import statistics, math
            stdev = statistics.stdev(rets) if len(rets) > 1 else 0.0
            rolling_vol = float(stdev * math.sqrt(252.0))  # Annualized

            # Cooldown
            now = datetime.utcnow()
            cooldown_min = int(getattr(self.config, 'circuit_breaker_cooldown_minutes', 30))
            last_ts = self._cb_last_trigger.get(symbol)
            if last_ts and (now - last_ts) < timedelta(minutes=cooldown_min):
                # Still in cooldown; don't spam new actions
                return {"triggered": False, "volatility": rolling_vol}

            action = None
            warn_th = float(getattr(self.config, 'volatility_warning_threshold', 0.10))
            red_th = float(getattr(self.config, 'volatility_reduce_only_threshold', 0.12))
            kill_th = float(getattr(self.config, 'volatility_kill_switch_threshold', 0.15))

            if rolling_vol >= kill_th:
                self.config.kill_switch_enabled = True
                action = 'kill_switch'
            elif rolling_vol >= red_th:
                self.set_reduce_only_mode(True, reason="volatility_circuit_breaker")
                action = 'reduce_only'
            elif rolling_vol >= warn_th:
                action = 'warning'

            if action:
                self._cb_last_trigger[symbol] = now
                self._log_risk_event(
                    "volatility_circuit_breaker",
                    {
                        "symbol": symbol,
                        "rolling_volatility": f"{rolling_vol:.6f}",
                        "action": action,
                        "warning_threshold": warn_th,
                        "reduce_only_threshold": red_th,
                        "kill_switch_threshold": kill_th,
                    },
                    guard="volatility_circuit_breaker",
                )
                msg = f"Volatility CB: {symbol} vol={rolling_vol:.3f} action={action}"
                logger.warning(msg)
                return {"triggered": True, "action": action, "volatility": rolling_vol}

            return {"triggered": False, "volatility": rolling_vol}
        except Exception as exc:
            raise RiskGuardComputationError(
                guard="volatility_circuit_breaker",
                message="Volatility circuit breaker computation failed",
                details={'symbol': symbol},
                original=exc,
            ) from exc
    
    # ========== Helper Methods ==========
    
    def _is_reducing_position(
        self,
        symbol: str,
        side: str,
        current_positions: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if order would reduce existing position."""
        if not current_positions or symbol not in current_positions:
            return False  # No position to reduce
        
        pos_data = current_positions[symbol]
        pos_side = pos_data.get('side', '').lower()
        
        # Reducing if opposite sides
        if pos_side == 'long' and side.lower() == 'sell':
            return True
        if pos_side == 'short' and side.lower() == 'buy':
            return True
        
        return False
    
    def _log_risk_event(self, event_type: str, details: Dict[str, Any], *, guard: Optional[str] = None) -> None:
        """Log risk event to EventStore, surfacing telemetry failures."""

        try:
            self.event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "event_type": event_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    **details
                }
            )
        except Exception as exc:
            raise RiskGuardTelemetryError(
                guard=guard or event_type,
                message=f"Failed to persist risk event '{event_type}'",
                details=details,
                original=exc,
            ) from exc
    
    def reset_daily_tracking(self, current_equity: Decimal) -> None:
        """Reset daily tracking at start of new day."""
        self.daily_pnl = Decimal("0")
        self.start_of_day_equity = current_equity
        logger.info(f"Reset daily tracking - Start of day equity: {current_equity}")
