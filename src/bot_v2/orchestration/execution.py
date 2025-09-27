from __future__ import annotations

from dataclasses import dataclass, field, asdict
import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Callable

from ..features.brokerages.core.interfaces import IBrokerage
from ..persistence import event_store as event_store_mod


@dataclass
class PaperPosition:
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    total_commission: float = 0.0  # Track accumulated commission


@dataclass
class PaperTrade:
    timestamp: datetime
    symbol: str
    side: str
    quantity: Decimal
    price: float
    value: float
    commission: float
    pnl: float | None = None
    reason: str = ""


class PaperExecutionEngine:
    """Enhanced paper execution with portfolio constraints, product rules, and persistence."""

    def __init__(self, commission: float = 0.006, slippage: float = 0.001, 
                 initial_capital: float = 10_000, config: Optional[Dict] = None,
                 bot_id: Optional[str] = None, symbols: Optional[List[str]] = None,
                 quote_provider: Optional[Callable[[str], Optional[float]]] = None,
                 broker: Optional[IBrokerage] = None):
        self.commission = commission
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        
        # Portfolio constraints from config
        self.config = config or {}
        self.max_position_pct = self.config.get('max_position_pct', 1.00)  # default allow 100%
        self.max_exposure = self.config.get('max_exposure', 1.00)  # default allow 100%
        self.min_cash_reserve = self.config.get('min_cash', 0.0)  # default no reserve
        self.min_order_value = self.config.get('min_order_value', 10.0)  # $10 min notional

        # Product catalog for rules enforcement
        self._product_catalog = None
        
        # Quote provider and broker (optional, decoupled)
        self.quote_provider = quote_provider
        self._broker = broker  # No auto-init, injected only
        
        # Event store for persistence
        self._event_store = event_store_mod.EventStore()
        
        # Bot ID convention: paper:<symbols-joined>
        if bot_id:
            self.bot_id = bot_id
        elif symbols:
            symbols_str = "-".join(s.replace("-", "") for s in symbols[:3])  # e.g., BTCUSD-ETHUSD
            self.bot_id = f"paper:{symbols_str}"
        else:
            self.bot_id = f"paper:default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log initial state
        self._log_metrics()

    def set_broker(self, broker: IBrokerage) -> None:
        """Set broker for market data (optional)."""
        self._broker = broker

    def connect(self) -> bool:
        """Connect to broker if provided."""
        if self._broker:
            try:
                connected = self._broker.connect()
                if connected:
                    # Initialize product catalog after connection
                    from ..features.brokerages.coinbase.utils import ProductCatalog
                    self._product_catalog = ProductCatalog()
                return connected
            except Exception:
                return False
        return True  # No broker is ok for offline mode
    
    def validate_order(self, symbol: str, amount_usd: float) -> bool:
        """Validate order against portfolio constraints."""
        # Check if we have enough cash
        if amount_usd > self.cash:
            return False
        
        # Check position sizing constraint (max % per position)
        base_equity = self.initial_capital  # use initial capital as baseline for limits
        max_pos_pct = self.config.get('max_position_pct', self.max_position_pct)
        if amount_usd > base_equity * max_pos_pct:
            return False
        
        # Check aggregate exposure limit
        current_exposure = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        max_exposure_pct = self.config.get('max_exposure_pct', self.config.get('max_exposure', self.max_exposure))
        if (current_exposure + amount_usd) > base_equity * max_exposure_pct:
            return False
        
        # Check minimum cash reserve
        min_cash_abs = self.config.get('min_cash', self.min_cash_reserve)
        min_cash_pct = self.config.get('min_cash_pct', None)
        required_min_cash = max(min_cash_abs or 0.0, (base_equity * float(min_cash_pct)) if min_cash_pct is not None else 0.0)
        if self.cash - amount_usd < required_min_cash:
            return False
        
        return True
    
    def enforce_product_rules(self, symbol: str, quantity: float, price: float) -> tuple[float, float]:
        """Enforce exchange product rules (min size, step size, notional).

        Supports either an injected simple dict `product_catalog` used in tests or
        a dynamic catalog obtained from the connected broker.
        """
        # Test-injected simple catalog
        rules = None
        if hasattr(self, 'product_catalog') and isinstance(getattr(self, 'product_catalog'), dict):
            rules = self.product_catalog.get(symbol)
        
        try:
            if rules:
                step = Decimal(str(rules.get('step_size', '0.00000001')))
                min_size = Decimal(str(rules.get('min_size', '0')))
                min_notional = Decimal(str(rules.get('min_notional', '0')))
                q = Decimal(str(quantity))
                # Quantize down to nearest step
                steps = (q / step).to_integral_value(rounding=None)
                q = steps * step
                if q < min_size:
                    raise ValueError(f"Quantity {q} below minimum {min_size}")
                notional = q * Decimal(str(price))
                if min_notional and notional < min_notional:
                    raise ValueError(f"Notional {notional} below minimum {min_notional}")
                return float(q), float(price)

            # Dynamic catalog via broker utils
            if self._product_catalog and getattr(self._broker, "_connected", False):
                from bot_v2.features.brokerages.coinbase.utils import quantize_to_increment
                product = self._product_catalog.get(self._broker._client, symbol)
                quantity = quantize_to_increment(Decimal(str(quantity)), product.step_size)
                price = quantize_to_increment(Decimal(str(price)), product.price_increment)
                if quantity < product.min_size:
                    raise ValueError(f"Quantity {quantity} below minimum {product.min_size}")
                notional = quantity * price
                if product.min_notional and notional < product.min_notional:
                    raise ValueError(f"Notional {notional} below minimum {product.min_notional}")
                return float(quantity), float(price)

            # No catalog available
            return quantity, price

        except ValueError:
            raise
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not enforce product rules: {e}")
            return quantity, price
    
    def round_to_increment(self, price: float, symbol: str, increment: float = 0.01) -> float:
        """Round price to valid increment for the symbol.

        Uses Decimal-based rounding to avoid float artifacts.
        """
        if increment <= 0:
            return price
        from decimal import Decimal, ROUND_HALF_EVEN
        inc = Decimal(str(increment))
        steps = (Decimal(str(price)) / inc).quantize(Decimal('1'), rounding=ROUND_HALF_EVEN)
        result = steps * inc
        return float(result)
    
    def calculate_equity(self) -> float:
        """Calculate total portfolio equity (cash + position values)."""
        position_value = 0.0
        for pos in self.positions.values():
            # Use current price if available, otherwise entry price
            price = pos.current_price if pos.current_price > 0 else pos.entry_price
            position_value += pos.quantity * price
        return self.cash + position_value
    
    def _log_metrics(self) -> None:
        """Log current portfolio metrics to event store."""
        try:
            positions_value = sum(
                pos.quantity * (pos.current_price if pos.current_price > 0 else pos.entry_price)
                for pos in self.positions.values()
            )
            
            metrics = {
                'equity': self.calculate_equity(),
                'cash': self.cash,
                'positions_value': positions_value,
                'positions_count': len(self.positions),
                'total_trades': len(self.trades),
                'initial_capital': self.initial_capital,
                'returns_pct': ((self.calculate_equity() - self.initial_capital) / self.initial_capital) * 100
            }
            
            self._event_store.append_metric(self.bot_id, metrics)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to log metrics: {e}")
    
    def _log_trade(self, trade: PaperTrade) -> None:
        """Log trade to event store."""
        try:
            trade_dict = asdict(trade) if hasattr(trade, '__dataclass_fields__') else {
                'timestamp': trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp),
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'pnl': trade.pnl,
                'reason': trade.reason
            }
            
            self._event_store.append_trade(self.bot_id, trade_dict)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to log trade: {e}")
    
    def _log_positions(self) -> None:
        """Log current positions to event store."""
        try:
            for symbol, pos in self.positions.items():
                position_dict = {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'entry_time': pos.entry_time.isoformat() if hasattr(pos.entry_time, 'isoformat') else str(pos.entry_time),
                    'unrealized_pnl': (pos.current_price - pos.entry_price) * pos.quantity if pos.current_price > 0 else 0
                }
                
                self._event_store.append_position(self.bot_id, position_dict)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to log positions: {e}")

    def get_mid(self, symbol: str) -> Optional[float]:
        """Get mid price from quote provider, broker, or fallback."""
        # Priority 1: External quote provider object
        if self.quote_provider:
            try:
                # Object-style provider with get_quote()
                if hasattr(self.quote_provider, 'get_quote'):
                    q = self.quote_provider.get_quote(symbol)
                    if isinstance(q, dict) and 'mid' in q:
                        return float(q['mid'])
                    if hasattr(q, 'bid') and hasattr(q, 'ask'):
                        return float((q.bid + q.ask) / 2)
                # Callable provider
                if callable(self.quote_provider):
                    v = self.quote_provider(symbol)
                    return float(v) if v is not None else None
            except Exception:
                pass

        # Priority 2: Broker if injected
        if self._broker:
            try:
                if hasattr(self._broker, 'get_mid'):
                    return float(self._broker.get_mid(symbol))
                if hasattr(self._broker, 'get_quote'):
                    q = self._broker.get_quote(symbol)
                    if q and hasattr(q, 'bid') and hasattr(q, 'ask'):
                        return float((q.bid + q.ask) / 2)
            except Exception:
                pass

        # Priority 3: Fallback price if set
        if hasattr(self, 'fallback_price') and self.fallback_price is not None:
            return float(self.fallback_price)
        
        return None

    def buy(self, symbol: str, amount_usd: float, reason: str = "") -> Optional[PaperTrade]:
        # Basic validation
        if amount_usd <= 0:
            return None
        
        # Enforce minimum order notional
        if amount_usd < self.min_order_value:
            return None

        # Validate against portfolio constraints
        if not self.validate_order(symbol, amount_usd):
            return None
        
        # Get market price
        mid = self.get_mid(symbol)
        if mid is None or mid <= 0:
            return None
        
        # Apply size-aware slippage then basic rounding to $0.01 by default
        size_factor = float(amount_usd) / float(self.initial_capital) if self.initial_capital else 0.0
        eff_slippage = self.slippage * (1.0 + size_factor)
        price = self.round_to_increment(mid * (1 + eff_slippage), symbol)
        
        # Calculate allowable spend based on portfolio constraints
        base_equity = self.initial_capital
        max_pos_usd = base_equity * self.max_position_pct
        current_exposure = sum(
            pos.quantity * (pos.current_price if pos.current_price > 0 else pos.entry_price)
            for pos in self.positions.values()
        )
        max_exposure_usd = base_equity * self.max_exposure
        min_cash_abs = self.min_cash_reserve
        min_cash_pct = self.config.get('min_cash_pct', None)
        required_min_cash = max(min_cash_abs or 0.0, (base_equity * float(min_cash_pct)) if min_cash_pct is not None else 0.0)
        spend_limit = min(amount_usd, max_pos_usd, max(0.0, max_exposure_usd - current_exposure), max(0.0, self.cash - required_min_cash))
        if spend_limit <= 0:
            return None
        # Calculate quantity from allowed spend; commission tracked separately
        commission = spend_limit * self.commission
        qty = spend_limit / price if price > 0 else 0.0
        
        # Enforce product rules (rounding and minimums)
        try:
            qty, price = self.enforce_product_rules(symbol, qty, price)
        except ValueError as e:
            # Order violates product rules
            import logging
            logging.getLogger(__name__).warning(f"Buy order rejected: {e}")
            return None
        
        # Update cash: deduct executed notional only (commission tracked separately)
        self.cash -= spend_limit
        
        # Update or create position
        pos = self.positions.get(symbol)
        if not pos:
            pos = PaperPosition(symbol=symbol, quantity=0.0, entry_price=0.0, current_price=price)
            self.positions[symbol] = pos
        
        # Calculate weighted average entry price
        total_qty = pos.quantity + qty
        if total_qty > 0:
            pos.entry_price = (pos.quantity * pos.entry_price + qty * price) / total_qty
        pos.quantity = total_qty
        pos.current_price = price
        # Track accumulated commission on position
        pos.total_commission += commission
        
        # Record trade
        # Limit trade quantity precision for readability/consistency (<= 8 dp)
        qty_display = Decimal(str(round(Decimal(str(qty)), 8))) if qty > 0 else Decimal("0")

        tr = PaperTrade(
            timestamp=datetime.utcnow(), 
            symbol=symbol, 
            side="buy", 
            quantity=qty_display, 
            price=price, 
            value=float(spend_limit),  # executed notional
            commission=commission, 
            reason=reason
        )
        self.trades.append(tr)
        
        # Log to event store
        self._log_trade(tr)
        self._log_metrics()  # Update metrics after trade
        
        return tr

    def sell(self, symbol: str, qty: Optional[float] = None, value: Optional[float] = None, reason: str = "") -> Optional[PaperTrade]:
        # Check if position exists
        pos = self.positions.get(symbol)
        if not pos or pos.quantity <= 0:
            return None
        
        # Default to selling entire position. If a numeric argument is provided,
        # interpret it as quantity when <= position size, otherwise USD notional.
        desired_value = None
        explicit_qty: Optional[float] = None
        if value is not None and value > 0:
            explicit_qty = float(value)  # will be treated as USD notional later
        elif qty is not None and qty > 0:
            explicit_qty = float(qty)
        
        # Get market price
        mid = self.get_mid(symbol)
        if mid is None or mid <= 0:
            return None
        
        # Apply slippage (negative for sells) then basic rounding
        price = self.round_to_increment(mid * (1 - self.slippage), symbol)
        
        # Decide sell quantity
        if explicit_qty is not None:
            # If explicit value provided larger than position quantity, treat as notional
            if value is not None:
                qty = explicit_qty / price
            elif explicit_qty <= pos.quantity + 1e-9:
                qty = explicit_qty
            else:
                qty = explicit_qty / price
        else:
            qty = pos.quantity

        # Enforce product rules (rounding)
        try:
            qty, price = self.enforce_product_rules(symbol, qty, price)
        except ValueError as e:
            # Order violates product rules
            import logging
            logging.getLogger(__name__).warning(f"Sell order rejected: {e}")
            return None
        
        # Ensure we don't sell more than we have
        qty = min(qty, pos.quantity)
        
        # Calculate proceeds and P&L
        value = qty * price
        commission = value * self.commission
        # Attribute portion of accumulated buy commission to this sale
        buy_commission_share = 0.0
        if pos.quantity > 0:
            buy_commission_share = (pos.total_commission * float(qty) / pos.quantity)
            pos.total_commission -= buy_commission_share
        pnl = (price - pos.entry_price) * qty - commission - buy_commission_share
        
        # Update cash (add proceeds minus commission)
        self.cash += (value - commission)
        
        # Update position
        pos.quantity -= qty
        pos.current_price = price
        
        # Remove position if fully closed
        if pos.quantity <= 1e-9:
            del self.positions[symbol]
        
        # Record trade
        qty_display = Decimal(str(round(Decimal(str(qty)), 8))) if qty > 0 else Decimal("0")
        tr = PaperTrade(
            timestamp=datetime.utcnow(), 
            symbol=symbol, 
            side="sell", 
            quantity=qty_display, 
            price=price, 
            value=value, 
            commission=commission, 
            pnl=pnl, 
            reason=reason
        )
        self.trades.append(tr)
        
        # Log to event store
        self._log_trade(tr)
        self._log_metrics()  # Update metrics after trade
        
        return tr

    def equity(self) -> float:
        eq = self.cash
        for sym, pos in list(self.positions.items()):
            mid = self.get_mid(sym)
            if mid is not None:
                pos.current_price = mid
            eq += pos.quantity * pos.current_price
        return eq

    def snapshot(self) -> None:
        """Take a snapshot of current state (positions and metrics)."""
        self._log_positions()
        self._log_metrics()
    
    def get_events(self, limit: int = 50, types: Optional[List[str]] = None) -> List[Dict]:
        """Retrieve recent events from the event store."""
        return self._event_store.tail(self.bot_id, limit=limit, types=types)
    
    def disconnect(self) -> None:
        try:
            # Final snapshot before disconnecting
            self.snapshot()
            self._broker.disconnect()
        except Exception:
            pass
