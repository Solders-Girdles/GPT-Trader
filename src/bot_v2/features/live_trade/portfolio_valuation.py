"""
Portfolio Valuation Service for Production Trading.

Provides real-time portfolio equity calculation, mark-to-market valuation,
and unified PnL tracking with funding accruals for production readiness.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import logging

from .pnl_tracker import PnLTracker, PositionState
from ..brokerages.core.interfaces import Balance, Position

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    """
    Complete portfolio valuation snapshot.
    
    Captures equity, positions, PnL breakdown, and metadata
    for a specific point in time.
    """
    timestamp: datetime
    total_equity_usd: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    
    # PnL breakdown
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    funding_pnl: Decimal
    fees_paid: Decimal
    
    # Position details
    positions: Dict[str, Dict] = field(default_factory=dict)
    
    # Risk metrics
    leverage: Decimal = Decimal('0')
    margin_used: Decimal = Decimal('0')
    margin_available: Decimal = Decimal('0')
    
    # Data quality
    stale_marks: Set[str] = field(default_factory=set)
    missing_positions: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_equity_usd': float(self.total_equity_usd),
            'cash_balance': float(self.cash_balance),
            'positions_value': float(self.positions_value),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'funding_pnl': float(self.funding_pnl),
            'fees_paid': float(self.fees_paid),
            'positions': {
                symbol: {k: float(v) if isinstance(v, Decimal) else v for k, v in pos.items()}
                for symbol, pos in self.positions.items()
            },
            'leverage': float(self.leverage),
            'margin_used': float(self.margin_used),
            'margin_available': float(self.margin_available),
            'stale_marks': list(self.stale_marks),
            'missing_positions': list(self.missing_positions)
        }


class MarkDataSource:
    """
    Mark price data source with staleness detection.
    
    Provides mark prices from multiple sources (WS, REST fallback)
    with staleness guards and quality metrics.
    """
    
    def __init__(self, staleness_threshold_seconds: int = 30):
        self.staleness_threshold = timedelta(seconds=staleness_threshold_seconds)
        self._mark_data: Dict[str, Tuple[Decimal, datetime]] = {}
        
    def update_mark(self, symbol: str, price: Decimal, timestamp: Optional[datetime] = None):
        """Update mark price for symbol."""
        if timestamp is None:
            timestamp = datetime.now()
        self._mark_data[symbol] = (price, timestamp)
        
    def get_mark(self, symbol: str) -> Optional[Tuple[Decimal, bool]]:
        """
        Get mark price and staleness status.
        
        Returns:
            Tuple of (price, is_stale) or None if no data
        """
        if symbol not in self._mark_data:
            return None
        
        price, timestamp = self._mark_data[symbol]
        is_stale = (datetime.now() - timestamp) > self.staleness_threshold
        return price, is_stale
    
    def get_all_marks(self) -> Dict[str, Decimal]:
        """Get all current marks (excluding stale)."""
        marks = {}
        for symbol, (price, timestamp) in self._mark_data.items():
            if (datetime.now() - timestamp) <= self.staleness_threshold:
                marks[symbol] = price
        return marks
    
    def get_stale_symbols(self) -> Set[str]:
        """Get symbols with stale mark data."""
        stale = set()
        now = datetime.now()
        for symbol, (_, timestamp) in self._mark_data.items():
            if (now - timestamp) > self.staleness_threshold:
                stale.add(symbol)
        return stale


class PortfolioValuationService:
    """
    Production portfolio valuation service.
    
    Aggregates account balances, positions, and marks to compute
    unified equity in USD with realized/unrealized PnL tracking.
    """
    
    def __init__(
        self,
        pnl_tracker: Optional[PnLTracker] = None,
        mark_staleness_seconds: int = 30,
        snapshot_interval_minutes: int = 5
    ):
        self.pnl_tracker = pnl_tracker or PnLTracker()
        self.mark_source = MarkDataSource(staleness_threshold_seconds=mark_staleness_seconds)
        
        # Snapshot management
        self.snapshot_interval = timedelta(minutes=snapshot_interval_minutes)
        self._last_snapshot_time: Optional[datetime] = None
        self._snapshots: List[PortfolioSnapshot] = []
        
        # Account data cache
        self._cached_balances: Dict[str, Balance] = {}
        self._cached_positions: Dict[str, Position] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=30)  # 30s cache
        
        logger.info(f"PortfolioValuationService initialized - mark staleness: {mark_staleness_seconds}s")
    
    def update_account_data(
        self,
        balances: List[Balance],
        positions: List[Position]
    ):
        """Update cached account data."""
        self._cached_balances = {b.currency: b for b in balances}
        self._cached_positions = {p.symbol: p for p in positions}
        self._cache_timestamp = datetime.now()
        
        logger.debug(f"Account data updated - {len(balances)} balances, {len(positions)} positions")
    
    def update_mark_prices(self, mark_prices: Dict[str, Decimal]):
        """Update mark prices from market data."""
        for symbol, price in mark_prices.items():
            self.mark_source.update_mark(symbol, price)
        
        # Update PnL tracker
        self.pnl_tracker.update_marks(mark_prices)
    
    def update_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        fees: Decimal = Decimal('0'),
        is_reduce: bool = False
    ) -> Dict[str, Decimal]:
        """Update with new trade execution."""
        # Update PnL tracker
        pnl_result = self.pnl_tracker.update_position(symbol, side, quantity, price, is_reduce)
        
        # Update fee tracking (would integrate with FeesEngine)
        if fees > 0:
            logger.info(f"Trade fees recorded: {fees} for {symbol} {side} {quantity}")
            
        logger.debug(f"Trade updated - {symbol} {side} {quantity} @ {price}, realized PnL: {pnl_result.get('realized_pnl', 0)}")
        
        return pnl_result
    
    def compute_current_valuation(self) -> PortfolioSnapshot:
        """
        Compute current portfolio valuation.
        
        Aggregates balances, positions, and marks to generate
        unified equity calculation with PnL breakdown.
        """
        now = datetime.now()
        
        # Get account data
        if not self._is_cache_valid():
            logger.warning("Account data cache is stale or missing")
        
        balances = self._cached_balances
        positions = self._cached_positions
        
        # Get current marks
        current_marks = self.mark_source.get_all_marks()
        stale_marks = self.mark_source.get_stale_symbols()
        
        # Calculate cash balance (USD/USDC)
        cash_balance = Decimal('0')
        for currency in ['USD', 'USDC', 'USDT']:
            if currency in balances:
                cash_balance += balances[currency].available
        
        # Calculate positions value and update unrealized PnL
        positions_value = Decimal('0')
        position_details = {}
        missing_positions = set()
        
        for symbol, position in positions.items():
            if position.quantity == 0:
                continue
                
            # Get mark price
            mark_data = self.mark_source.get_mark(symbol)
            if not mark_data:
                missing_positions.add(symbol)
                logger.warning(f"No mark price for position {symbol}")
                continue
            
            mark_price, is_stale = mark_data
            if is_stale:
                stale_marks.add(symbol)
            
            # Calculate position value
            notional_value = abs(position.quantity) * mark_price
            positions_value += notional_value
            
            # Get position state from PnL tracker
            pnl_position = self.pnl_tracker.get_or_create_position(symbol)
            pnl_position.update_mark(mark_price)
            
            position_details[symbol] = {
                'side': 'long' if position.quantity > 0 else 'short',
                'quantity': position.quantity,
                'mark_price': mark_price,
                'notional_value': notional_value,
                'unrealized_pnl': pnl_position.unrealized_pnl,
                'realized_pnl': pnl_position.realized_pnl,
                'funding_paid': pnl_position.funding_paid,
                'avg_entry_price': pnl_position.avg_entry_price,
                'is_stale': is_stale
            }
        
        # Get total PnL from tracker
        total_pnl = self.pnl_tracker.get_total_pnl()
        
        # Calculate total equity
        total_equity = cash_balance + total_pnl['total']
        
        # Calculate margin metrics (simplified)
        margin_used = positions_value * Decimal('0.1')  # Assume 10x leverage
        margin_available = max(Decimal('0'), cash_balance - margin_used)
        leverage = positions_value / max(cash_balance, Decimal('1')) if cash_balance > 0 else Decimal('0')
        
        # Create snapshot
        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_equity_usd=total_equity,
            cash_balance=cash_balance,
            positions_value=positions_value,
            realized_pnl=total_pnl['realized'],
            unrealized_pnl=total_pnl['unrealized'],
            funding_pnl=total_pnl['funding'],
            fees_paid=Decimal('0'),  # Would integrate with FeesEngine
            positions=position_details,
            leverage=leverage,
            margin_used=margin_used,
            margin_available=margin_available,
            stale_marks=stale_marks,
            missing_positions=missing_positions
        )
        
        # Store snapshot if interval reached
        if self._should_create_snapshot():
            self._store_snapshot(snapshot)
        
        return snapshot
    
    def get_equity_curve(self, hours_back: int = 24) -> List[Dict]:
        """Get equity curve for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        curve_data = []
        for snapshot in self._snapshots:
            if snapshot.timestamp >= cutoff_time:
                curve_data.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'equity': float(snapshot.total_equity_usd),
                    'realized_pnl': float(snapshot.realized_pnl),
                    'unrealized_pnl': float(snapshot.unrealized_pnl)
                })
        
        return curve_data
    
    def get_daily_metrics(self) -> Dict:
        """Generate daily performance metrics."""
        if not self._snapshots:
            return {}
        
        latest = self._snapshots[-1]
        return self.pnl_tracker.generate_daily_metrics(latest.total_equity_usd)
    
    def _is_cache_valid(self) -> bool:
        """Check if account data cache is valid."""
        if not self._cache_timestamp:
            return False
        return (datetime.now() - self._cache_timestamp) <= self._cache_ttl
    
    def _should_create_snapshot(self) -> bool:
        """Check if we should create a new snapshot."""
        if not self._last_snapshot_time:
            return True
        return (datetime.now() - self._last_snapshot_time) >= self.snapshot_interval
    
    def _store_snapshot(self, snapshot: PortfolioSnapshot):
        """Store snapshot and manage retention."""
        self._snapshots.append(snapshot)
        self._last_snapshot_time = snapshot.timestamp
        
        # Keep last 7 days of snapshots (assuming 5min intervals = 2016 snapshots)
        max_snapshots = 2016
        if len(self._snapshots) > max_snapshots:
            self._snapshots = self._snapshots[-max_snapshots:]
        
        logger.debug(f"Portfolio snapshot stored - equity: {snapshot.total_equity_usd}")


async def create_portfolio_valuation_service(
    pnl_tracker: Optional[PnLTracker] = None,
    **kwargs
) -> PortfolioValuationService:
    """Create and initialize portfolio valuation service."""
    service = PortfolioValuationService(pnl_tracker=pnl_tracker, **kwargs)
    logger.info("PortfolioValuationService created and ready")
    return service