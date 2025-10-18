"""
Liquidity Service for Production Trading.

Analyzes order book depth, liquidity conditions, and market impact
to optimize order execution and sizing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="liquidity_service")


class LiquidityCondition(Enum):
    """Market liquidity conditions."""

    EXCELLENT = "excellent"  # Deep, tight markets
    GOOD = "good"  # Normal liquidity
    FAIR = "fair"  # Moderate liquidity concerns
    POOR = "poor"  # Shallow, wide markets
    CRITICAL = "critical"  # Very poor liquidity


@dataclass
class OrderBookLevel:
    """Single order book level."""

    price: Decimal
    size: Decimal
    cumulative_size: Decimal = field(init=False)

    def __post_init__(self) -> None:
        self.cumulative_size = self.size


@dataclass
class DepthAnalysis:
    """Order book depth analysis."""

    symbol: str
    timestamp: datetime

    # Level 1 data
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    spread: Decimal
    spread_bps: Decimal

    # Depth metrics
    depth_usd_1: Decimal  # Depth within 1% of mid
    depth_usd_5: Decimal  # Depth within 5% of mid
    depth_usd_10: Decimal  # Depth within 10% of mid

    # Imbalance metrics
    bid_ask_ratio: Decimal  # Bid size / Ask size
    depth_imbalance: Decimal  # (Bid depth - Ask depth) / Total depth

    # Liquidity scoring
    liquidity_score: Decimal  # 0-100 composite score
    condition: LiquidityCondition

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid_price": float(self.bid_price),
            "ask_price": float(self.ask_price),
            "bid_size": float(self.bid_size),
            "ask_size": float(self.ask_size),
            "spread": float(self.spread),
            "spread_bps": float(self.spread_bps),
            "depth_usd_1": float(self.depth_usd_1),
            "depth_usd_5": float(self.depth_usd_5),
            "depth_usd_10": float(self.depth_usd_10),
            "bid_ask_ratio": float(self.bid_ask_ratio),
            "depth_imbalance": float(self.depth_imbalance),
            "liquidity_score": float(self.liquidity_score),
            "condition": self.condition.value,
        }


@dataclass
class ImpactEstimate:
    """Market impact estimate for order execution."""

    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    estimated_impact_bps: Decimal
    estimated_avg_price: Decimal
    max_impact_price: Decimal
    slippage_cost: Decimal

    # Execution recommendation
    recommended_slicing: bool
    max_slice_size: Decimal | None
    use_post_only: bool

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "estimated_impact_bps": float(self.estimated_impact_bps),
            "estimated_avg_price": float(self.estimated_avg_price),
            "max_impact_price": float(self.max_impact_price),
            "slippage_cost": float(self.slippage_cost),
            "recommended_slicing": self.recommended_slicing,
            "max_slice_size": float(self.max_slice_size) if self.max_slice_size else None,
            "use_post_only": self.use_post_only,
        }


class LiquidityMetrics:
    """
    Rolling liquidity metrics calculator.

    Maintains rolling windows of volume, price impact, and depth
    for liquidity analysis.
    """

    def __init__(self, window_minutes: int = 15) -> None:
        self.window_duration = timedelta(minutes=window_minutes)
        self._volume_data: list[tuple[datetime, Decimal]] = []  # (timestamp, volume)
        self._spread_data: list[tuple[datetime, Decimal]] = []  # (timestamp, spread_bps)
        self._trade_data: list[tuple[datetime, Decimal, Decimal]] = []  # (timestamp, price, size)

    def add_trade(self, price: Decimal, size: Decimal, timestamp: datetime | None = None) -> None:
        """Add trade data for volume calculation."""
        if timestamp is None:
            timestamp = datetime.now()

        notional = price * size
        self._volume_data.append((timestamp, notional))
        self._trade_data.append((timestamp, price, size))

        # Clean old data
        self._clean_old_data()

    def add_spread(self, spread_bps: Decimal, timestamp: datetime | None = None) -> None:
        """Add spread data."""
        if timestamp is None:
            timestamp = datetime.now()

        self._spread_data.append((timestamp, spread_bps))
        self._clean_old_data()

    def get_volume_metrics(self) -> dict[str, Decimal | int]:
        """Calculate volume metrics."""
        if not self._volume_data:
            return {
                "volume_1m": Decimal("0"),
                "volume_5m": Decimal("0"),
                "volume_15m": Decimal("0"),
                "trade_count": 0,
                "avg_trade_size": Decimal("0"),
            }

        now = datetime.now()

        # Calculate volumes for different windows
        volumes: dict[str, Decimal] = {
            "1m": Decimal("0"),
            "5m": Decimal("0"),
            "15m": Decimal("0"),
        }
        windows = {"1m": 1, "5m": 5, "15m": 15}

        for window_name, minutes in windows.items():
            cutoff = now - timedelta(minutes=minutes)
            window_volume = sum(
                (volume for timestamp, volume in self._volume_data if timestamp >= cutoff),
                Decimal("0"),
            )
            volumes[window_name] = window_volume

        # Trade metrics
        trades_15m = [
            (ts, size) for ts, price, size in self._trade_data if ts >= now - timedelta(minutes=15)
        ]

        trade_count = len(trades_15m)
        avg_trade_size = Decimal("0")
        if trade_count > 0:
            total_size = sum((size for _, size in trades_15m), Decimal("0"))
            avg_trade_size = total_size / Decimal(trade_count)

        return {
            "volume_1m": volumes["1m"],
            "volume_5m": volumes["5m"],
            "volume_15m": volumes["15m"],
            "trade_count": trade_count,
            "avg_trade_size": avg_trade_size,
        }

    def get_spread_metrics(self) -> dict[str, Decimal]:
        """Calculate spread metrics."""
        if not self._spread_data:
            return {
                "avg_spread_bps": Decimal("0"),
                "min_spread_bps": Decimal("0"),
                "max_spread_bps": Decimal("0"),
            }

        # Get spreads from last 5 minutes
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)

        recent_spreads = [spread for timestamp, spread in self._spread_data if timestamp >= cutoff]

        if not recent_spreads:
            return {
                "avg_spread_bps": Decimal("0"),
                "min_spread_bps": Decimal("0"),
                "max_spread_bps": Decimal("0"),
            }

        total_spread = sum(recent_spreads, Decimal("0"))
        count = Decimal(len(recent_spreads))
        return {
            "avg_spread_bps": total_spread / count if count > 0 else Decimal("0"),
            "min_spread_bps": min(recent_spreads),
            "max_spread_bps": max(recent_spreads),
        }

    def _clean_old_data(self) -> None:
        """Remove data older than window duration."""
        cutoff = datetime.now() - self.window_duration

        self._volume_data = [(ts, vol) for ts, vol in self._volume_data if ts >= cutoff]
        self._spread_data = [(ts, spread) for ts, spread in self._spread_data if ts >= cutoff]
        self._trade_data = [
            (ts, price, size) for ts, price, size in self._trade_data if ts >= cutoff
        ]


class LiquidityService:
    """
    Production liquidity analysis service.

    Analyzes order book depth, calculates market impact,
    and provides execution recommendations.
    """

    def __init__(
        self,
        max_impact_bps: Decimal = Decimal("50"),  # 5bps max impact
        depth_analysis_levels: int = 20,
        volume_window_minutes: int = 15,
    ) -> None:
        self.max_impact_bps = max_impact_bps
        self.depth_levels = depth_analysis_levels

        # Per-symbol metrics
        self._symbol_metrics: dict[str, LiquidityMetrics] = {}
        self._latest_analysis: dict[str, DepthAnalysis] = {}

        logger.info(f"LiquidityService initialized - max impact: {max_impact_bps}bps")

    def _get_metrics(self, symbol: str) -> LiquidityMetrics:
        """Get or create metrics tracker for symbol."""
        if symbol not in self._symbol_metrics:
            self._symbol_metrics[symbol] = LiquidityMetrics(window_minutes=15)
        return self._symbol_metrics[symbol]

    def update_trade_data(self, symbol: str, price: Decimal, size: Decimal) -> None:
        """Update with new trade data."""
        metrics = self._get_metrics(symbol)
        metrics.add_trade(price, size)

    def analyze_order_book(
        self,
        symbol: str,
        bids: list[tuple[Decimal, Decimal]],  # [(price, size), ...]
        asks: list[tuple[Decimal, Decimal]],
        timestamp: datetime | None = None,
    ) -> DepthAnalysis:
        """
        Analyze order book depth and liquidity conditions.

        Args:
            symbol: Trading symbol
            bids: Bid levels [(price, size), ...]
            asks: Ask levels [(price, size), ...]
            timestamp: Analysis timestamp

        Returns:
            Depth analysis result
        """
        if timestamp is None:
            timestamp = datetime.now()

        if not bids or not asks:
            # Return empty analysis for missing data
            return DepthAnalysis(
                symbol=symbol,
                timestamp=timestamp,
                bid_price=Decimal("0"),
                ask_price=Decimal("0"),
                bid_size=Decimal("0"),
                ask_size=Decimal("0"),
                spread=Decimal("0"),
                spread_bps=Decimal("10000"),  # 100% spread indicates no data
                depth_usd_1=Decimal("0"),
                depth_usd_5=Decimal("0"),
                depth_usd_10=Decimal("0"),
                bid_ask_ratio=Decimal("1"),
                depth_imbalance=Decimal("0"),
                liquidity_score=Decimal("0"),
                condition=LiquidityCondition.CRITICAL,
            )

        # Level 1 data
        best_bid, bid_size = bids[0]
        best_ask, ask_size = asks[0]

        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else Decimal("10000")

        # Calculate depth at various levels
        depth_1pct = mid_price * Decimal("0.01")
        depth_5pct = mid_price * Decimal("0.05")
        depth_10pct = mid_price * Decimal("0.10")

        bid_depth_1 = self._calculate_depth(bids, best_bid - depth_1pct, best_bid)
        ask_depth_1 = self._calculate_depth(asks, best_ask, best_ask + depth_1pct)
        depth_usd_1 = (bid_depth_1 + ask_depth_1) * mid_price

        bid_depth_5 = self._calculate_depth(bids, best_bid - depth_5pct, best_bid)
        ask_depth_5 = self._calculate_depth(asks, best_ask, best_ask + depth_5pct)
        depth_usd_5 = (bid_depth_5 + ask_depth_5) * mid_price

        bid_depth_10 = self._calculate_depth(bids, best_bid - depth_10pct, best_bid)
        ask_depth_10 = self._calculate_depth(asks, best_ask, best_ask + depth_10pct)
        depth_usd_10 = (bid_depth_10 + ask_depth_10) * mid_price

        # Imbalance metrics
        bid_size + ask_size
        bid_ask_ratio = bid_size / ask_size if ask_size > 0 else Decimal("999")

        total_depth_5 = bid_depth_5 + ask_depth_5
        depth_imbalance = (
            ((bid_depth_5 - ask_depth_5) / total_depth_5) if total_depth_5 > 0 else Decimal("0")
        )

        # Liquidity scoring (0-100)
        score_components = {
            "spread": self._score_spread(spread_bps),
            "depth_1": self._score_depth(depth_usd_1, mid_price),
            "depth_5": self._score_depth(depth_usd_5, mid_price),
            "imbalance": self._score_imbalance(abs(depth_imbalance)),
        }

        liquidity_score = sum(score_components.values(), Decimal("0")) / Decimal(
            len(score_components)
        )
        condition = self._determine_condition(liquidity_score)

        # Update spread metrics
        metrics = self._get_metrics(symbol)
        metrics.add_spread(spread_bps, timestamp)

        analysis = DepthAnalysis(
            symbol=symbol,
            timestamp=timestamp,
            bid_price=best_bid,
            ask_price=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=spread,
            spread_bps=spread_bps,
            depth_usd_1=depth_usd_1,
            depth_usd_5=depth_usd_5,
            depth_usd_10=depth_usd_10,
            bid_ask_ratio=bid_ask_ratio,
            depth_imbalance=depth_imbalance,
            liquidity_score=liquidity_score,
            condition=condition,
        )

        # Cache latest analysis
        self._latest_analysis[symbol] = analysis

        return analysis

    def estimate_market_impact(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        book_data: (
            tuple[list[tuple[Decimal, Decimal]], list[tuple[Decimal, Decimal]]] | None
        ) = None,
    ) -> ImpactEstimate:
        """
        Estimate market impact for order execution.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            book_data: Optional (bids, asks) tuple

        Returns:
            Impact estimate with execution recommendations
        """
        # Get latest depth analysis
        analysis = self._latest_analysis.get(symbol)
        if not analysis:
            logger.warning(f"No depth analysis available for {symbol}")
            # Return conservative estimate
            return ImpactEstimate(
                symbol=symbol,
                side=side,
                quantity=quantity,
                estimated_impact_bps=Decimal("100"),  # 10bps conservative
                estimated_avg_price=Decimal("0"),
                max_impact_price=Decimal("0"),
                slippage_cost=Decimal("0"),
                recommended_slicing=True,
                max_slice_size=quantity / 10,
                use_post_only=True,
            )

        # Use refined impact model
        if book_data:
            bids, asks = book_data
        else:
            # Approximate from analysis
            pass

        # Calculate impact using square-root model with depth adjustment
        mid_price = (analysis.bid_price + analysis.ask_price) / 2
        notional = quantity * mid_price

        # Base impact from square-root model
        volume_metrics = self._get_metrics(symbol).get_volume_metrics()
        volume_15m = max(volume_metrics["volume_15m"], Decimal("1000"))  # Min $1k volume

        base_impact_bps = (notional / volume_15m).sqrt() * 100  # Convert to bps

        # Adjust for depth
        relevant_depth = analysis.depth_usd_5
        if notional > relevant_depth:
            depth_multiplier = (notional / relevant_depth).sqrt()
            base_impact_bps *= depth_multiplier

        # Adjust for spread and conditions
        spread_multiplier = 1 + (analysis.spread_bps / 1000)  # Add fraction of spread
        condition_multiplier = {
            LiquidityCondition.EXCELLENT: Decimal("0.5"),
            LiquidityCondition.GOOD: Decimal("1.0"),
            LiquidityCondition.FAIR: Decimal("1.5"),
            LiquidityCondition.POOR: Decimal("2.0"),
            LiquidityCondition.CRITICAL: Decimal("3.0"),
        }[analysis.condition]

        final_impact_bps = base_impact_bps * spread_multiplier * condition_multiplier

        # Calculate prices and costs
        if side == "buy":
            estimated_avg_price = mid_price * (1 + final_impact_bps / 10000)
            max_impact_price = mid_price * (1 + final_impact_bps * Decimal("1.5") / 10000)
        else:  # sell
            estimated_avg_price = mid_price * (1 - final_impact_bps / 10000)
            max_impact_price = mid_price * (1 - final_impact_bps * Decimal("1.5") / 10000)

        slippage_cost = abs(estimated_avg_price - mid_price) * quantity

        # Execution recommendations
        recommended_slicing = final_impact_bps > self.max_impact_bps
        max_slice_size = None
        if recommended_slicing:
            # Size slices to keep impact under threshold
            target_impact = self.max_impact_bps
            target_notional = (target_impact / base_impact_bps) ** 2 * notional
            max_slice_size = target_notional / mid_price

        use_post_only = (
            analysis.condition
            in [LiquidityCondition.FAIR, LiquidityCondition.POOR, LiquidityCondition.CRITICAL]
            or final_impact_bps > self.max_impact_bps / 2
        )

        return ImpactEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            estimated_impact_bps=final_impact_bps,
            estimated_avg_price=estimated_avg_price,
            max_impact_price=max_impact_price,
            slippage_cost=slippage_cost,
            recommended_slicing=recommended_slicing,
            max_slice_size=max_slice_size,
            use_post_only=use_post_only,
        )

    def get_liquidity_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Get current liquidity snapshot for symbol."""
        analysis = self._latest_analysis.get(symbol)
        if not analysis:
            return None

        metrics = self._get_metrics(symbol)
        volume_data = metrics.get_volume_metrics()
        spread_data = metrics.get_spread_metrics()

        return {**analysis.to_dict(), **volume_data, **spread_data}

    def _calculate_depth(
        self, levels: list[tuple[Decimal, Decimal]], min_price: Decimal, max_price: Decimal
    ) -> Decimal:
        """Calculate total size within price range."""
        total_size = Decimal("0")
        for price, size in levels:
            if min_price <= price <= max_price:
                total_size += size
            elif (min_price <= price <= max_price) is False:
                break  # Levels are sorted, so we can break early
        return total_size

    def _score_spread(self, spread_bps: Decimal) -> Decimal:
        """Score spread component (0-100)."""
        if spread_bps <= 1:
            return Decimal("100")
        elif spread_bps <= 5:
            return Decimal("80")
        elif spread_bps <= 10:
            return Decimal("60")
        elif spread_bps <= 20:
            return Decimal("40")
        elif spread_bps <= 50:
            return Decimal("20")
        else:
            return Decimal("0")

    def _score_depth(self, depth_usd: Decimal, mid_price: Decimal) -> Decimal:
        """Score depth component (0-100)."""
        # Score based on depth relative to typical trade sizes
        depth_score = min(depth_usd / Decimal("10000"), Decimal("1")) * Decimal("100")
        return depth_score

    def _score_imbalance(self, imbalance: Decimal) -> Decimal:
        """Score imbalance component (0-100, lower imbalance = better)."""
        return max(Decimal("0"), Decimal("100") - imbalance * Decimal("200"))

    def _determine_condition(self, score: Decimal) -> LiquidityCondition:
        """Determine liquidity condition from score."""
        if score >= Decimal("80"):
            return LiquidityCondition.EXCELLENT
        elif score >= Decimal("60"):
            return LiquidityCondition.GOOD
        elif score >= Decimal("40"):
            return LiquidityCondition.FAIR
        elif score >= Decimal("20"):
            return LiquidityCondition.POOR
        else:
            return LiquidityCondition.CRITICAL


async def create_liquidity_service(**kwargs: Any) -> LiquidityService:
    """Create and initialize liquidity service."""
    service = LiquidityService(**kwargs)
    logger.info("LiquidityService created and ready")
    return service
