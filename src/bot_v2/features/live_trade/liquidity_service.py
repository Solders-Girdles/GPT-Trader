"""
Liquidity Service for Production Trading.

Analyzes order book depth, liquidity conditions, and market impact
to optimize order execution and sizing decisions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.depth_analyzer import DepthAnalyzer
from bot_v2.features.live_trade.impact_estimator import ImpactEstimator
from bot_v2.features.live_trade.liquidity_metrics_tracker import MetricsTracker
from bot_v2.features.live_trade.liquidity_models import (
    DepthAnalysis,
    ImpactEstimate,
    LiquidityCondition,
)
from bot_v2.features.live_trade.liquidity_scorer import LiquidityScorer

logger = logging.getLogger(__name__)


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
        metrics_tracker: MetricsTracker | None = None,
        depth_analyzer: DepthAnalyzer | None = None,
        impact_estimator: ImpactEstimator | None = None,
        liquidity_scorer: LiquidityScorer | None = None,
    ) -> None:
        self.max_impact_bps = max_impact_bps
        self.depth_levels = depth_analysis_levels

        # Service dependencies (injected or default)
        self._metrics_tracker = metrics_tracker or MetricsTracker(
            window_minutes=volume_window_minutes
        )
        self._depth_analyzer = depth_analyzer or DepthAnalyzer()
        self._impact_estimator = impact_estimator or ImpactEstimator(max_impact_bps=max_impact_bps)
        self._liquidity_scorer = liquidity_scorer or LiquidityScorer()
        self._latest_analysis: dict[str, DepthAnalysis] = {}

        logger.info(f"LiquidityService initialized - max impact: {max_impact_bps}bps")

    def update_trade_data(self, symbol: str, price: Decimal, size: Decimal) -> None:
        """Update with new trade data."""
        self._metrics_tracker.add_trade(symbol, price, size)

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

        # Delegate depth analysis to DepthAnalyzer
        depth_data = self._depth_analyzer.analyze_depth(bids, asks)

        if not depth_data:
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

        # Liquidity scoring (0-100)
        liquidity_score = self._liquidity_scorer.calculate_composite_score(
            spread_bps=depth_data.spread_bps,
            depth_usd_1=depth_data.depth_usd_1,
            depth_usd_5=depth_data.depth_usd_5,
            depth_imbalance=depth_data.depth_imbalance,
            mid_price=depth_data.mid_price,
        )
        condition = self._liquidity_scorer.determine_condition(liquidity_score)

        # Update spread metrics
        self._metrics_tracker.add_spread(symbol, depth_data.spread_bps, timestamp)

        analysis = DepthAnalysis(
            symbol=symbol,
            timestamp=timestamp,
            bid_price=depth_data.best_bid,
            ask_price=depth_data.best_ask,
            bid_size=depth_data.bid_size,
            ask_size=depth_data.ask_size,
            spread=depth_data.spread,
            spread_bps=depth_data.spread_bps,
            depth_usd_1=depth_data.depth_usd_1,
            depth_usd_5=depth_data.depth_usd_5,
            depth_usd_10=depth_data.depth_usd_10,
            bid_ask_ratio=depth_data.bid_ask_ratio,
            depth_imbalance=depth_data.depth_imbalance,
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
            return self._impact_estimator.estimate_conservative(symbol, side, quantity)

        # Delegate to ImpactEstimator
        volume_metrics = self._metrics_tracker.get_volume_metrics(symbol)
        return self._impact_estimator.estimate(symbol, side, quantity, analysis, volume_metrics)

    def get_liquidity_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Get current liquidity snapshot for symbol."""
        analysis = self._latest_analysis.get(symbol)
        if not analysis:
            return None

        volume_data = self._metrics_tracker.get_volume_metrics(symbol)
        spread_data = self._metrics_tracker.get_spread_metrics(symbol)

        return {**analysis.to_dict(), **volume_data, **spread_data}


async def create_liquidity_service(**kwargs: Any) -> LiquidityService:
    """Create and initialize liquidity service."""
    service = LiquidityService(**kwargs)
    logger.info("LiquidityService created and ready")
    return service
