"""
Comprehensive tests for LiquidityService and LiquidityMetrics.

Tests cover:
- LiquidityMetrics rolling window calculations
- Volume and spread metric aggregation
- Order book depth analysis
- Market impact estimation
- Liquidity scoring and condition determination
- Empty order book handling
- Data cleanup and staleness
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.live_trade.liquidity_service import (
    DepthAnalysis,
    ImpactEstimate,
    LiquidityCondition,
    LiquidityMetrics,
    LiquidityService,
    OrderBookLevel,
    create_liquidity_service,
)


# ============================================================================
# Test: OrderBookLevel
# ============================================================================


class TestOrderBookLevel:
    """Test OrderBookLevel dataclass."""

    def test_creates_level_with_cumulative_size(self):
        """Test that cumulative size is initialized."""
        level = OrderBookLevel(price=Decimal("100"), size=Decimal("10"))

        assert level.price == Decimal("100")
        assert level.size == Decimal("10")
        assert level.cumulative_size == Decimal("10")


# ============================================================================
# Test: LiquidityMetrics
# ============================================================================


class TestLiquidityMetrics:
    """Test LiquidityMetrics rolling window calculations."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = LiquidityMetrics(window_minutes=15)

        assert metrics.window_duration == timedelta(minutes=15)
        assert len(metrics._volume_data) == 0
        assert len(metrics._spread_data) == 0
        assert len(metrics._trade_data) == 0

    def test_add_trade_with_timestamp(self):
        """Test adding trade data with explicit timestamp."""
        metrics = LiquidityMetrics()
        # Use recent timestamp to avoid cleanup
        timestamp = datetime.now() - timedelta(minutes=1)

        metrics.add_trade(
            price=Decimal("100"),
            size=Decimal("10"),
            timestamp=timestamp
        )

        assert len(metrics._volume_data) == 1
        assert len(metrics._trade_data) == 1

        ts, volume = metrics._volume_data[0]
        assert ts == timestamp
        assert volume == Decimal("1000")  # price * size

    def test_add_trade_without_timestamp(self):
        """Test adding trade data uses current time."""
        metrics = LiquidityMetrics()

        before = datetime.now()
        metrics.add_trade(price=Decimal("100"), size=Decimal("10"))
        after = datetime.now()

        ts, _ = metrics._volume_data[0]
        assert before <= ts <= after

    def test_add_spread_with_timestamp(self):
        """Test adding spread data."""
        metrics = LiquidityMetrics()
        # Use recent timestamp to avoid cleanup
        timestamp = datetime.now() - timedelta(minutes=1)

        metrics.add_spread(spread_bps=Decimal("5"), timestamp=timestamp)

        assert len(metrics._spread_data) == 1
        ts, spread = metrics._spread_data[0]
        assert ts == timestamp
        assert spread == Decimal("5")

    def test_get_volume_metrics_empty(self):
        """Test volume metrics with no data returns zeros."""
        metrics = LiquidityMetrics()

        result = metrics.get_volume_metrics()

        assert result["volume_1m"] == Decimal("0")
        assert result["volume_5m"] == Decimal("0")
        assert result["volume_15m"] == Decimal("0")
        assert result["trade_count"] == 0
        assert result["avg_trade_size"] == Decimal("0")

    def test_get_volume_metrics_with_recent_trades(self):
        """Test volume metrics calculation with recent trades."""
        metrics = LiquidityMetrics()
        base_time = datetime.now()

        # Add trades at different times
        metrics.add_trade(Decimal("100"), Decimal("10"), base_time)
        metrics.add_trade(Decimal("100"), Decimal("20"), base_time - timedelta(minutes=2))
        metrics.add_trade(Decimal("100"), Decimal("30"), base_time - timedelta(minutes=7))

        result = metrics.get_volume_metrics()

        # 1m window should have 1 trade (1000)
        assert result["volume_1m"] == Decimal("1000")
        # 5m window should have 2 trades (3000)
        assert result["volume_5m"] == Decimal("3000")
        # 15m window should have all 3 trades (6000)
        assert result["volume_15m"] == Decimal("6000")
        assert result["trade_count"] == 3
        assert result["avg_trade_size"] == Decimal("20")  # (10+20+30)/3

    def test_get_spread_metrics_empty(self):
        """Test spread metrics with no data returns zeros."""
        metrics = LiquidityMetrics()

        result = metrics.get_spread_metrics()

        assert result["avg_spread_bps"] == Decimal("0")
        assert result["min_spread_bps"] == Decimal("0")
        assert result["max_spread_bps"] == Decimal("0")

    def test_get_spread_metrics_with_recent_spreads(self):
        """Test spread metrics calculation."""
        metrics = LiquidityMetrics()
        base_time = datetime.now()

        # Add spreads within 5 minute window
        metrics.add_spread(Decimal("5"), base_time)
        metrics.add_spread(Decimal("10"), base_time - timedelta(minutes=2))
        metrics.add_spread(Decimal("15"), base_time - timedelta(minutes=4))

        result = metrics.get_spread_metrics()

        assert result["avg_spread_bps"] == Decimal("10")  # (5+10+15)/3
        assert result["min_spread_bps"] == Decimal("5")
        assert result["max_spread_bps"] == Decimal("15")

    def test_get_spread_metrics_ignores_old_data(self):
        """Test spread metrics ignores data older than 5 minutes."""
        metrics = LiquidityMetrics()
        base_time = datetime.now()

        # Add recent spread
        metrics.add_spread(Decimal("10"), base_time)
        # Add old spread (should be ignored)
        metrics.add_spread(Decimal("100"), base_time - timedelta(minutes=10))

        result = metrics.get_spread_metrics()

        assert result["avg_spread_bps"] == Decimal("10")
        assert result["min_spread_bps"] == Decimal("10")
        assert result["max_spread_bps"] == Decimal("10")

    def test_data_cleanup(self):
        """Test that old data is cleaned up."""
        metrics = LiquidityMetrics(window_minutes=5)
        base_time = datetime.now()

        # Add old data
        old_time = base_time - timedelta(minutes=10)
        metrics.add_trade(Decimal("100"), Decimal("10"), old_time)
        metrics.add_spread(Decimal("5"), old_time)

        # Add recent data
        metrics.add_trade(Decimal("100"), Decimal("10"), base_time)

        # Old data should be cleaned
        assert len(metrics._volume_data) == 1
        assert len(metrics._spread_data) == 0
        assert len(metrics._trade_data) == 1


# ============================================================================
# Test: LiquidityService Initialization
# ============================================================================


class TestLiquidityServiceInitialization:
    """Test LiquidityService initialization."""

    def test_default_initialization(self):
        """Test service initialization with defaults."""
        service = LiquidityService()

        assert service.max_impact_bps == Decimal("50")
        assert service.depth_levels == 20
        assert isinstance(service._symbol_metrics, dict)
        assert isinstance(service._latest_analysis, dict)

    def test_custom_initialization(self):
        """Test service initialization with custom parameters."""
        service = LiquidityService(
            max_impact_bps=Decimal("25"),
            depth_analysis_levels=30,
            volume_window_minutes=10,
        )

        assert service.max_impact_bps == Decimal("25")
        assert service.depth_levels == 30

    @pytest.mark.asyncio
    async def test_create_liquidity_service(self):
        """Test async factory function."""
        service = await create_liquidity_service(max_impact_bps=Decimal("30"))

        assert isinstance(service, LiquidityService)
        assert service.max_impact_bps == Decimal("30")


# ============================================================================
# Test: LiquidityService Trade Updates
# ============================================================================


class TestLiquidityServiceTradeUpdates:
    """Test trade data updates."""

    def test_update_trade_data(self):
        """Test updating trade data creates metrics."""
        service = LiquidityService()

        service.update_trade_data("BTC-USD", Decimal("50000"), Decimal("1.5"))

        assert "BTC-USD" in service._symbol_metrics
        metrics = service._symbol_metrics["BTC-USD"]
        assert len(metrics._trade_data) == 1


# ============================================================================
# Test: Order Book Analysis
# ============================================================================


class TestLiquidityServiceOrderBookAnalysis:
    """Test analyze_order_book method."""

    def test_analyze_empty_order_book(self):
        """Test analysis with empty order book."""
        service = LiquidityService()

        analysis = service.analyze_order_book(
            symbol="BTC-USD",
            bids=[],
            asks=[]
        )

        assert analysis.symbol == "BTC-USD"
        assert analysis.bid_price == Decimal("0")
        assert analysis.ask_price == Decimal("0")
        assert analysis.spread_bps == Decimal("10000")
        assert analysis.condition == LiquidityCondition.CRITICAL
        assert analysis.liquidity_score == Decimal("0")

    def test_analyze_deep_order_book(self):
        """Test analysis with deep, liquid order book."""
        service = LiquidityService()

        # Create deep order book
        bids = [
            (Decimal("50000"), Decimal("10")),  # Best bid
            (Decimal("49950"), Decimal("20")),
            (Decimal("49900"), Decimal("30")),
        ]
        asks = [
            (Decimal("50010"), Decimal("10")),  # Best ask
            (Decimal("50050"), Decimal("20")),
            (Decimal("50100"), Decimal("30")),
        ]

        analysis = service.analyze_order_book(
            symbol="BTC-USD",
            bids=bids,
            asks=asks
        )

        assert analysis.symbol == "BTC-USD"
        assert analysis.bid_price == Decimal("50000")
        assert analysis.ask_price == Decimal("50010")
        assert analysis.spread == Decimal("10")
        assert analysis.bid_size == Decimal("10")
        assert analysis.ask_size == Decimal("10")
        # Spread should be ~2 bps (10/50005 * 10000)
        assert Decimal("1.5") < analysis.spread_bps < Decimal("2.5")
        assert analysis.liquidity_score > Decimal("0")

    def test_analyze_order_book_caches_result(self):
        """Test that analysis is cached."""
        service = LiquidityService()

        bids = [(Decimal("100"), Decimal("10"))]
        asks = [(Decimal("101"), Decimal("10"))]

        analysis = service.analyze_order_book("BTC-USD", bids, asks)

        assert "BTC-USD" in service._latest_analysis
        assert service._latest_analysis["BTC-USD"] == analysis

    def test_analyze_order_book_calculates_depth_metrics(self):
        """Test depth metric calculations."""
        service = LiquidityService()

        mid_price = Decimal("50000")
        bids = [
            (Decimal("50000"), Decimal("10")),
            (Decimal("49500"), Decimal("20")),  # Within 1%
            (Decimal("48000"), Decimal("30")),  # Within 5%, 10%
        ]
        asks = [
            (Decimal("50010"), Decimal("10")),
            (Decimal("50500"), Decimal("20")),  # Within 1%
            (Decimal("52000"), Decimal("30")),  # Within 5%, 10%
        ]

        analysis = service.analyze_order_book("BTC-USD", bids, asks)

        assert analysis.depth_usd_1 > Decimal("0")
        assert analysis.depth_usd_5 > Decimal("0")
        assert analysis.depth_usd_10 > Decimal("0")
        # Depth should increase with wider ranges
        assert analysis.depth_usd_10 >= analysis.depth_usd_5 >= analysis.depth_usd_1

    def test_analyze_order_book_imbalance_metrics(self):
        """Test bid/ask imbalance calculations."""
        service = LiquidityService()

        # Bid-heavy order book
        bids = [(Decimal("100"), Decimal("100"))]
        asks = [(Decimal("101"), Decimal("10"))]

        analysis = service.analyze_order_book("BTC-USD", bids, asks)

        assert analysis.bid_ask_ratio == Decimal("10")  # 100/10
        assert analysis.depth_imbalance > Decimal("0")  # More bid depth

    def test_analyze_order_book_with_timestamp(self):
        """Test analysis with explicit timestamp."""
        service = LiquidityService()
        timestamp = datetime(2025, 1, 1, 12, 0, 0)

        bids = [(Decimal("100"), Decimal("10"))]
        asks = [(Decimal("101"), Decimal("10"))]

        analysis = service.analyze_order_book(
            "BTC-USD", bids, asks, timestamp=timestamp
        )

        assert analysis.timestamp == timestamp


# ============================================================================
# Test: Market Impact Estimation
# ============================================================================


class TestLiquidityServiceMarketImpact:
    """Test estimate_market_impact method."""

    def test_estimate_impact_without_analysis(self):
        """Test impact estimation without prior depth analysis."""
        service = LiquidityService()

        estimate = service.estimate_market_impact(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1.0")
        )

        assert estimate.symbol == "BTC-USD"
        assert estimate.side == "buy"
        assert estimate.quantity == Decimal("1.0")
        assert estimate.estimated_impact_bps == Decimal("100")
        assert estimate.recommended_slicing is True
        assert estimate.use_post_only is True

    def test_estimate_impact_with_analysis(self):
        """Test impact estimation with depth analysis."""
        service = LiquidityService()

        # First analyze order book
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50010"), Decimal("10"))]
        service.analyze_order_book("BTC-USD", bids, asks)

        # Add some volume history
        service.update_trade_data("BTC-USD", Decimal("50000"), Decimal("1"))

        estimate = service.estimate_market_impact(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.5")
        )

        assert estimate.symbol == "BTC-USD"
        assert estimate.estimated_impact_bps > Decimal("0")
        assert estimate.estimated_avg_price > Decimal("0")
        assert estimate.slippage_cost >= Decimal("0")

    def test_estimate_impact_buy_vs_sell(self):
        """Test that buy and sell impacts differ correctly."""
        service = LiquidityService()

        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50010"), Decimal("10"))]
        service.analyze_order_book("BTC-USD", bids, asks)
        service.update_trade_data("BTC-USD", Decimal("50000"), Decimal("1"))

        buy_estimate = service.estimate_market_impact(
            "BTC-USD", "buy", Decimal("0.5")
        )
        sell_estimate = service.estimate_market_impact(
            "BTC-USD", "sell", Decimal("0.5")
        )

        # Buy should have higher avg price, sell should have lower
        mid = Decimal("50005")
        assert buy_estimate.estimated_avg_price > mid
        assert sell_estimate.estimated_avg_price < mid

    def test_estimate_impact_large_order_recommends_slicing(self):
        """Test that large orders trigger slicing recommendation."""
        service = LiquidityService(max_impact_bps=Decimal("10"))

        bids = [(Decimal("50000"), Decimal("1"))]
        asks = [(Decimal("50010"), Decimal("1"))]
        service.analyze_order_book("BTC-USD", bids, asks)
        service.update_trade_data("BTC-USD", Decimal("50000"), Decimal("0.1"))

        estimate = service.estimate_market_impact(
            "BTC-USD", "buy", Decimal("10")  # Large order
        )

        assert estimate.recommended_slicing is True
        assert estimate.max_slice_size is not None
        assert estimate.max_slice_size < Decimal("10")

    def test_estimate_impact_poor_liquidity_uses_post_only(self):
        """Test post-only recommendation for poor liquidity."""
        service = LiquidityService()

        # Create poor liquidity scenario (wide spread, low depth)
        bids = [(Decimal("50000"), Decimal("0.1"))]
        asks = [(Decimal("50500"), Decimal("0.1"))]  # 1% spread
        service.analyze_order_book("BTC-USD", bids, asks)
        service.update_trade_data("BTC-USD", Decimal("50000"), Decimal("0.01"))

        estimate = service.estimate_market_impact(
            "BTC-USD", "buy", Decimal("0.5")
        )

        assert estimate.use_post_only is True


# ============================================================================
# Test: Liquidity Snapshot
# ============================================================================


class TestLiquidityServiceSnapshot:
    """Test get_liquidity_snapshot method."""

    def test_get_snapshot_no_analysis(self):
        """Test snapshot returns None without analysis."""
        service = LiquidityService()

        snapshot = service.get_liquidity_snapshot("BTC-USD")

        assert snapshot is None

    def test_get_snapshot_with_analysis(self):
        """Test snapshot includes all metrics."""
        service = LiquidityService()

        # Create analysis
        bids = [(Decimal("50000"), Decimal("10"))]
        asks = [(Decimal("50010"), Decimal("10"))]
        service.analyze_order_book("BTC-USD", bids, asks)

        # Add trade data
        service.update_trade_data("BTC-USD", Decimal("50000"), Decimal("1"))

        snapshot = service.get_liquidity_snapshot("BTC-USD")

        assert snapshot is not None
        assert "symbol" in snapshot
        assert "liquidity_score" in snapshot
        assert "volume_1m" in snapshot
        assert "avg_spread_bps" in snapshot


# ============================================================================
# Test: Scoring Functions
# ============================================================================


class TestLiquidityServiceScoring:
    """Test internal scoring functions."""

    def test_score_spread_excellent(self):
        """Test spread scoring for tight spreads."""
        service = LiquidityService()

        score = service._score_spread(Decimal("1"))
        assert score == Decimal("100")

    def test_score_spread_poor(self):
        """Test spread scoring for wide spreads."""
        service = LiquidityService()

        score = service._score_spread(Decimal("100"))
        assert score == Decimal("0")

    def test_score_depth(self):
        """Test depth scoring."""
        service = LiquidityService()

        # High depth should score well
        score_high = service._score_depth(Decimal("20000"), Decimal("50000"))
        # Low depth should score poorly
        score_low = service._score_depth(Decimal("100"), Decimal("50000"))

        assert score_high > score_low

    def test_score_imbalance(self):
        """Test imbalance scoring."""
        service = LiquidityService()

        # Balanced book scores high
        score_balanced = service._score_imbalance(Decimal("0"))
        # Imbalanced book scores low
        score_imbalanced = service._score_imbalance(Decimal("0.5"))

        assert score_balanced > score_imbalanced
        assert score_balanced == Decimal("100")

    def test_determine_condition(self):
        """Test liquidity condition determination."""
        service = LiquidityService()

        assert service._determine_condition(Decimal("90")) == LiquidityCondition.EXCELLENT
        assert service._determine_condition(Decimal("70")) == LiquidityCondition.GOOD
        assert service._determine_condition(Decimal("50")) == LiquidityCondition.FAIR
        assert service._determine_condition(Decimal("30")) == LiquidityCondition.POOR
        assert service._determine_condition(Decimal("10")) == LiquidityCondition.CRITICAL


# ============================================================================
# Test: DepthAnalysis to_dict
# ============================================================================


class TestDepthAnalysisSerialization:
    """Test DepthAnalysis serialization."""

    def test_to_dict(self):
        """Test conversion to dict for serialization."""
        analysis = DepthAnalysis(
            symbol="BTC-USD",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            spread=Decimal("10"),
            spread_bps=Decimal("2"),
            depth_usd_1=Decimal("100000"),
            depth_usd_5=Decimal("500000"),
            depth_usd_10=Decimal("1000000"),
            bid_ask_ratio=Decimal("1"),
            depth_imbalance=Decimal("0"),
            liquidity_score=Decimal("85"),
            condition=LiquidityCondition.EXCELLENT,
        )

        result = analysis.to_dict()

        assert result["symbol"] == "BTC-USD"
        assert result["bid_price"] == 50000.0
        assert result["spread_bps"] == 2.0
        assert result["condition"] == "excellent"


# ============================================================================
# Test: ImpactEstimate to_dict
# ============================================================================


class TestImpactEstimateSerialization:
    """Test ImpactEstimate serialization."""

    def test_to_dict(self):
        """Test conversion to dict for serialization."""
        estimate = ImpactEstimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1.0"),
            estimated_impact_bps=Decimal("5"),
            estimated_avg_price=Decimal("50005"),
            max_impact_price=Decimal("50010"),
            slippage_cost=Decimal("5"),
            recommended_slicing=True,
            max_slice_size=Decimal("0.5"),
            use_post_only=False,
        )

        result = estimate.to_dict()

        assert result["symbol"] == "BTC-USD"
        assert result["quantity"] == 1.0
        assert result["estimated_impact_bps"] == 5.0
        assert result["recommended_slicing"] is True
        assert result["max_slice_size"] == 0.5

    def test_to_dict_with_none_slice_size(self):
        """Test serialization with None max_slice_size."""
        estimate = ImpactEstimate(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1.0"),
            estimated_impact_bps=Decimal("5"),
            estimated_avg_price=Decimal("50005"),
            max_impact_price=Decimal("50010"),
            slippage_cost=Decimal("5"),
            recommended_slicing=False,
            max_slice_size=None,
            use_post_only=False,
        )

        result = estimate.to_dict()

        assert result["max_slice_size"] is None
