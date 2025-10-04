"""
Characterization test for LiquidityService integration.

This test locks in the behavior of the refactored LiquidityService
across all four extracted components:
- MetricsTracker: Volume/spread tracking
- DepthAnalyzer: Order book depth analysis
- ImpactEstimator: Market impact calculation
- LiquidityScorer: Liquidity quality scoring

The test verifies the complete flow:
metrics → depth → scoring → impact estimation
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.liquidity_models import LiquidityCondition
from bot_v2.features.live_trade.liquidity_service import LiquidityService


class TestLiquidityServiceCharacterization:
    """Characterization tests for full LiquidityService integration."""

    @pytest.fixture
    def service(self):
        """Create LiquidityService with default configuration."""
        return LiquidityService(
            max_impact_bps=Decimal("50"),  # 5bps max impact threshold
            depth_analysis_levels=20,
            volume_window_minutes=15,
        )

    @pytest.fixture
    def btc_order_book(self):
        """Sample BTC order book with good liquidity."""
        bids = [
            (Decimal("50000"), Decimal("10")),  # $500k at best bid
            (Decimal("49950"), Decimal("20")),  # $999k
            (Decimal("49900"), Decimal("30")),  # $1.497M
            (Decimal("49850"), Decimal("25")),  # $1.246M
            (Decimal("49800"), Decimal("15")),  # $747k
        ]
        asks = [
            (Decimal("50100"), Decimal("8")),  # $400.8k at best ask
            (Decimal("50150"), Decimal("15")),  # $752.25k
            (Decimal("50200"), Decimal("25")),  # $1.255M
            (Decimal("50250"), Decimal("20")),  # $1.005M
            (Decimal("50300"), Decimal("12")),  # $603.6k
        ]
        return bids, asks

    @pytest.fixture
    def eth_order_book_shallow(self):
        """Sample ETH order book with poor liquidity."""
        bids = [
            (Decimal("3000"), Decimal("0.5")),  # $1.5k at best bid
            (Decimal("2990"), Decimal("0.3")),  # $0.897k
            (Decimal("2980"), Decimal("0.2")),  # $0.596k
        ]
        asks = [
            (Decimal("3050"), Decimal("0.4")),  # $1.22k at best ask
            (Decimal("3060"), Decimal("0.3")),  # $0.918k
            (Decimal("3070"), Decimal("0.2")),  # $0.614k
        ]
        return bids, asks

    def test_full_integration_good_liquidity(self, service, btc_order_book):
        """Test complete flow with good liquidity conditions."""
        symbol = "BTC-USD"
        bids, asks = btc_order_book
        now = datetime.now()

        # Step 1: Feed trade data to build volume metrics
        service.update_trade_data(symbol, Decimal("50050"), Decimal("5"))
        service.update_trade_data(symbol, Decimal("50100"), Decimal("3"))
        service.update_trade_data(symbol, Decimal("50000"), Decimal("7"))

        # Step 2: Analyze order book (triggers depth → scoring pipeline)
        analysis = service.analyze_order_book(symbol, bids, asks, timestamp=now)

        # Verify depth analysis
        assert analysis.symbol == symbol
        assert analysis.bid_price == Decimal("50000")
        assert analysis.ask_price == Decimal("50100")
        assert analysis.spread == Decimal("100")  # 100 USD spread
        assert 19 < analysis.spread_bps < 21  # ~20 bps spread

        # Verify depth metrics (should have good depth)
        assert analysis.depth_usd_5 > Decimal("100000")  # >$100k depth within 5%
        assert analysis.depth_usd_1 > Decimal("10000")  # >$10k depth within 1%

        # Verify liquidity scoring
        assert analysis.liquidity_score > Decimal("50")  # Reasonable liquidity
        assert analysis.condition in [
            LiquidityCondition.GOOD,
            LiquidityCondition.FAIR,
        ]

        # Step 3: Estimate market impact
        impact = service.estimate_market_impact(
            symbol=symbol,
            side="buy",
            quantity=Decimal("1"),  # Buy 1 BTC
        )

        # Verify impact estimation
        assert impact.symbol == symbol
        assert impact.side == "buy"
        assert impact.quantity == Decimal("1")
        assert impact.estimated_impact_bps > Decimal("0")
        assert impact.estimated_impact_bps < Decimal("200")  # Reasonable impact
        assert impact.estimated_avg_price > Decimal("50050")  # Above mid price (buy)
        assert impact.slippage_cost > Decimal("0")

        # For small order in good liquidity, shouldn't need slicing
        assert impact.recommended_slicing is False or impact.estimated_impact_bps > Decimal("50")

        # Step 4: Verify snapshot aggregation
        snapshot = service.get_liquidity_snapshot(symbol)
        assert snapshot is not None
        assert snapshot["symbol"] == symbol
        assert snapshot["liquidity_score"] == float(analysis.liquidity_score)
        assert snapshot["condition"] == analysis.condition.value

        # Volume metrics should be present
        assert "volume_15m" in snapshot
        assert snapshot["volume_15m"] > 0

    def test_full_integration_poor_liquidity(self, service, eth_order_book_shallow):
        """Test complete flow with poor liquidity conditions."""
        symbol = "ETH-USD"
        bids, asks = eth_order_book_shallow
        now = datetime.now()

        # Step 1: Minimal trade data (low volume)
        service.update_trade_data(symbol, Decimal("3025"), Decimal("0.5"))

        # Step 2: Analyze shallow order book
        analysis = service.analyze_order_book(symbol, bids, asks, timestamp=now)

        # Verify depth analysis shows poor liquidity
        assert analysis.spread == Decimal("50")  # Wide spread
        assert analysis.spread_bps > Decimal("150")  # >15bps spread is poor
        assert analysis.depth_usd_5 < Decimal("50000")  # Shallow depth

        # Verify liquidity scoring reflects poor conditions
        assert analysis.liquidity_score < Decimal("60")  # Low score
        assert analysis.condition in [
            LiquidityCondition.FAIR,
            LiquidityCondition.POOR,
            LiquidityCondition.CRITICAL,
        ]

        # Step 3: Estimate impact for larger order
        impact = service.estimate_market_impact(
            symbol=symbol,
            side="sell",
            quantity=Decimal("10"),  # Sell 10 ETH (large relative to depth)
        )

        # Verify high impact due to poor liquidity
        assert impact.estimated_impact_bps > Decimal("50")  # High impact
        assert impact.estimated_avg_price < Decimal("3025")  # Below mid price (sell)

        # Should recommend slicing for large impact
        assert impact.recommended_slicing is True
        assert impact.max_slice_size is not None
        assert impact.max_slice_size < Decimal("10")

        # Should recommend post-only due to poor liquidity
        assert impact.use_post_only is True

    def test_integration_with_stale_data(self, service, btc_order_book):
        """Test behavior when no recent trade data is available."""
        symbol = "XRP-USD"
        bids, asks = btc_order_book

        # Analyze order book without any trade data
        analysis = service.analyze_order_book(symbol, bids, asks)

        # Should still produce valid analysis
        assert analysis.symbol == symbol
        assert analysis.liquidity_score >= Decimal("0")

        # Impact estimation with no volume history
        impact = service.estimate_market_impact(
            symbol=symbol,
            side="buy",
            quantity=Decimal("1000"),
        )

        # Should use minimum volume floor
        assert impact.estimated_impact_bps > Decimal("0")
        # Conservative recommendations without volume data
        assert impact.recommended_slicing is True

    def test_integration_without_order_book(self, service):
        """Test conservative fallback when no order book data."""
        symbol = "DOGE-USD"

        # Try to estimate impact without order book analysis
        impact = service.estimate_market_impact(
            symbol=symbol,
            side="buy",
            quantity=Decimal("1000"),
        )

        # Should return conservative estimate
        assert impact.estimated_impact_bps == Decimal("100")  # 10bps conservative
        assert impact.recommended_slicing is True
        assert impact.max_slice_size == Decimal("100")  # quantity / 10
        assert impact.use_post_only is True

        # Snapshot should be None without analysis
        snapshot = service.get_liquidity_snapshot(symbol)
        assert snapshot is None

    def test_integration_multiple_symbols(self, service, btc_order_book, eth_order_book_shallow):
        """Test concurrent analysis of multiple symbols."""
        btc_bids, btc_asks = btc_order_book
        eth_bids, eth_asks = eth_order_book_shallow

        # Feed trade data for both symbols
        service.update_trade_data("BTC-USD", Decimal("50050"), Decimal("5"))
        service.update_trade_data("ETH-USD", Decimal("3025"), Decimal("2"))

        # Analyze both order books
        btc_analysis = service.analyze_order_book("BTC-USD", btc_bids, btc_asks)
        eth_analysis = service.analyze_order_book("ETH-USD", eth_bids, eth_asks)

        # Both should be cached independently
        assert btc_analysis.symbol == "BTC-USD"
        assert eth_analysis.symbol == "ETH-USD"

        # Get snapshots for both
        btc_snapshot = service.get_liquidity_snapshot("BTC-USD")
        eth_snapshot = service.get_liquidity_snapshot("ETH-USD")

        assert btc_snapshot["symbol"] == "BTC-USD"
        assert eth_snapshot["symbol"] == "ETH-USD"

        # BTC should have better liquidity than ETH
        assert btc_snapshot["liquidity_score"] > eth_snapshot["liquidity_score"]

    def test_integration_time_based_metrics(self, service, btc_order_book):
        """Test volume metrics respect time windows."""
        symbol = "BTC-USD"
        bids, asks = btc_order_book
        now = datetime.now()

        # Add recent trade (within 15min window)
        service.update_trade_data(symbol, Decimal("50000"), Decimal("10"))

        # Add old trade (outside 15min window) - shouldn't affect volume_15m
        old_time = now - timedelta(minutes=20)
        service._metrics_tracker.add_trade(symbol, Decimal("50000"), Decimal("100"), old_time)

        # Analyze and check metrics
        service.analyze_order_book(symbol, bids, asks, timestamp=now)

        volume_metrics = service._metrics_tracker.get_volume_metrics(symbol)

        # Only recent trade should count in 15min window
        # Note: exact value depends on timestamp, but should be closer to 10 than 110
        assert volume_metrics["volume_15m"] < Decimal("600000")  # Not including old trade

    def test_integration_preserves_api_surface(self, service, btc_order_book):
        """Test that refactored service maintains original API."""
        symbol = "BTC-USD"
        bids, asks = btc_order_book

        # All original public methods should exist and work

        # 1. update_trade_data
        service.update_trade_data(symbol, Decimal("50000"), Decimal("1"))

        # 2. analyze_order_book
        analysis = service.analyze_order_book(symbol, bids, asks)
        assert analysis is not None

        # 3. estimate_market_impact
        impact = service.estimate_market_impact(symbol, "buy", Decimal("1"))
        assert impact is not None

        # 4. get_liquidity_snapshot
        snapshot = service.get_liquidity_snapshot(symbol)
        assert snapshot is not None

        # All dataclasses should have to_dict() methods
        assert callable(getattr(analysis, "to_dict", None))
        assert callable(getattr(impact, "to_dict", None))

        analysis_dict = analysis.to_dict()
        impact_dict = impact.to_dict()

        assert isinstance(analysis_dict, dict)
        assert isinstance(impact_dict, dict)

    def test_integration_dependency_injection(self):
        """Test that all dependencies can be injected."""
        from bot_v2.features.live_trade.depth_analyzer import DepthAnalyzer
        from bot_v2.features.live_trade.impact_estimator import ImpactEstimator
        from bot_v2.features.live_trade.liquidity_metrics_tracker import MetricsTracker
        from bot_v2.features.live_trade.liquidity_scorer import LiquidityScorer

        # Create custom instances
        metrics_tracker = MetricsTracker(window_minutes=30)
        depth_analyzer = DepthAnalyzer()
        impact_estimator = ImpactEstimator(max_impact_bps=Decimal("100"))
        liquidity_scorer = LiquidityScorer()

        # Inject all dependencies
        service = LiquidityService(
            metrics_tracker=metrics_tracker,
            depth_analyzer=depth_analyzer,
            impact_estimator=impact_estimator,
            liquidity_scorer=liquidity_scorer,
        )

        # Verify injection worked
        assert service._metrics_tracker is metrics_tracker
        assert service._depth_analyzer is depth_analyzer
        assert service._impact_estimator is impact_estimator
        assert service._liquidity_scorer is liquidity_scorer

        # Service should still function
        assert service.max_impact_bps == Decimal("50")  # Default value
