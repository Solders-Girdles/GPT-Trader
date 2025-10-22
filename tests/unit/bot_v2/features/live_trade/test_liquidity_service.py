from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from tests.fixtures.live_trade import build_order_book, build_trade_stream

from bot_v2.features.live_trade.liquidity_service import (
    LiquidityCondition,
    LiquidityService,
)


def test_analyze_order_book_scores_and_caches():
    service = LiquidityService(max_impact_bps=Decimal("30"))
    bids, asks = build_order_book(mid_price=Decimal("100"), spread=Decimal("0.5"), levels=3)

    analysis = service.analyze_order_book("BTC-USD", bids, asks, timestamp=datetime.now())

    assert analysis.spread > 0
    assert analysis.depth_usd_5 > 0
    assert analysis.condition in LiquidityCondition
    snapshot = service.get_liquidity_snapshot("BTC-USD")
    assert snapshot is not None
    assert "volume_1m" in snapshot


def test_analyze_order_book_handles_missing_data():
    service = LiquidityService()
    analysis = service.analyze_order_book("ETH-USD", [], [])
    assert analysis.condition == LiquidityCondition.CRITICAL
    assert analysis.depth_usd_5 == Decimal("0")


def test_metrics_record_volume_and_spread():
    service = LiquidityService()
    service.update_trade_data("BTC-USD", Decimal("100"), Decimal("0.5"))
    past_trades = build_trade_stream(mid_price=Decimal("100"), count=3, step=Decimal("0.1"))
    for ts, price, size in past_trades:
        service._get_metrics("BTC-USD")._trade_data.append((ts - timedelta(minutes=1), price, size))
    service._get_metrics("BTC-USD").add_spread(Decimal("10"))

    bids, asks = build_order_book(mid_price=Decimal("100"), spread=Decimal("0.5"), levels=2)
    service.analyze_order_book("BTC-USD", bids, asks)

    snapshot = service.get_liquidity_snapshot("BTC-USD")
    assert snapshot is not None
    assert snapshot["volume_1m"] > 0
    assert snapshot["avg_spread_bps"] >= Decimal("0")


def test_estimate_market_impact_uses_latest_analysis():
    service = LiquidityService(max_impact_bps=Decimal("20"))
    bids, asks = build_order_book(mid_price=Decimal("200"), spread=Decimal("0.5"), levels=5)
    service.analyze_order_book("BTC-USD", bids, asks)

    impact = service.estimate_market_impact("BTC-USD", "buy", Decimal("2"))
    assert impact.estimated_impact_bps > 0
    assert impact.recommended_slicing in {True, False}


def test_estimate_market_impact_without_analysis():
    service = LiquidityService()
    impact = service.estimate_market_impact("ETH-USD", "sell", Decimal("1"))
    assert impact.recommended_slicing is True
    assert impact.use_post_only is True


def test_datetime_compatibility_with_naive_and_aware_timestamps():
    """Test that both naive and aware datetimes work correctly without crashes."""
    service = LiquidityService()
    bids, asks = build_order_book(mid_price=Decimal("100"), spread=Decimal("0.5"), levels=3)

    # Test with naive datetime (existing code pattern) - should not crash
    naive_timestamp = datetime.now()
    analysis1 = service.analyze_order_book("BTC-USD", bids, asks, timestamp=naive_timestamp)
    assert analysis1.spread > 0

    # Test with aware datetime (new pattern) - should not crash
    aware_timestamp = datetime.now(tz=timezone.utc)
    analysis2 = service.analyze_order_book("ETH-USD", bids, asks, timestamp=aware_timestamp)
    assert analysis2.spread > 0

    # Test with mixed patterns - no TypeError should occur
    service.update_trade_data("BTC-USD", Decimal("100"), Decimal("1"))
    metrics = service.get_liquidity_snapshot("BTC-USD")
    assert "volume_1m" in metrics

    # Test LiquidityMetrics directly with mixed datetime types
    from bot_v2.features.live_trade.liquidity_service import LiquidityMetrics

    metrics_obj = LiquidityMetrics()

    # Use current timestamps to ensure they're within the window
    current_naive = datetime.now()
    current_aware = datetime.now(tz=timezone.utc)

    # THE KEY TEST: These should not crash with mixed datetime types
    # This would have thrown "TypeError: can't compare offset-naive and offset-aware datetimes" before the fix
    try:
        metrics_obj.add_trade(Decimal("100"), Decimal("1"), current_naive)
        metrics_obj.add_trade(Decimal("101"), Decimal("2"), current_aware)
        metrics_obj.add_spread(Decimal("5"), current_naive)

        # If we get here without exceptions, the datetime compatibility fix is working
        volume_metrics = metrics_obj.get_volume_metrics()
        spread_metrics = metrics_obj.get_spread_metrics()

        # The specific values don't matter as much as the fact that no crash occurred
        assert isinstance(volume_metrics, dict)
        assert isinstance(spread_metrics, dict)

    except TypeError as e:
        if "can't compare offset-naive and offset-aware datetimes" in str(e):
            # This should not happen with our fix
            assert False, f"DateTime compatibility fix failed: {e}"
        else:
            # Some other TypeError - re-raise
            raise
