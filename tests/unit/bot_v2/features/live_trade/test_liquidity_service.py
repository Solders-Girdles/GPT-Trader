from __future__ import annotations

from datetime import datetime, timedelta
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
