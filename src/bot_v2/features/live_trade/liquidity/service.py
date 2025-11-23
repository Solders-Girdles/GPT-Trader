from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

from .metrics import LiquidityMetrics
from .models import DepthAnalysis, ImpactEstimate, LiquidityCondition
from .utils import ensure_utc_aware, utc_now

logger = get_logger(__name__, component="liquidity_service")


class LiquidityService:
    """Production liquidity analysis service."""

    def __init__(
        self,
        max_impact_bps: Decimal = Decimal("50"),
        depth_analysis_levels: int = 20,
        volume_window_minutes: int = 15,
    ) -> None:
        self.max_impact_bps = max_impact_bps
        self.depth_levels = depth_analysis_levels
        self._symbol_metrics: dict[str, LiquidityMetrics] = {}
        self._latest_analysis: dict[str, DepthAnalysis] = {}

        logger.info("LiquidityService initialized - max impact: %sbps", max_impact_bps)

    def _get_metrics(self, symbol: str) -> LiquidityMetrics:
        metrics = self._symbol_metrics.get(symbol)
        if metrics is None:
            metrics = LiquidityMetrics(window_minutes=15)
            self._symbol_metrics[symbol] = metrics
        return metrics

    def update_trade_data(self, symbol: str, price: Decimal, size: Decimal) -> None:
        self._get_metrics(symbol).add_trade(price, size)

    def analyze_order_book(
        self,
        symbol: str,
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
        timestamp: datetime | None = None,
    ) -> DepthAnalysis:
        ts = utc_now() if timestamp is None else ensure_utc_aware(timestamp)

        if not bids or not asks:
            analysis = DepthAnalysis(
                symbol=symbol,
                timestamp=ts,
                bid_price=Decimal("0"),
                ask_price=Decimal("0"),
                bid_size=Decimal("0"),
                ask_size=Decimal("0"),
                spread=Decimal("0"),
                spread_bps=Decimal("10000"),
                depth_usd_1=Decimal("0"),
                depth_usd_5=Decimal("0"),
                depth_usd_10=Decimal("0"),
                bid_ask_ratio=Decimal("1"),
                depth_imbalance=Decimal("0"),
                liquidity_score=Decimal("0"),
                condition=LiquidityCondition.CRITICAL,
            )
            self._latest_analysis[symbol] = analysis
            return analysis

        best_bid, bid_size = bids[0]
        best_ask, ask_size = asks[0]

        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else Decimal("10000")

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

        bid_ask_ratio = bid_size / ask_size if ask_size > 0 else Decimal("999")
        total_depth_5 = bid_depth_5 + ask_depth_5
        depth_imbalance = (
            ((bid_depth_5 - ask_depth_5) / total_depth_5) if total_depth_5 > 0 else Decimal("0")
        )

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

        self._get_metrics(symbol).add_spread(spread_bps, ts)

        analysis = DepthAnalysis(
            symbol=symbol,
            timestamp=ts,
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
        analysis = self._latest_analysis.get(symbol)
        if not analysis:
            logger.warning("No depth analysis available for %s", symbol)
            return ImpactEstimate(
                symbol=symbol,
                side=side,
                quantity=quantity,
                estimated_impact_bps=Decimal("100"),
                estimated_avg_price=Decimal("0"),
                max_impact_price=Decimal("0"),
                slippage_cost=Decimal("0"),
                recommended_slicing=True,
                max_slice_size=quantity / 10,
                use_post_only=True,
            )

        if book_data:
            _bids, _asks = book_data  # placeholder for future integration

        mid_price = (analysis.bid_price + analysis.ask_price) / 2
        notional = quantity * mid_price

        volume_metrics = self._get_metrics(symbol).get_volume_metrics()
        volume_15m = max(volume_metrics["volume_15m"], Decimal("1000"))
        base_impact_bps = (notional / volume_15m).sqrt() * 100

        relevant_depth = analysis.depth_usd_5
        if relevant_depth > 0 and notional > relevant_depth:
            depth_multiplier = (notional / relevant_depth).sqrt()
            base_impact_bps *= depth_multiplier

        spread_multiplier = 1 + (analysis.spread_bps / 1000)
        condition_multiplier = {
            LiquidityCondition.EXCELLENT: Decimal("0.5"),
            LiquidityCondition.GOOD: Decimal("1.0"),
            LiquidityCondition.FAIR: Decimal("1.5"),
            LiquidityCondition.POOR: Decimal("2.0"),
            LiquidityCondition.CRITICAL: Decimal("3.0"),
        }[analysis.condition]

        final_impact_bps = base_impact_bps * spread_multiplier * condition_multiplier

        if side == "buy":
            estimated_avg_price = mid_price * (1 + final_impact_bps / 10000)
            max_impact_price = mid_price * (1 + final_impact_bps * Decimal("1.5") / 10000)
        else:
            estimated_avg_price = mid_price * (1 - final_impact_bps / 10000)
            max_impact_price = mid_price * (1 - final_impact_bps * Decimal("1.5") / 10000)

        slippage_cost = abs(estimated_avg_price - mid_price) * quantity

        recommended_slicing = final_impact_bps > self.max_impact_bps
        max_slice_size = None
        if recommended_slicing and base_impact_bps > 0:
            target_notional = (self.max_impact_bps / base_impact_bps) ** 2 * notional
            max_slice_size = target_notional / mid_price if mid_price > 0 else None

        use_post_only = (
            analysis.condition
            in {LiquidityCondition.FAIR, LiquidityCondition.POOR, LiquidityCondition.CRITICAL}
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
        analysis = self._latest_analysis.get(symbol)
        if not analysis:
            return None

        metrics = self._get_metrics(symbol)
        return {
            **analysis.to_dict(),
            **metrics.get_volume_metrics(),
            **metrics.get_spread_metrics(),
        }

    def _calculate_depth(
        self, levels: list[tuple[Decimal, Decimal]], min_price: Decimal, max_price: Decimal
    ) -> Decimal:
        total_size = Decimal("0")
        for price, size in levels:
            if min_price <= price <= max_price:
                total_size += size
            elif price > max_price:
                break
        return total_size

    def _score_spread(self, spread_bps: Decimal) -> Decimal:
        if spread_bps <= 1:
            return Decimal("100")
        if spread_bps <= 5:
            return Decimal("80")
        if spread_bps <= 10:
            return Decimal("60")
        if spread_bps <= 20:
            return Decimal("40")
        if spread_bps <= 50:
            return Decimal("20")
        return Decimal("0")

    def _score_depth(self, depth_usd: Decimal, mid_price: Decimal) -> Decimal:
        return min(depth_usd / Decimal("10000"), Decimal("1")) * Decimal("100")

    def _score_imbalance(self, imbalance: Decimal) -> Decimal:
        return max(Decimal("0"), Decimal("100") - imbalance * Decimal("200"))

    def _determine_condition(self, score: Decimal) -> LiquidityCondition:
        if score >= Decimal("80"):
            return LiquidityCondition.EXCELLENT
        if score >= Decimal("60"):
            return LiquidityCondition.GOOD
        if score >= Decimal("40"):
            return LiquidityCondition.FAIR
        if score >= Decimal("20"):
            return LiquidityCondition.POOR
        return LiquidityCondition.CRITICAL


async def create_liquidity_service(**kwargs: Any) -> LiquidityService:
    service = LiquidityService(**kwargs)
    logger.info("LiquidityService created and ready")
    return service


__all__ = ["LiquidityService", "create_liquidity_service"]
