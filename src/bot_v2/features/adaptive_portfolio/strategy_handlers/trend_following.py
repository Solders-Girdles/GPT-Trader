"""Trend following strategy handler."""

import logging
from typing import Any

from bot_v2.data_providers import DataProvider
from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    TierConfig,
    TradingSignal,
)


class TrendFollowingStrategyHandler:
    """Generates trend following trading signals."""

    def __init__(
        self,
        data_provider: DataProvider,
        position_size_calculator: PositionSizeCalculator,
    ) -> None:
        self.data_provider = data_provider
        self.position_size_calculator = position_size_calculator
        self.logger = logging.getLogger(__name__)

    def generate_signals(
        self,
        symbols: list[str],
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> list[TradingSignal]:
        """Generate trend following signals based on MA alignment."""
        signals = []

        for symbol in symbols:
            try:
                hist = self.data_provider.get_historical_data(symbol, period="100d")

                if len(hist) < 50:
                    continue

                price = self._safe_get_price(hist, "Close", -1)

                if price is None:
                    continue

                # Calculate moving averages
                if hasattr(hist, "data"):
                    closes = hist.data["Close"]
                    if len(closes) >= 50:
                        sma_10 = sum(closes[-10:]) / 10
                        sma_30 = sum(closes[-30:]) / 30
                        sma_50 = sum(closes[-50:]) / 50
                    else:
                        continue
                else:
                    sma_10 = hist["Close"].rolling(10).mean().iloc[-1]
                    sma_30 = hist["Close"].rolling(30).mean().iloc[-1]
                    sma_50 = hist["Close"].rolling(50).mean().iloc[-1]

                # Trend following signal (all MAs aligned)
                if sma_10 > sma_30 > sma_50 and price > sma_10:
                    trend_strength = ((sma_10 - sma_50) / sma_50) * 100
                    confidence = min(0.9, trend_strength / 10)

                    position_size = self.position_size_calculator.calculate(
                        confidence, tier_config, portfolio_snapshot
                    )

                    signal = TradingSignal(
                        symbol=symbol,
                        action="BUY",
                        confidence=confidence,
                        target_position_size=position_size,
                        stop_loss_pct=tier_config.risk.position_stop_loss_pct,
                        strategy_source="trend_following",
                        reasoning=f"Strong uptrend, trend strength: {trend_strength:.1f}%",
                    )
                    signals.append(signal)

            except Exception as e:
                self.logger.warning(f"Error analyzing {symbol} for trend following: {e}")
                continue

        return signals

    def _safe_get_price(self, data: Any, column: str, index: int) -> float | None:
        if hasattr(data, "data"):
            if column in data.data and abs(index) < len(data.data[column]):
                try:
                    return float(data.data[column][index])
                except (TypeError, ValueError):
                    return None
            return None
        else:
            columns = getattr(data, "columns", [])
            if column in columns and abs(index) < len(data):
                value = data[column].iloc[index]
                try:
                    return float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return None
            return None
