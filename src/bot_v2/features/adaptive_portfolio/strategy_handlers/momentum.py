"""Momentum strategy handler."""

import logging
from typing import Any

from bot_v2.data_providers import DataProvider
from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    TierConfig,
    TradingSignal,
)


class MomentumStrategyHandler:
    """Generates momentum-based trading signals."""

    def __init__(
        self,
        data_provider: DataProvider,
        position_size_calculator: PositionSizeCalculator,
    ) -> None:
        """
        Initialize momentum strategy handler.

        Args:
            data_provider: Data provider for historical data
            position_size_calculator: Calculator for position sizing
        """
        self.data_provider = data_provider
        self.position_size_calculator = position_size_calculator
        self.logger = logging.getLogger(__name__)

    def generate_signals(
        self,
        symbols: list[str],
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> list[TradingSignal]:
        """
        Generate momentum-based trading signals.

        Identifies symbols with strong 5-day and 20-day returns.

        Args:
            symbols: List of symbols to analyze
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state

        Returns:
            List of trading signals
        """
        signals = []

        for symbol in symbols:
            try:
                # Get recent data using data provider
                hist = self.data_provider.get_historical_data(symbol, period="60d")

                if len(hist) < 20:
                    continue

                # Calculate momentum indicators
                current_price = self._safe_get_price(hist, "Close", -1)
                price_5d_ago = self._safe_get_price(hist, "Close", -6)
                price_20d_ago = self._safe_get_price(hist, "Close", -21)

                if current_price is None or price_5d_ago is None or price_20d_ago is None:
                    continue

                returns_5d = (current_price / price_5d_ago - 1) * 100
                returns_20d = (current_price / price_20d_ago - 1) * 100

                # Simple momentum signal
                if returns_5d > 2 and returns_20d > 5:
                    confidence = min(0.8, (returns_5d + returns_20d) / 20)

                    # Calculate position size
                    position_size = self.position_size_calculator.calculate(
                        confidence, tier_config, portfolio_snapshot
                    )

                    signal = TradingSignal(
                        symbol=symbol,
                        action="BUY",
                        confidence=confidence,
                        target_position_size=position_size,
                        stop_loss_pct=tier_config.risk.position_stop_loss_pct,
                        strategy_source="momentum",
                        reasoning=f"5d return: {returns_5d:.1f}%, 20d return: {returns_20d:.1f}%",
                    )
                    signals.append(signal)

            except Exception as e:
                self.logger.warning(f"Error analyzing {symbol} for momentum: {e}")
                continue

        return signals

    def _safe_get_price(self, data: Any, column: str, index: int) -> float | None:
        """Safely get price data from either pandas DataFrame or SimpleDataFrame."""
        if hasattr(data, "data"):
            # SimpleDataFrame
            if column in data.data and abs(index) < len(data.data[column]):
                try:
                    return float(data.data[column][index])
                except (TypeError, ValueError):
                    return None
            return None
        else:
            # pandas DataFrame
            columns = getattr(data, "columns", [])
            if column in columns and abs(index) < len(data):
                value = data[column].iloc[index]
                try:
                    return float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return None
            return None
