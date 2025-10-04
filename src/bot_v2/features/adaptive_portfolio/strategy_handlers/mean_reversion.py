"""Mean reversion strategy handler."""

import logging
from typing import Any

from bot_v2.data_providers import DataProvider
from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    TierConfig,
    TradingSignal,
)


class MeanReversionStrategyHandler:
    """Generates mean reversion trading signals."""

    def __init__(
        self,
        data_provider: DataProvider,
        position_size_calculator: PositionSizeCalculator,
    ) -> None:
        """
        Initialize mean reversion strategy handler.

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
        Generate mean reversion trading signals.

        Identifies oversold symbols using z-score.

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

                # Calculate mean reversion indicators
                price = self._safe_get_price(hist, "Close", -1)

                if price is None:
                    continue

                # Calculate simple moving average and std dev
                if hasattr(hist, "data"):
                    # SimpleDataFrame - manual calculation
                    closes = hist.data["Close"][-20:]  # Last 20 closes
                    sma_20 = sum(closes) / len(closes)
                    variance = sum((x - sma_20) ** 2 for x in closes) / len(closes)
                    std_20 = variance**0.5
                else:
                    # pandas DataFrame
                    sma_20 = hist["Close"].rolling(20).mean().iloc[-1]
                    std_20 = hist["Close"].rolling(20).std().iloc[-1]

                # Z-score for mean reversion
                z_score = (price - sma_20) / std_20 if std_20 > 0 else 0

                # Mean reversion signal (buy when oversold)
                if z_score < -1.5:  # Oversold
                    confidence = min(0.8, abs(z_score) / 3)

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
                        strategy_source="mean_reversion",
                        reasoning=f"Z-score: {z_score:.2f} (oversold)",
                    )
                    signals.append(signal)

            except Exception as e:
                self.logger.warning(f"Error analyzing {symbol} for mean reversion: {e}")
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
