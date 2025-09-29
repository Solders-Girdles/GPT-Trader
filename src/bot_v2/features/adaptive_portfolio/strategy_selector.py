"""
Strategy selection based on portfolio tier.

Selects and configures appropriate trading strategies for each tier.
"""

from typing import TYPE_CHECKING, Any, TypeAlias

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]

HAS_PANDAS = pd is not None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import DataFrame as _PandasDataFrame

    DataFrame: TypeAlias = _PandasDataFrame
else:
    DataFrame: TypeAlias = Any

import logging

from ...data_providers import DataProvider
from .types import PortfolioConfig, PortfolioSnapshot, TierConfig, TradingSignal


class StrategySelector:
    """Selects appropriate strategies based on portfolio tier."""

    def __init__(self, config: PortfolioConfig, data_provider: DataProvider) -> None:
        """Initialize with portfolio configuration and data provider."""
        self.config = config
        self.data_provider = data_provider
        self.logger = logging.getLogger(__name__)

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

    def generate_signals(
        self,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
        market_data: dict[str, DataFrame] | None = None,
    ) -> list[TradingSignal]:
        """
        Generate trading signals appropriate for current tier.

        Args:
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state
            market_data: Optional market data for analysis

        Returns:
            List of tier-appropriate trading signals
        """
        signals = []

        # Get symbols to analyze (simplified - would use more sophisticated universe selection)
        symbols = self._get_symbol_universe(tier_config, portfolio_snapshot)

        # Generate signals for each strategy in tier
        for strategy_name in tier_config.strategies:
            strategy_signals = self._generate_strategy_signals(
                strategy_name, symbols, tier_config, portfolio_snapshot, market_data
            )
            signals.extend(strategy_signals)

        # Filter and rank signals
        filtered_signals = self._filter_signals(signals, tier_config, portfolio_snapshot)
        ranked_signals = self._rank_signals(filtered_signals, tier_config)

        # Limit to appropriate number for tier
        max_signals = self._calculate_max_signals(tier_config, portfolio_snapshot)
        final_signals = ranked_signals[:max_signals]

        self.logger.info(
            f"Generated {len(final_signals)} signals for {tier_config.name} "
            f"using strategies: {', '.join(tier_config.strategies)}"
        )

        return final_signals

    def _get_symbol_universe(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> list[str]:
        """Get appropriate symbol universe for tier."""

        # Simplified universe - would use more sophisticated selection in production
        base_universe = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "NFLX",
            "SPY",
            "QQQ",
            "IWM",
            "VTI",
            "BRK-B",
            "JNJ",
            "V",
            "JPM",
            "UNH",
            "HD",
            "PG",
            "DIS",
            "MA",
            "BAC",
            "ADBE",
            "CRM",
            "PYPL",
        ]

        # Adjust universe size based on tier
        if tier_config.name == "Micro Portfolio":
            # Small universe for micro portfolios
            return base_universe[:8]
        elif tier_config.name == "Small Portfolio":
            return base_universe[:12]
        elif tier_config.name == "Medium Portfolio":
            return base_universe[:18]
        else:  # Large Portfolio
            return base_universe

    def _generate_strategy_signals(
        self,
        strategy_name: str,
        symbols: list[str],
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
        market_data: dict[str, DataFrame] | None,
    ) -> list[TradingSignal]:
        """Generate signals for specific strategy."""

        if strategy_name == "momentum":
            return self._momentum_strategy(symbols, tier_config, portfolio_snapshot)
        elif strategy_name == "mean_reversion":
            return self._mean_reversion_strategy(symbols, tier_config, portfolio_snapshot)
        elif strategy_name == "trend_following":
            return self._trend_following_strategy(symbols, tier_config, portfolio_snapshot)
        elif strategy_name == "ml_enhanced":
            return self._ml_enhanced_strategy(symbols, tier_config, portfolio_snapshot)
        else:
            self.logger.warning(f"Unknown strategy: {strategy_name}")
            return []

    def _momentum_strategy(
        self, symbols: list[str], tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> list[TradingSignal]:
        """Simple momentum strategy."""

        # Can work with either pandas or SimpleDataFrame

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
                    position_size = self._calculate_signal_position_size(
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

    def _mean_reversion_strategy(
        self, symbols: list[str], tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> list[TradingSignal]:
        """Simple mean reversion strategy."""

        # Can work with either pandas or SimpleDataFrame

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
                    position_size = self._calculate_signal_position_size(
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

    def _trend_following_strategy(
        self, symbols: list[str], tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> list[TradingSignal]:
        """Simple trend following strategy."""

        # Can work with either pandas or SimpleDataFrame

        signals = []

        for symbol in symbols:
            try:
                # Get recent data using data provider
                hist = self.data_provider.get_historical_data(symbol, period="100d")

                if len(hist) < 50:
                    continue

                # Calculate trend indicators
                price = self._safe_get_price(hist, "Close", -1)

                if price is None:
                    continue

                # Calculate moving averages
                if hasattr(hist, "data"):
                    # SimpleDataFrame - manual calculation
                    closes = hist.data["Close"]
                    if len(closes) >= 50:
                        sma_10 = sum(closes[-10:]) / 10
                        sma_30 = sum(closes[-30:]) / 30
                        sma_50 = sum(closes[-50:]) / 50
                    else:
                        continue
                else:
                    # pandas DataFrame
                    sma_10 = hist["Close"].rolling(10).mean().iloc[-1]
                    sma_30 = hist["Close"].rolling(30).mean().iloc[-1]
                    sma_50 = hist["Close"].rolling(50).mean().iloc[-1]

                # Trend following signal (all MAs aligned)
                if sma_10 > sma_30 > sma_50 and price > sma_10:
                    # Calculate trend strength
                    trend_strength = ((sma_10 - sma_50) / sma_50) * 100
                    confidence = min(0.9, trend_strength / 10)

                    # Calculate position size
                    position_size = self._calculate_signal_position_size(
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

    def _ml_enhanced_strategy(
        self, symbols: list[str], tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> list[TradingSignal]:
        """ML-enhanced strategy (simplified version)."""

        # This would integrate with the ml_strategy slice in production
        # For now, return enhanced momentum signals
        signals = []

        momentum_signals = self._momentum_strategy(symbols, tier_config, portfolio_snapshot)

        # "Enhance" momentum signals with ML-like adjustments
        for signal in momentum_signals:
            if signal.confidence > 0.6:  # Only enhance high-confidence signals
                # Simulate ML enhancement by adjusting confidence
                enhanced_confidence = min(0.95, signal.confidence * 1.2)

                enhanced_signal = TradingSignal(
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=enhanced_confidence,
                    target_position_size=signal.target_position_size,
                    stop_loss_pct=signal.stop_loss_pct,
                    strategy_source="ml_enhanced",
                    reasoning=f"ML-enhanced: {signal.reasoning}",
                )
                signals.append(enhanced_signal)

        return signals

    def _filter_signals(
        self,
        signals: list[TradingSignal],
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> list[TradingSignal]:
        """Filter signals based on tier constraints and portfolio state."""

        filtered = []

        for signal in signals:
            # Check if we already have this position
            existing_position = any(
                pos.symbol == signal.symbol for pos in portfolio_snapshot.positions
            )

            if existing_position:
                continue  # Skip if we already own it

            # Check minimum confidence for tier
            min_confidence = self._get_min_confidence_for_tier(tier_config)
            if signal.confidence < min_confidence:
                continue

            # Check if position size meets tier requirements
            if signal.target_position_size < tier_config.min_position_size:
                continue

            # Check market constraints
            if not self._meets_market_constraints(signal.symbol):
                continue

            filtered.append(signal)

        return filtered

    def _rank_signals(
        self, signals: list[TradingSignal], tier_config: TierConfig
    ) -> list[TradingSignal]:
        """Rank signals by attractiveness for tier."""

        # Simple ranking by confidence for now
        # In production, would use more sophisticated ranking
        return sorted(signals, key=lambda s: s.confidence, reverse=True)

    def _calculate_max_signals(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> int:
        """Calculate maximum number of new signals to generate."""

        current_positions = portfolio_snapshot.positions_count
        max_positions = tier_config.positions.max_positions
        target_positions = tier_config.positions.target_positions

        # Prioritize reaching target positions
        spots_available = max_positions - current_positions
        spots_to_target = target_positions - current_positions

        # Return the minimum of available spots and spots needed to reach target
        return max(0, min(spots_available, spots_to_target))

    def _calculate_signal_position_size(
        self, confidence: float, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> float:
        """Calculate position size for signal based on confidence and tier."""

        # Base position size
        target_positions = tier_config.positions.target_positions
        base_size = portfolio_snapshot.total_value / target_positions

        # Adjust for confidence
        confidence_adjusted = base_size * confidence

        # Ensure minimum size
        min_size = tier_config.min_position_size
        final_size = max(confidence_adjusted, min_size)

        # Ensure not too large (25% max)
        max_size = portfolio_snapshot.total_value * 0.25
        final_size = min(final_size, max_size)

        return final_size

    def _get_min_confidence_for_tier(self, tier_config: TierConfig) -> float:
        """Get minimum confidence threshold for tier."""

        # More conservative for smaller portfolios
        tier_name = tier_config.name.lower()

        if "micro" in tier_name:
            return 0.7  # High confidence required
        elif "small" in tier_name:
            return 0.6  # Moderate confidence
        elif "medium" in tier_name:
            return 0.5  # Standard confidence
        else:  # Large
            return 0.4  # More aggressive with large portfolios

    def _meets_market_constraints(self, symbol: str) -> bool:
        """Check if symbol meets market constraints."""

        constraints = self.config.market_constraints

        # Check excluded symbols
        for excluded in constraints.excluded_symbols:
            if excluded.upper() in symbol.upper():
                return False

        # Additional checks would go here (price, volume, etc.)
        # For now, assume all pass
        return True

    def get_strategy_allocation(
        self, tier_config: TierConfig, portfolio_snapshot: PortfolioSnapshot
    ) -> dict[str, float]:
        """
        Get recommended allocation percentages for each strategy in tier.

        Args:
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state

        Returns:
            Dictionary of strategy name -> allocation percentage
        """
        strategies = tier_config.strategies

        if len(strategies) == 1:
            return {strategies[0]: 100.0}

        # Default equal allocation
        equal_weight = 100.0 / len(strategies)
        allocation = {strategy: equal_weight for strategy in strategies}

        # Adjust allocation based on tier and market conditions
        # This is simplified - would use more sophisticated allocation in production

        if "micro" in tier_config.name.lower():
            # Conservative allocation for micro portfolios
            if "momentum" in allocation:
                allocation["momentum"] = 60.0
            if "mean_reversion" in allocation:
                allocation["mean_reversion"] = 40.0

        elif "large" in tier_config.name.lower():
            # More balanced for large portfolios
            if "ml_enhanced" in allocation:
                allocation["ml_enhanced"] = 40.0
                # Redistribute remainder equally
                remaining = 60.0 / (len(strategies) - 1)
                for strategy in strategies:
                    if strategy != "ml_enhanced":
                        allocation[strategy] = remaining

        return allocation
