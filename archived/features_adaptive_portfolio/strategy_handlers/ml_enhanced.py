"""ML-enhanced strategy handler."""

from bot_v2.features.adaptive_portfolio.strategy_handlers.momentum import MomentumStrategyHandler
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    TierConfig,
    TradingSignal,
)


class MLEnhancedStrategyHandler:
    """ML-enhanced strategy that boosts momentum signals."""

    def __init__(self, momentum_handler: MomentumStrategyHandler) -> None:
        """
        Initialize ML-enhanced strategy handler.

        Args:
            momentum_handler: Momentum strategy to enhance
        """
        self.momentum_handler = momentum_handler

    def generate_signals(
        self,
        symbols: list[str],
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> list[TradingSignal]:
        """
        Generate ML-enhanced signals by boosting momentum signals.

        Args:
            symbols: List of symbols to analyze
            tier_config: Current tier configuration
            portfolio_snapshot: Current portfolio state

        Returns:
            List of ML-enhanced trading signals
        """
        signals = []

        momentum_signals = self.momentum_handler.generate_signals(
            symbols, tier_config, portfolio_snapshot
        )

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
