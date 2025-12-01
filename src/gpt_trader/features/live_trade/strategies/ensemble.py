"""
Ensemble Strategy that orchestrates multiple signals and a combiner.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from gpt_trader.core import Product
from gpt_trader.features.live_trade.combiners.regime import (
    RegimeAwareCombiner,
    RegimeCombinerConfig,
)
from gpt_trader.features.live_trade.signals.mean_reversion import (
    MeanReversionSignal,
    MeanReversionSignalConfig,
)
from gpt_trader.features.live_trade.signals.momentum import (
    MomentumSignal,
    MomentumSignalConfig,
)
from gpt_trader.features.live_trade.signals.protocol import (
    SignalCombiner,
    SignalGenerator,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.trend import TrendSignal, TrendSignalConfig
from gpt_trader.features.live_trade.strategies.base import BaseStrategy
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


@dataclass
class EnsembleStrategyConfig:
    """Configuration for EnsembleStrategy."""

    # Thresholds for net signal
    buy_threshold: float = 0.2
    sell_threshold: float = -0.2
    close_threshold: float = 0.1  # |signal| < threshold -> close

    # Risk
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05

    # Components Config
    trend_config: TrendSignalConfig = field(default_factory=TrendSignalConfig)
    mean_reversion_config: MeanReversionSignalConfig = field(
        default_factory=MeanReversionSignalConfig
    )
    momentum_config: MomentumSignalConfig = field(default_factory=MomentumSignalConfig)
    combiner_config: RegimeCombinerConfig = field(default_factory=RegimeCombinerConfig)

    def __post_init__(self) -> None:
        """Convert nested dicts to dataclasses if needed."""
        if isinstance(self.trend_config, dict):
            self.trend_config = TrendSignalConfig(**self.trend_config)
        if isinstance(self.mean_reversion_config, dict):
            self.mean_reversion_config = MeanReversionSignalConfig(**self.mean_reversion_config)
        if isinstance(self.momentum_config, dict):
            self.momentum_config = MomentumSignalConfig(**self.momentum_config)
        if isinstance(self.combiner_config, dict):
            self.combiner_config = RegimeCombinerConfig(**self.combiner_config)


class EnsembleStrategy(BaseStrategy):
    """
    Strategy that uses an ensemble of signals and a regime-aware combiner.
    """

    def __init__(
        self,
        config: EnsembleStrategyConfig | None = None,
        signals: list[SignalGenerator] | None = None,
        combiner: SignalCombiner | None = None,
    ) -> None:
        self.config = config or EnsembleStrategyConfig()

        # Initialize default signals if not provided
        if signals is None:
            self.signals = [
                TrendSignal(self.config.trend_config),
                MeanReversionSignal(self.config.mean_reversion_config),
                MomentumSignal(self.config.momentum_config),
            ]
        else:
            self.signals = signals

        # Initialize default combiner if not provided
        if combiner is None:
            self.combiner: SignalCombiner = RegimeAwareCombiner(self.config.combiner_config)
        else:
            self.combiner = combiner

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
        candles: Sequence[Any] | None = None,
    ) -> Decision:
        """Generate trading decision using signal ensemble."""

        # Build Context
        context = StrategyContext(
            symbol=symbol,
            current_mark=current_mark,
            position_state=position_state,
            recent_marks=recent_marks,
            equity=equity,
            product=product,
            candles=candles,
        )

        # 1. Generate Signals
        raw_signals = []
        for signal_gen in self.signals:
            try:
                output = signal_gen.generate(context)
                raw_signals.append(output)
            except Exception as e:
                # Log error and continue?
                print(f"Signal generation failed: {e}")
                pass

        # 2. Combine Signals
        net_signal = self.combiner.combine(raw_signals, context)

        # 3. Determine Action
        strength = net_signal.strength
        confidence = net_signal.confidence

        action = Action.HOLD
        reason = f"Net Signal: {strength:.2f} (Conf: {confidence:.2f})"

        # Check for existing position
        has_position = position_state is not None and position_state.get("quantity", 0) != 0
        position_side = position_state.get("side", "none") if position_state else "none"

        if has_position:
            # Exit logic
            if position_side == "long":
                if strength < -self.config.close_threshold:
                    action = Action.CLOSE
                    reason = f"Signal reversed to bearish ({strength:.2f})"
                elif strength < self.config.close_threshold:
                    # Weak signal, maybe close?
                    # For now, only close if it drops below close_threshold (near zero)
                    # If close_threshold is 0.1, and strength is 0.05, we close.
                    pass
            elif position_side == "short":
                if strength > self.config.close_threshold:
                    action = Action.CLOSE
                    reason = f"Signal reversed to bullish ({strength:.2f})"

            # Stop Loss / Take Profit (Risk Overlay)
            entry_price = position_state.get("entry_price") if position_state else None
            if entry_price:
                entry = float(entry_price)
                current = float(current_mark)
                pnl_pct = (
                    (current - entry) / entry
                    if position_side == "long"
                    else (entry - current) / entry
                )

                if pnl_pct < -self.config.stop_loss_pct:
                    return Decision(
                        Action.CLOSE, f"Stop Loss ({pnl_pct:.2%})", 1.0, net_signal.metadata
                    )
                if pnl_pct > self.config.take_profit_pct:
                    return Decision(
                        Action.CLOSE, f"Take Profit ({pnl_pct:.2%})", 1.0, net_signal.metadata
                    )

        else:
            # Entry logic
            if strength > self.config.buy_threshold:
                action = Action.BUY
                reason = f"Strong Buy Signal ({strength:.2f})"
            elif strength < self.config.sell_threshold:
                action = Action.SELL
                reason = f"Strong Sell Signal ({strength:.2f})"

        return Decision(
            action=action,
            reason=reason,
            confidence=confidence,
            indicators=net_signal.metadata,
        )
