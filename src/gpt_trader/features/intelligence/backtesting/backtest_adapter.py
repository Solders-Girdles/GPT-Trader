"""
Backtest adapter for ensemble orchestration.

Provides integration between EnsembleOrchestrator and the backtesting engine,
tracking decisions, regime states, and performance metrics during backtests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ..ensemble.models import EnsembleConfig
from ..ensemble.orchestrator import EnsembleOrchestrator
from ..regime.detector import MarketRegimeDetector
from ..regime.models import RegimeConfig, RegimeType
from ..sizing.position_sizer import PositionSizer, PositionSizingConfig, SizingResult
from .batch_regime import RegimeHistory, RegimeSnapshot

if TYPE_CHECKING:
    from gpt_trader.core import Product
    from gpt_trader.features.live_trade.interfaces import TradingStrategy
    from gpt_trader.core import Decision


@dataclass
class DecisionRecord:
    """Record of an ensemble decision during backtest."""

    timestamp: datetime
    symbol: str
    price: Decimal
    action: str
    confidence: float
    reason: str

    # Regime context
    regime: str
    regime_confidence: float

    # Strategy votes
    votes: dict[str, dict[str, Any]]

    # Position sizing
    position_fraction: float | None = None
    position_value: Decimal | None = None

    # Outcome (filled in later)
    outcome_pnl: float | None = None
    is_winner: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "price": str(self.price),
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 4),
            "votes": self.votes,
            "position_fraction": (
                round(self.position_fraction, 6) if self.position_fraction else None
            ),
            "position_value": str(self.position_value) if self.position_value else None,
            "outcome_pnl": round(self.outcome_pnl, 2) if self.outcome_pnl else None,
            "is_winner": self.is_winner,
        }


@dataclass
class EnsembleBacktestResult:
    """Results from an ensemble backtest run.

    Contains decision history, regime history, and performance analytics.
    """

    # Time range
    start_time: datetime
    end_time: datetime

    # Decision history
    decisions: list[DecisionRecord] = field(default_factory=list)

    # Regime history per symbol
    regime_histories: dict[str, RegimeHistory] = field(default_factory=dict)

    # Performance by regime
    regime_performance: dict[str, dict[str, float]] = field(default_factory=dict)

    # Strategy contribution
    strategy_performance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_decision(self, record: DecisionRecord) -> None:
        """Add a decision record."""
        self.decisions.append(record)

    def record_outcome(
        self,
        symbol: str,
        timestamp: datetime,
        pnl: float,
        is_winner: bool,
    ) -> None:
        """Record the outcome of a trade.

        Finds the most recent decision for the symbol and records outcome.
        """
        # Find matching decision
        for decision in reversed(self.decisions):
            if decision.symbol == symbol and decision.outcome_pnl is None:
                decision.outcome_pnl = pnl
                decision.is_winner = is_winner
                break

    def get_decisions_by_regime(self, regime: RegimeType) -> list[DecisionRecord]:
        """Get all decisions made during a specific regime."""
        return [d for d in self.decisions if d.regime == regime.name]

    def get_win_rate_by_regime(self) -> dict[str, float]:
        """Calculate win rate for each regime.

        Returns:
            Dict mapping regime names to win rates (0-1)
        """
        regime_outcomes: dict[str, tuple[int, int]] = {}  # wins, total

        for decision in self.decisions:
            if decision.is_winner is None:
                continue

            regime = decision.regime
            wins, total = regime_outcomes.get(regime, (0, 0))
            if decision.is_winner:
                wins += 1
            total += 1
            regime_outcomes[regime] = (wins, total)

        return {
            regime: wins / total if total > 0 else 0.0
            for regime, (wins, total) in regime_outcomes.items()
        }

    def get_pnl_by_regime(self) -> dict[str, float]:
        """Calculate total PnL for each regime.

        Returns:
            Dict mapping regime names to total PnL
        """
        regime_pnl: dict[str, float] = {}

        for decision in self.decisions:
            if decision.outcome_pnl is None:
                continue

            regime = decision.regime
            regime_pnl[regime] = regime_pnl.get(regime, 0.0) + decision.outcome_pnl

        return regime_pnl

    def get_strategy_hit_rate(self) -> dict[str, float]:
        """Calculate how often each strategy's vote matched the final action.

        Returns:
            Dict mapping strategy names to hit rates (0-1)
        """
        strategy_matches: dict[str, tuple[int, int]] = {}  # matches, total

        for decision in self.decisions:
            final_action = decision.action

            for strategy_name, vote_data in decision.votes.items():
                vote_action = vote_data.get("action", "HOLD")
                matches, total = strategy_matches.get(strategy_name, (0, 0))

                if vote_action == final_action:
                    matches += 1
                total += 1
                strategy_matches[strategy_name] = (matches, total)

        return {
            strategy: matches / total if total > 0 else 0.0
            for strategy, (matches, total) in strategy_matches.items()
        }

    def summary(self) -> dict[str, Any]:
        """Get comprehensive summary of backtest results."""
        total_decisions = len(self.decisions)
        completed_trades = sum(1 for d in self.decisions if d.outcome_pnl is not None)
        winning_trades = sum(1 for d in self.decisions if d.is_winner is True)

        total_pnl = sum(d.outcome_pnl for d in self.decisions if d.outcome_pnl is not None)

        return {
            "time_range": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
            },
            "decisions": {
                "total": total_decisions,
                "completed_trades": completed_trades,
                "winning_trades": winning_trades,
                "win_rate": winning_trades / completed_trades if completed_trades > 0 else 0.0,
            },
            "pnl": {
                "total": round(total_pnl, 2),
                "by_regime": {k: round(v, 2) for k, v in self.get_pnl_by_regime().items()},
            },
            "win_rate_by_regime": {
                k: round(v, 4) for k, v in self.get_win_rate_by_regime().items()
            },
            "strategy_hit_rate": {k: round(v, 4) for k, v in self.get_strategy_hit_rate().items()},
            "regime_distribution": {
                symbol: history.get_regime_distribution()
                for symbol, history in self.regime_histories.items()
            },
        }


class EnsembleBacktestAdapter:
    """Adapter for running ensemble orchestrator in backtests.

    Wraps the EnsembleOrchestrator to provide:
    - Decision tracking and history
    - Regime history recording
    - Position sizing integration
    - Outcome recording for performance analysis

    Example:
        # Create strategies
        strategies = {
            "baseline": BaselineStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }

        # Create adapter
        adapter = EnsembleBacktestAdapter(strategies=strategies)

        # Run backtest
        async for bar_time, bars, quotes in runner.run():
            for symbol, bar in bars.items():
                decision, sizing = adapter.process_bar(
                    symbol=symbol,
                    price=bar.close,
                    timestamp=bar_time,
                    equity=Decimal("10000"),
                    position_state=None,
                    recent_marks=[...],
                    product=None,
                )

                if decision.action != "HOLD":
                    # Execute trade...
                    pass

        # Record outcomes
        adapter.record_trade_outcome("BTC-USD", pnl=100.0, is_winner=True)

        # Get results
        results = adapter.get_results()
        print(results.summary())
    """

    def __init__(
        self,
        strategies: dict[str, TradingStrategy],
        ensemble_config: EnsembleConfig | None = None,
        regime_config: RegimeConfig | None = None,
        sizing_config: PositionSizingConfig | None = None,
        enable_sizing: bool = True,
    ):
        """Initialize backtest adapter.

        Args:
            strategies: Dictionary of strategy instances
            ensemble_config: Ensemble configuration
            regime_config: Regime detection configuration
            sizing_config: Position sizing configuration
            enable_sizing: Whether to calculate position sizes
        """
        # Create regime detector
        self.regime_detector = MarketRegimeDetector(regime_config or RegimeConfig())

        # Create ensemble orchestrator
        self.orchestrator = EnsembleOrchestrator(
            strategies=strategies,
            regime_detector=self.regime_detector,
            config=ensemble_config or EnsembleConfig(),
            regime_config=regime_config,
        )

        # Create position sizer
        self.enable_sizing = enable_sizing
        if enable_sizing:
            self.sizer: PositionSizer | None = PositionSizer(
                regime_detector=self.regime_detector,
                config=sizing_config or PositionSizingConfig(),
            )
        else:
            self.sizer = None

        # Results tracking
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        self._decisions: list[DecisionRecord] = []
        self._regime_snapshots: dict[str, list[RegimeSnapshot]] = {}

    def process_bar(
        self,
        symbol: str,
        price: Decimal,
        timestamp: datetime,
        equity: Decimal,
        position_state: dict[str, Any] | None = None,
        recent_marks: Sequence[Decimal] | None = None,
        product: Product | None = None,
    ) -> tuple[Decision, SizingResult | None]:
        """Process a single bar and generate ensemble decision.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Bar timestamp
            equity: Account equity
            position_state: Current position info
            recent_marks: Recent price history
            product: Product specification

        Returns:
            Tuple of (decision, sizing_result or None)
        """
        # Track time range
        if self._start_time is None:
            self._start_time = timestamp
        self._end_time = timestamp

        # Convert recent_marks to list if needed
        if recent_marks is None:
            recent_marks = [price]

        # Get ensemble decision
        decision = self.orchestrator.decide(
            symbol=symbol,
            current_mark=price,
            position_state=position_state,
            recent_marks=list(recent_marks),
            equity=equity,
            product=product,
        )

        # Calculate position size if enabled
        sizing_result = None
        if self.enable_sizing and self.sizer is not None:
            sizing_result = self.sizer.calculate_size(
                symbol=symbol,
                current_price=price,
                equity=equity,
                decision_confidence=decision.confidence,
            )

        # Get regime state
        regime_state = self.orchestrator.get_regime(symbol)

        # Extract votes from decision metadata
        votes = {}
        ensemble_meta = decision.indicators.get("ensemble", {})
        for vote_data in ensemble_meta.get("votes", []):
            strategy_name = vote_data.get("strategy_name", "unknown")
            votes[strategy_name] = vote_data

        # Record decision
        record = DecisionRecord(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            action=decision.action.value,
            confidence=decision.confidence,
            reason=decision.reason,
            regime=regime_state.regime.name,
            regime_confidence=regime_state.confidence,
            votes=votes,
            position_fraction=sizing_result.position_fraction if sizing_result else None,
            position_value=sizing_result.position_value if sizing_result else None,
        )
        self._decisions.append(record)

        # Record regime snapshot
        if symbol not in self._regime_snapshots:
            self._regime_snapshots[symbol] = []

        snapshot = RegimeSnapshot(
            timestamp=timestamp,
            price=price,
            regime=regime_state.regime,
            confidence=regime_state.confidence,
            volatility_percentile=regime_state.volatility_percentile,
            trend_percentile=(regime_state.trend_score + 1.0) / 2.0,
        )
        self._regime_snapshots[symbol].append(snapshot)

        return decision, sizing_result

    def record_trade_outcome(
        self,
        symbol: str,
        pnl: float,
        is_winner: bool,
    ) -> None:
        """Record the outcome of a completed trade.

        Args:
            symbol: Trading symbol
            pnl: Profit/loss amount
            is_winner: Whether trade was profitable
        """
        # Find most recent unresolved decision for symbol
        for decision in reversed(self._decisions):
            if decision.symbol == symbol and decision.outcome_pnl is None:
                decision.outcome_pnl = pnl
                decision.is_winner = is_winner

                # Also record for adaptive learning
                self.orchestrator.record_trade_outcome(
                    symbol=symbol,
                    is_success=is_winner,
                    pnl=pnl,
                )
                break

    def get_results(self) -> EnsembleBacktestResult:
        """Get backtest results.

        Returns:
            EnsembleBacktestResult with full history and analytics
        """
        # Build regime histories
        regime_histories = {}
        for symbol, snapshots in self._regime_snapshots.items():
            history = RegimeHistory(symbol=symbol, snapshots=snapshots)
            regime_histories[symbol] = history

        # Get strategy performance from orchestrator
        strategy_performance = self.orchestrator.get_strategy_performance()

        result = EnsembleBacktestResult(
            start_time=self._start_time or datetime.now(),
            end_time=self._end_time or datetime.now(),
            decisions=self._decisions,
            regime_histories=regime_histories,
            strategy_performance=strategy_performance,
        )

        return result

    def reset(self) -> None:
        """Reset adapter state for a new backtest."""
        self._start_time = None
        self._end_time = None
        self._decisions = []
        self._regime_snapshots = {}

        # Reset underlying components
        self.regime_detector = MarketRegimeDetector(
            self.regime_detector._config
            if hasattr(self.regime_detector, "_config")
            else RegimeConfig()
        )

        # Recreate orchestrator with fresh detector
        config = self.orchestrator.config
        strategies = self.orchestrator.strategies
        self.orchestrator = EnsembleOrchestrator(
            strategies=strategies,
            regime_detector=self.regime_detector,
            config=config,
        )

        if self.sizer is not None:
            self.sizer.regime_detector = self.regime_detector


__all__ = [
    "DecisionRecord",
    "EnsembleBacktestAdapter",
    "EnsembleBacktestResult",
]
