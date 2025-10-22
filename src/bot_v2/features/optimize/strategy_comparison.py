"""
Strategy comparison framework for validation.

Compares baseline vs enhanced strategies across:
- Different market regimes (trend/range/high-vol)
- Out-of-sample periods (using purged CV)
- Multiple performance metrics

Used to validate Phase 2 exit criteria: enhanced strategy must beat baseline.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from bot_v2.features.live_trade.strategies.enhanced_strategy import EnhancedStrategy
from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.utilities.logging_patterns import get_logger

from .backtest_engine import BacktestEngine
from .purged_cv import AnchoredWalkForwardCV, CVSplit
from .regime_detection import MarketRegime, RegimeDetector, split_data_by_regime
from .types_v2 import BacktestConfig, BacktestResult

logger = get_logger(__name__, component="strategy_comparison")


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""

    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float  # Return / Max DD

    def score(self) -> float:
        """
        Calculate composite score for ranking strategies.

        Higher is better. Weights:
        - Sharpe ratio: 40%
        - Total return: 30%
        - Max drawdown: 20% (negative)
        - Win rate: 10%
        """
        sharpe_component = self.sharpe_ratio * 0.4
        return_component = self.total_return * 0.3
        dd_component = -self.max_drawdown * 0.2  # Negative (lower DD is better)
        winrate_component = self.win_rate * 0.1

        return sharpe_component + return_component + dd_component + winrate_component


@dataclass
class ComparisonResult:
    """Result of comparing two strategies."""

    baseline_performance: StrategyPerformance
    enhanced_performance: StrategyPerformance
    regime: str | None = None  # None = full dataset
    split_id: int | None = None  # None = full dataset, else CV split number

    def enhanced_wins(self) -> bool:
        """Check if enhanced strategy beats baseline on composite score."""
        return self.enhanced_performance.score() > self.baseline_performance.score()

    def improvement_pct(self) -> dict[str, float]:
        """Calculate improvement percentage for each metric."""
        return {
            "return": self._pct_change(
                self.baseline_performance.total_return,
                self.enhanced_performance.total_return,
            ),
            "sharpe": self._pct_change(
                self.baseline_performance.sharpe_ratio,
                self.enhanced_performance.sharpe_ratio,
            ),
            "max_dd": self._pct_change(
                self.baseline_performance.max_drawdown,
                self.enhanced_performance.max_drawdown,
            ),
            "win_rate": self._pct_change(
                self.baseline_performance.win_rate,
                self.enhanced_performance.win_rate,
            ),
        }

    def _pct_change(self, baseline: float, enhanced: float) -> float:
        """Calculate percentage change from baseline to enhanced."""
        if baseline == 0:
            return 0.0
        return ((enhanced - baseline) / abs(baseline)) * 100


class StrategyComparator:
    """
    Compares baseline vs enhanced strategy performance.

    Validates Phase 2 exit criteria:
    - Enhanced must beat baseline across all 3 regimes
    - Enhanced must beat baseline on out-of-sample data
    """

    def __init__(
        self,
        *,
        baseline_strategy: BaselinePerpsStrategy,
        enhanced_strategy: EnhancedStrategy,
        backtest_config: BacktestConfig | None = None,
    ):
        """
        Initialize comparator.

        Args:
            baseline_strategy: Baseline MA crossover strategy
            enhanced_strategy: Enhanced strategy with filters
            backtest_config: Backtest configuration
        """
        self.baseline_strategy = baseline_strategy
        self.enhanced_strategy = enhanced_strategy
        self.backtest_config = backtest_config or BacktestConfig()

    def compare_full_dataset(
        self, *, data: pd.DataFrame, symbol: str
    ) -> ComparisonResult:
        """
        Compare strategies on full dataset.

        Args:
            data: Historical OHLC data
            symbol: Trading symbol

        Returns:
            ComparisonResult
        """
        logger.info("Comparing strategies on full dataset | symbol=%s | bars=%d", symbol, len(data))

        baseline_perf = self._run_backtest(
            strategy=self.baseline_strategy,
            data=data,
            symbol=symbol,
            strategy_name="Baseline",
        )

        enhanced_perf = self._run_backtest(
            strategy=self.enhanced_strategy,
            data=data,
            symbol=symbol,
            strategy_name="Enhanced",
        )

        result = ComparisonResult(
            baseline_performance=baseline_perf,
            enhanced_performance=enhanced_perf,
        )

        logger.info(
            "Full dataset comparison | enhanced_wins=%s | improvement=%.2f%%",
            result.enhanced_wins(),
            result.improvement_pct()["sharpe"],
        )

        return result

    def compare_by_regime(
        self, *, data: pd.DataFrame, symbol: str, detector: RegimeDetector | None = None
    ) -> dict[str, ComparisonResult]:
        """
        Compare strategies across different market regimes.

        Args:
            data: Historical OHLC data
            symbol: Trading symbol
            detector: Regime detector (creates default if None)

        Returns:
            Dict mapping regime name -> ComparisonResult
        """
        logger.info("Comparing strategies by regime | symbol=%s", symbol)

        # Split data by regime
        regime_data = split_data_by_regime(data, detector)

        results: dict[str, ComparisonResult] = {}

        for regime, regime_df in regime_data.items():
            logger.info(
                "Running regime comparison | regime=%s | bars=%d",
                regime.value,
                len(regime_df),
            )

            if len(regime_df) < 50:
                logger.warning("Skipping regime %s (insufficient data: %d bars)", regime.value, len(regime_df))
                continue

            baseline_perf = self._run_backtest(
                strategy=self.baseline_strategy,
                data=regime_df,
                symbol=symbol,
                strategy_name=f"Baseline-{regime.value}",
            )

            enhanced_perf = self._run_backtest(
                strategy=self.enhanced_strategy,
                data=regime_df,
                symbol=symbol,
                strategy_name=f"Enhanced-{regime.value}",
            )

            result = ComparisonResult(
                baseline_performance=baseline_perf,
                enhanced_performance=enhanced_perf,
                regime=regime.value,
            )

            results[regime.value] = result

            logger.info(
                "Regime %s | enhanced_wins=%s | sharpe_improvement=%.2f%%",
                regime.value,
                result.enhanced_wins(),
                result.improvement_pct()["sharpe"],
            )

        return results

    def compare_walk_forward(
        self, *, data: pd.DataFrame, symbol: str, n_splits: int = 5
    ) -> list[ComparisonResult]:
        """
        Compare strategies using purged walk-forward CV.

        Args:
            data: Historical OHLC data
            symbol: Trading symbol
            n_splits: Number of CV splits

        Returns:
            List of ComparisonResult (one per split)
        """
        logger.info("Comparing strategies with walk-forward CV | symbol=%s | n_splits=%d", symbol, n_splits)

        cv = AnchoredWalkForwardCV(n_splits=n_splits, embargo_pct=0.02)
        splits = cv.split(data)

        results: list[ComparisonResult] = []

        for split in splits:
            test_data = split.get_test_data(data)

            logger.info(
                "Running CV split %d | test_size=%d | test_range=[%d:%d]",
                split.split_id,
                split.test_size,
                split.test_start_idx,
                split.test_end_idx,
            )

            baseline_perf = self._run_backtest(
                strategy=self.baseline_strategy,
                data=test_data,
                symbol=symbol,
                strategy_name=f"Baseline-Split{split.split_id}",
            )

            enhanced_perf = self._run_backtest(
                strategy=self.enhanced_strategy,
                data=test_data,
                symbol=symbol,
                strategy_name=f"Enhanced-Split{split.split_id}",
            )

            result = ComparisonResult(
                baseline_performance=baseline_perf,
                enhanced_performance=enhanced_perf,
                split_id=split.split_id,
            )

            results.append(result)

            logger.info(
                "Split %d | enhanced_wins=%s | sharpe_improvement=%.2f%%",
                split.split_id,
                result.enhanced_wins(),
                result.improvement_pct()["sharpe"],
            )

        return results

    def _run_backtest(
        self,
        *,
        strategy: BaselinePerpsStrategy | EnhancedStrategy,
        data: pd.DataFrame,
        symbol: str,
        strategy_name: str,
    ) -> StrategyPerformance:
        """Run backtest and extract performance metrics."""
        # Reset strategy state before each run
        strategy.reset()

        engine = BacktestEngine(strategy=strategy, config=self.backtest_config)
        result = engine.run(data=data, symbol=symbol)

        return self._extract_performance(result, strategy_name)

    def _extract_performance(
        self, result: BacktestResult, strategy_name: str
    ) -> StrategyPerformance:
        """Extract performance metrics from backtest result."""
        metrics = result.metrics if result.metrics else None

        if metrics is None:
            # No metrics (no trades or error)
            return StrategyPerformance(
                strategy_name=strategy_name,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_return=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
            )

        total_return = metrics.total_return
        sharpe_ratio = metrics.sharpe_ratio
        max_drawdown = metrics.max_drawdown
        win_rate = metrics.win_rate
        total_trades = metrics.total_trades
        volatility = metrics.volatility

        avg_trade_return = total_return / total_trades if total_trades > 0 else 0.0
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        return StrategyPerformance(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
        )

    def generate_report(
        self,
        *,
        full_result: ComparisonResult | None = None,
        regime_results: dict[str, ComparisonResult] | None = None,
        cv_results: list[ComparisonResult] | None = None,
    ) -> str:
        """
        Generate comprehensive comparison report.

        Args:
            full_result: Full dataset comparison
            regime_results: Regime-specific comparisons
            cv_results: Walk-forward CV results

        Returns:
            Report string
        """
        report = "=" * 80 + "\n"
        report += "STRATEGY COMPARISON REPORT\n"
        report += "=" * 80 + "\n\n"

        # Full dataset
        if full_result:
            report += "FULL DATASET\n"
            report += "-" * 80 + "\n"
            report += self._format_comparison(full_result)
            report += "\n"

        # Regime comparisons
        if regime_results:
            report += "=" * 80 + "\n"
            report += "REGIME-SPECIFIC PERFORMANCE\n"
            report += "=" * 80 + "\n\n"

            for regime, result in regime_results.items():
                report += f"Regime: {regime.upper()}\n"
                report += "-" * 80 + "\n"
                report += self._format_comparison(result)
                report += "\n"

            # Summary
            wins = sum(1 for r in regime_results.values() if r.enhanced_wins())
            total = len(regime_results)
            report += f"Enhanced wins {wins}/{total} regimes\n\n"

        # CV results
        if cv_results:
            report += "=" * 80 + "\n"
            report += "WALK-FORWARD CROSS-VALIDATION\n"
            report += "=" * 80 + "\n\n"

            for result in cv_results:
                report += f"Split {result.split_id}\n"
                report += "-" * 40 + "\n"
                report += self._format_comparison(result, compact=True)
                report += "\n"

            # Summary
            wins = sum(1 for r in cv_results if r.enhanced_wins())
            total = len(cv_results)
            avg_sharpe_improvement = sum(r.improvement_pct()["sharpe"] for r in cv_results) / total
            report += f"Enhanced wins {wins}/{total} splits\n"
            report += f"Average Sharpe improvement: {avg_sharpe_improvement:.2f}%\n\n"

        # Phase 2 exit criteria
        report += "=" * 80 + "\n"
        report += "PHASE 2 EXIT CRITERIA CHECK\n"
        report += "=" * 80 + "\n\n"

        all_passed = True

        if regime_results:
            regime_wins = sum(1 for r in regime_results.values() if r.enhanced_wins())
            regime_total = len(regime_results)
            regime_passed = regime_wins == regime_total

            report += f"✓ Regime validation: {regime_wins}/{regime_total} regimes\n"
            if not regime_passed:
                report += "  ❌ FAILED: Enhanced must beat baseline in ALL regimes\n"
                all_passed = False
            else:
                report += "  ✅ PASSED: Enhanced beats baseline across all regimes\n"

        if cv_results:
            cv_wins = sum(1 for r in cv_results if r.enhanced_wins())
            cv_total = len(cv_results)
            cv_passed = cv_wins >= cv_total * 0.6  # 60% threshold

            report += f"✓ Out-of-sample validation: {cv_wins}/{cv_total} splits\n"
            if not cv_passed:
                report += "  ❌ FAILED: Enhanced must win >=60% of CV splits\n"
                all_passed = False
            else:
                report += "  ✅ PASSED: Enhanced wins majority of out-of-sample periods\n"

        report += "\n"
        if all_passed:
            report += "✅ ALL CHECKS PASSED - Enhanced strategy validated\n"
        else:
            report += "❌ SOME CHECKS FAILED - Review and adjust strategy\n"

        report += "=" * 80 + "\n"

        return report

    def _format_comparison(self, result: ComparisonResult, compact: bool = False) -> str:
        """Format comparison result as text."""
        baseline = result.baseline_performance
        enhanced = result.enhanced_performance
        improvements = result.improvement_pct()

        if compact:
            return (
                f"  Baseline: Return={baseline.total_return:.2%}, Sharpe={baseline.sharpe_ratio:.2f}\n"
                f"  Enhanced: Return={enhanced.total_return:.2%}, Sharpe={enhanced.sharpe_ratio:.2f}\n"
                f"  Winner: {'Enhanced ✅' if result.enhanced_wins() else 'Baseline ❌'}\n"
            )

        output = ""
        output += f"{'Metric':<20} {'Baseline':<15} {'Enhanced':<15} {'Improvement':<15}\n"
        output += "-" * 65 + "\n"
        output += f"{'Total Return':<20} {baseline.total_return:>14.2%} {enhanced.total_return:>14.2%} {improvements['return']:>14.2f}%\n"
        output += f"{'Sharpe Ratio':<20} {baseline.sharpe_ratio:>14.2f} {enhanced.sharpe_ratio:>14.2f} {improvements['sharpe']:>14.2f}%\n"
        output += f"{'Max Drawdown':<20} {baseline.max_drawdown:>14.2%} {enhanced.max_drawdown:>14.2%} {improvements['max_dd']:>14.2f}%\n"
        output += f"{'Win Rate':<20} {baseline.win_rate:>14.2%} {enhanced.win_rate:>14.2%} {improvements['win_rate']:>14.2f}%\n"
        output += f"{'Total Trades':<20} {baseline.total_trades:>14} {enhanced.total_trades:>14}\n"
        output += f"{'Calmar Ratio':<20} {baseline.calmar_ratio:>14.2f} {enhanced.calmar_ratio:>14.2f}\n"
        output += "\n"
        output += f"Winner: {'Enhanced ✅' if result.enhanced_wins() else 'Baseline ❌'}\n"

        return output


__all__ = ["StrategyComparator", "StrategyPerformance", "ComparisonResult"]
