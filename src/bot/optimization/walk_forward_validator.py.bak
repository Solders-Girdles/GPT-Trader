"""
Enhanced walk-forward testing for robust strategy validation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from bot.backtest.engine_portfolio import run_backtest
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward testing."""

    # Window settings
    train_months: int = Field(12, description="Training window in months")
    test_months: int = Field(6, description="Test window in months")
    step_months: int = Field(6, description="Step between windows in months")

    # Validation criteria
    min_windows: int = Field(3, description="Minimum number of windows required")
    min_test_trades: int = Field(10, description="Minimum trades per test window")
    min_test_sharpe: float = Field(0.0, description="Minimum Sharpe ratio per test window")

    # Performance thresholds
    min_mean_sharpe: float = Field(0.5, description="Minimum mean Sharpe across windows")
    max_sharpe_std: float = Field(0.8, description="Maximum Sharpe standard deviation")
    max_mean_drawdown: float = Field(0.15, description="Maximum mean drawdown")

    # Data settings
    symbols: list[str] = Field(default_factory=list)
    regime_on: bool = Field(True, description="Enable regime filtering")
    regime_symbol: str = Field("SPY", description="Regime symbol")
    regime_window: int = Field(200, description="Regime window")


class WalkForwardWindow(BaseModel):
    """A single walk-forward window."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_index: int


class WalkForwardResult(BaseModel):
    """Results from a single walk-forward window."""

    window: WalkForwardWindow
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_trades: int
    test_trades: int
    regime_coverage: float  # Percentage of test period in favorable regime


class WalkForwardValidator:
    """Enhanced walk-forward testing validator."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config
        self.windows: list[WalkForwardWindow] = []
        self.results: list[WalkForwardResult] = []

    def create_windows(self, start_date: datetime, end_date: datetime) -> None:
        """Create walk-forward windows."""
        logger.info(f"Creating walk-forward windows from {start_date.date()} to {end_date.date()}")

        current_start = start_date
        window_index = 0

        while True:
            # Calculate window boundaries
            train_end = self._add_months(current_start, self.config.train_months) - timedelta(
                days=1
            )
            test_start = train_end + timedelta(days=1)
            test_end = self._add_months(test_start, self.config.test_months) - timedelta(days=1)

            # Stop if test window extends beyond data
            if test_start > end_date:
                break

            # Clip test end to data end
            if test_end > end_date:
                test_end = end_date

            # Create window
            window = WalkForwardWindow(
                train_start=current_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_index=window_index,
            )

            self.windows.append(window)

            # Move to next window
            current_start = self._add_months(current_start, self.config.step_months)
            window_index += 1

            # Safety check
            if current_start > end_date:
                break

        logger.info(f"Created {len(self.windows)} walk-forward windows")

    def _add_months(self, date: datetime, months: int) -> datetime:
        """Add months to a date."""
        year = date.year + (date.month + months - 1) // 12
        month = (date.month + months - 1) % 12 + 1
        day = min(date.day, self._days_in_month(year, month))
        return datetime(year, month, day)

    def _days_in_month(self, year: int, month: int) -> int:
        """Get number of days in a month."""
        if month == 12:
            return 31
        return (datetime(year, month + 1, 1) - datetime(year, month, 1)).days

    def validate_strategy(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Validate a strategy using walk-forward testing."""
        logger.info(f"Validating strategy with parameters: {parameters}")

        # Create strategy
        strategy = TrendBreakoutStrategy(TrendBreakoutParams(**parameters))

        # Create portfolio rules
        rules = PortfolioRules(
            per_trade_risk_pct=0.005,  # 0.5%
            atr_k=parameters.get("atr_k", 2.0),
            max_positions=10,
            cost_bps=5.0,
        )

        # Validate each window
        window_results = []
        for window in self.windows:
            try:
                result = self._validate_window(strategy, rules, window)
                if result:
                    window_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to validate window {window.window_index}: {e}")

        # Analyze results
        return self._analyze_walk_forward_results(window_results)

    def _validate_window(
        self, strategy, rules, window: WalkForwardWindow
    ) -> WalkForwardResult | None:
        """Validate a single walk-forward window."""
        try:
            # Train period backtest
            train_result = run_backtest(
                symbol=None,
                symbol_list_csv=None,
                start=window.train_start,
                end=window.train_end,
                strategy=strategy,
                rules=rules,
                regime_on=self.config.regime_on,
                regime_symbol=self.config.regime_symbol,
                regime_window=self.config.regime_window,
                return_summary=True,
                quiet_mode=True,
                write_portfolio_csv=False,
                write_trades_csv=False,
                write_summary_csv=False,
                make_plot=False,
            )

            # Test period backtest
            test_result = run_backtest(
                symbol=None,
                symbol_list_csv=None,
                start=window.test_start,
                end=window.test_end,
                strategy=strategy,
                rules=rules,
                regime_on=self.config.regime_on,
                regime_symbol=self.config.regime_symbol,
                regime_window=self.config.regime_window,
                return_summary=True,
                quiet_mode=True,
                write_portfolio_csv=False,
                write_trades_csv=False,
                write_summary_csv=False,
                make_plot=False,
            )

            if not train_result or not test_result:
                return None

            # Calculate regime coverage
            regime_coverage = self._calculate_regime_coverage(window.test_start, window.test_end)

            return WalkForwardResult(
                window=window,
                train_metrics=train_result["summary"],
                test_metrics=test_result["summary"],
                train_trades=train_result["summary"].get("n_trades", 0),
                test_trades=test_result["summary"].get("n_trades", 0),
                regime_coverage=regime_coverage,
            )

        except Exception as e:
            logger.warning(f"Window validation failed: {e}")
            return None

    def _calculate_regime_coverage(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate percentage of period in favorable regime."""
        try:
            # This would need to be implemented based on your regime calculation
            # For now, return a placeholder
            return 0.8  # 80% in favorable regime
        except Exception:
            return 0.5  # Default to 50%

    def _analyze_walk_forward_results(self, results: list[WalkForwardResult]) -> dict[str, Any]:
        """Analyze walk-forward results."""
        if not results:
            return {"valid": False, "error": "No valid results", "n_windows": 0}

        # Extract test metrics
        test_sharpes = [r.test_metrics.get("sharpe", 0) for r in results]
        test_drawdowns = [r.test_metrics.get("max_drawdown", 0) for r in results]
        test_trades = [r.test_trades for r in results]
        regime_coverages = [r.regime_coverage for r in results]

        # Calculate statistics
        mean_sharpe = np.mean(test_sharpes)
        sharpe_std = np.std(test_sharpes)
        mean_drawdown = np.mean(test_drawdowns)
        mean_trades = np.mean(test_trades)
        mean_regime_coverage = np.mean(regime_coverages)

        # Calculate consistency metrics
        positive_sharpe_windows = sum(1 for s in test_sharpes if s > 0)
        sharpe_consistency = positive_sharpe_windows / len(test_sharpes)

        # Determine if strategy is valid
        valid = (
            len(results) >= self.config.min_windows
            and mean_sharpe >= self.config.min_mean_sharpe
            and sharpe_std <= self.config.max_sharpe_std
            and mean_drawdown <= self.config.max_mean_drawdown
            and mean_trades >= self.config.min_test_trades
        )

        return {
            "valid": valid,
            "n_windows": len(results),
            "mean_sharpe": mean_sharpe,
            "sharpe_std": sharpe_std,
            "mean_drawdown": mean_drawdown,
            "mean_trades": mean_trades,
            "mean_regime_coverage": mean_regime_coverage,
            "sharpe_consistency": sharpe_consistency,
            "positive_sharpe_windows": positive_sharpe_windows,
            "window_results": [
                {
                    "window_index": r.window.window_index,
                    "test_sharpe": r.test_metrics.get("sharpe", 0),
                    "test_drawdown": r.test_metrics.get("max_drawdown", 0),
                    "test_trades": r.test_trades,
                    "regime_coverage": r.regime_coverage,
                }
                for r in results
            ],
        }

    def validate_optimization_results(self, results_path: str, output_path: str) -> None:
        """Validate optimization results using walk-forward testing."""
        logger.info(f"Validating optimization results from {results_path}")

        # Load results
        results_df = pd.read_csv(results_path)

        # Create windows if not already created
        if not self.windows:
            start_date = datetime.strptime(
                results_df.get("start_date", "2020-01-01").iloc[0], "%Y-%m-%d"
            )
            end_date = datetime.strptime(
                results_df.get("end_date", "2024-01-01").iloc[0], "%Y-%m-%d"
            )
            self.create_windows(start_date, end_date)

        # Validate top strategies
        validated_results = []
        top_strategies = results_df.nlargest(50, "sharpe")  # Top 50 by Sharpe

        for idx, row in top_strategies.iterrows():
            try:
                # Extract parameters
                parameters = {}
                for col in row.index:
                    if col.startswith("param_"):
                        param_name = col.replace("param_", "")
                        parameters[param_name] = row[col]

                if not parameters:
                    continue

                # Validate strategy
                validation_result = self.validate_strategy(parameters)

                # Combine original results with validation
                combined_result = {
                    **row.to_dict(),
                    "walk_forward_valid": validation_result["valid"],
                    "wf_n_windows": validation_result["n_windows"],
                    "wf_mean_sharpe": validation_result["mean_sharpe"],
                    "wf_sharpe_std": validation_result["sharpe_std"],
                    "wf_mean_drawdown": validation_result["mean_drawdown"],
                    "wf_sharpe_consistency": validation_result["sharpe_consistency"],
                    "wf_validation_result": validation_result,
                }

                validated_results.append(combined_result)

                logger.info(
                    f"Validated strategy {idx}: WF valid={validation_result['valid']}, "
                    f"mean_sharpe={validation_result['mean_sharpe']:.3f}"
                )

            except Exception as e:
                logger.warning(f"Failed to validate strategy {idx}: {e}")

        # Save validated results
        validated_df = pd.DataFrame(validated_results)
        validated_df.to_csv(output_path, index=False)

        # Generate summary
        valid_strategies = validated_df[validated_df["walk_forward_valid"] is True]
        logger.info("Walk-forward validation complete:")
        logger.info(f"  Total strategies tested: {len(validated_results)}")
        logger.info(f"  Valid strategies: {len(valid_strategies)}")
        logger.info(f"  Results saved to: {output_path}")

        if len(valid_strategies) > 0:
            logger.info(f"  Best validated Sharpe: {valid_strategies['wf_mean_sharpe'].max():.3f}")
            logger.info(
                f"  Best validated consistency: {valid_strategies['wf_sharpe_consistency'].max():.3f}"
            )


def run_walk_forward_validation(
    optimization_results_path: str, config: WalkForwardConfig, output_path: str = None
) -> None:
    """Run walk-forward validation on optimization results."""
    if output_path is None:
        output_path = optimization_results_path.replace(".csv", "_wf_validated.csv")

    validator = WalkForwardValidator(config)
    validator.validate_optimization_results(optimization_results_path, output_path)


if __name__ == "__main__":
    # Example usage
    config = WalkForwardConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        train_months=12,
        test_months=6,
        step_months=6,
        min_windows=3,
        min_mean_sharpe=0.5,
        max_sharpe_std=0.8,
    )

    run_walk_forward_validation(
        optimization_results_path="data/optimization/all_results.csv", config=config
    )
