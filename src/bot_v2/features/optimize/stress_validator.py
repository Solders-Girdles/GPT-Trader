"""
Stress test validation framework for derivatives backtesting.

Orchestrates stress scenarios, validates results against exit criteria,
and generates comprehensive stress test reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import pandas as pd

from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.utilities.logging_patterns import get_logger

from .backtest_engine import BacktestEngine
from .backtest_portfolio_derivatives import DerivativesBacktestPortfolio
from .stress_scenarios import (
    FlashCrashScenario,
    FundingShockScenario,
    GapMoveScenario,
    HighVolatilityScenario,
    LiquidityCrisisScenario,
    StressScenarioType,
)
from .types_v2 import BacktestConfig, BacktestResult

logger = get_logger(__name__, component="optimize")


class StressTestStatus(Enum):
    """Status of stress test validation."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class StressTestCriteria:
    """Criteria for stress test validation."""

    # Liquidation criteria
    max_liquidations_allowed: int = 0  # Must avoid liquidation
    max_liquidation_warnings: int | None = None  # Optional warning threshold

    # Drawdown criteria
    max_drawdown_pct: float = 0.30  # Max 30% drawdown allowed
    max_drawdown_warning_pct: float = 0.20  # Warning at 20%

    # Funding cost criteria
    max_funding_cost_pct_of_pnl: float | None = None  # e.g., 0.10 = 10% of P&L

    # Margin criteria
    max_margin_utilization: float = 0.90  # Max 90% margin utilization
    max_leverage: float = 10.0  # Max 10x leverage

    # Performance criteria (optional)
    min_sharpe_ratio: float | None = None
    max_negative_return_pct: float | None = None  # e.g., -0.50 = lose max 50%


@dataclass
class StressTestResult:
    """Result of a single stress test."""

    scenario_name: str
    scenario_type: StressScenarioType
    backtest_result: BacktestResult
    status: StressTestStatus
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == StressTestStatus.PASSED

    def summary(self) -> str:
        """Generate summary string."""
        status_emoji = {"passed": "✅", "failed": "❌", "warning": "⚠️"}[self.status.value]

        summary = f"{status_emoji} {self.scenario_name} ({self.scenario_type.value})\n"
        summary += f"   Status: {self.status.value.upper()}\n"

        if self.metrics:
            summary += "   Metrics:\n"
            summary += f"      Total Return: {self.metrics.get('total_return', 0):.2%}\n"
            summary += f"      Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}\n"
            summary += f"      Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}\n"
            summary += f"      Liquidations: {self.metrics.get('liquidations', 0)}\n"

        if self.failures:
            summary += "   Failures:\n"
            for failure in self.failures:
                summary += f"      - {failure}\n"

        if self.warnings:
            summary += "   Warnings:\n"
            for warning in self.warnings:
                summary += f"      - {warning}\n"

        return summary


class StressTestValidator:
    """
    Orchestrates stress testing and validation.

    Runs multiple stress scenarios, validates results against criteria,
    and generates comprehensive reports.
    """

    def __init__(
        self,
        *,
        strategy: BaselinePerpsStrategy,
        criteria: StressTestCriteria | None = None,
        backtest_config: BacktestConfig | None = None,
    ):
        """
        Initialize stress test validator.

        Args:
            strategy: Production strategy to test
            criteria: Validation criteria
            backtest_config: Backtest configuration
        """
        self.strategy = strategy
        self.criteria = criteria or StressTestCriteria()
        self.backtest_config = backtest_config or BacktestConfig()
        self.results: list[StressTestResult] = []

    def run_stress_test(
        self,
        *,
        scenario_name: str,
        scenario_type: StressScenarioType,
        data: pd.DataFrame,
        symbol: str,
        scenario_generator: Any,
    ) -> StressTestResult:
        """
        Run a single stress test scenario.

        Args:
            scenario_name: Name for this test
            scenario_type: Type of scenario
            data: Historical data
            symbol: Trading symbol
            scenario_generator: Scenario generator instance

        Returns:
            StressTestResult
        """
        logger.info("Running stress test | scenario=%s | type=%s", scenario_name, scenario_type.value)

        # Apply scenario to data
        stressed_data = scenario_generator.apply(data)

        # Create derivatives-enabled portfolio
        portfolio = DerivativesBacktestPortfolio(
            initial_capital=self.backtest_config.initial_capital,
            commission_rate=self.backtest_config.commission_rate,
            slippage_rate=self.backtest_config.slippage_rate,
            enable_funding=True,
            enable_margin_tracking=True,
            enable_liquidation=True,
            leverage=Decimal(str(self.strategy.config.target_leverage)),
        )

        # Create engine with derivatives portfolio
        engine = BacktestEngine(strategy=self.strategy, config=self.backtest_config)
        engine.portfolio = portfolio  # Override with derivatives portfolio

        # Run backtest
        backtest_result = engine.run(data=stressed_data, symbol=symbol)

        # Get derivatives stats
        deriv_stats = portfolio.get_derivatives_stats()

        # Validate against criteria
        status, failures, warnings = self._validate_result(backtest_result, deriv_stats)

        # Collect metrics
        metrics = {
            "total_return": backtest_result.metrics.total_return if backtest_result.metrics else 0,
            "max_drawdown": backtest_result.metrics.max_drawdown if backtest_result.metrics else 0,
            "sharpe_ratio": backtest_result.metrics.sharpe_ratio if backtest_result.metrics else 0,
            "liquidations": deriv_stats.get("liquidation_count", 0),
            "liquidation_warnings": deriv_stats.get("liquidation_warnings", 0),
            "funding_paid": float(deriv_stats.get("total_funding_paid", 0)),
            "max_leverage": deriv_stats.get("current_leverage", 0),
        }

        result = StressTestResult(
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            backtest_result=backtest_result,
            status=status,
            failures=failures,
            warnings=warnings,
            metrics=metrics,
        )

        self.results.append(result)

        logger.info("Stress test complete | scenario=%s | status=%s", scenario_name, status.value)

        return result

    def _validate_result(
        self, backtest_result: BacktestResult, deriv_stats: dict[str, Any]
    ) -> tuple[StressTestStatus, list[str], list[str]]:
        """
        Validate backtest result against criteria.

        Returns:
            (status, failures, warnings)
        """
        failures = []
        warnings = []

        # Check liquidations
        liquidations = deriv_stats.get("liquidation_count", 0)
        if liquidations > self.criteria.max_liquidations_allowed:
            failures.append(
                f"Liquidations: {liquidations} (max allowed: {self.criteria.max_liquidations_allowed})"
            )

        liq_warnings = deriv_stats.get("liquidation_warnings", 0)
        if self.criteria.max_liquidation_warnings and liq_warnings > self.criteria.max_liquidation_warnings:
            warnings.append(
                f"Liquidation warnings: {liq_warnings} (threshold: {self.criteria.max_liquidation_warnings})"
            )

        # Check drawdown
        if backtest_result.metrics:
            dd = backtest_result.metrics.max_drawdown
            if dd > self.criteria.max_drawdown_pct:
                failures.append(
                    f"Drawdown: {dd:.2%} (max allowed: {self.criteria.max_drawdown_pct:.2%})"
                )
            elif dd > self.criteria.max_drawdown_warning_pct:
                warnings.append(
                    f"Drawdown: {dd:.2%} (warning threshold: {self.criteria.max_drawdown_warning_pct:.2%})"
                )

            # Check Sharpe ratio
            if self.criteria.min_sharpe_ratio and backtest_result.metrics.sharpe_ratio < self.criteria.min_sharpe_ratio:
                failures.append(
                    f"Sharpe: {backtest_result.metrics.sharpe_ratio:.2f} (min required: {self.criteria.min_sharpe_ratio:.2f})"
                )

            # Check return
            if self.criteria.max_negative_return_pct:
                ret = backtest_result.metrics.total_return
                if ret < -abs(self.criteria.max_negative_return_pct):
                    failures.append(
                        f"Return: {ret:.2%} (max loss allowed: {self.criteria.max_negative_return_pct:.2%})"
                    )

        # Check leverage
        max_lev = deriv_stats.get("current_leverage", 0)
        if max_lev > self.criteria.max_leverage:
            failures.append(
                f"Leverage: {max_lev:.1f}x (max allowed: {self.criteria.max_leverage:.1f}x)"
            )

        # Determine status
        if failures:
            status = StressTestStatus.FAILED
        elif warnings:
            status = StressTestStatus.WARNING
        else:
            status = StressTestStatus.PASSED

        return status, failures, warnings

    def run_standard_suite(
        self, *, data: pd.DataFrame, symbol: str
    ) -> list[StressTestResult]:
        """
        Run standard suite of stress tests.

        Args:
            data: Historical data
            symbol: Trading symbol

        Returns:
            List of stress test results
        """
        logger.info("Running standard stress test suite | symbol=%s", symbol)

        # 1. Gap move down (5%)
        self.run_stress_test(
            scenario_name="Gap Down 5%",
            scenario_type=StressScenarioType.GAP_MOVE,
            data=data,
            symbol=symbol,
            scenario_generator=GapMoveScenario(gap_size_pct=0.05, gap_direction="down", num_gaps=3),
        )

        # 2. Gap move up (5%)
        self.run_stress_test(
            scenario_name="Gap Up 5%",
            scenario_type=StressScenarioType.GAP_MOVE,
            data=data,
            symbol=symbol,
            scenario_generator=GapMoveScenario(gap_size_pct=0.05, gap_direction="up", num_gaps=3),
        )

        # 3. High volatility (2x)
        self.run_stress_test(
            scenario_name="High Volatility 2x",
            scenario_type=StressScenarioType.HIGH_VOLATILITY,
            data=data,
            symbol=symbol,
            scenario_generator=HighVolatilityScenario(volatility_multiplier=2.0),
        )

        # 4. Flash crash
        self.run_stress_test(
            scenario_name="Flash Crash 20%",
            scenario_type=StressScenarioType.FLASH_CRASH,
            data=data,
            symbol=symbol,
            scenario_generator=FlashCrashScenario(crash_size_pct=0.20),
        )

        # 5. Funding shock
        funding_gen = FundingShockScenario(shock_rate=0.02)
        funding_schedule = funding_gen.generate_funding_schedule(data)

        # Note: For funding shock, we need special handling
        # Store schedule for use during backtest
        self._funding_schedule = funding_schedule

        self.run_stress_test(
            scenario_name="Funding Rate Shock",
            scenario_type=StressScenarioType.FUNDING_SHOCK,
            data=data,
            symbol=symbol,
            scenario_generator=funding_gen,
        )

        logger.info("Standard stress test suite complete | total_tests=%d", len(self.results))

        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive stress test report."""
        if not self.results:
            return "No stress tests run yet."

        passed = sum(1 for r in self.results if r.status == StressTestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == StressTestStatus.FAILED)
        warned = sum(1 for r in self.results if r.status == StressTestStatus.WARNING)

        report = "=" * 80 + "\n"
        report += "DERIVATIVES STRESS TEST REPORT\n"
        report += "=" * 80 + "\n\n"

        report += f"Total Tests: {len(self.results)}\n"
        report += f"✅ Passed: {passed}\n"
        report += f"❌ Failed: {failed}\n"
        report += f"⚠️  Warnings: {warned}\n\n"

        report += "=" * 80 + "\n"
        report += "INDIVIDUAL TEST RESULTS\n"
        report += "=" * 80 + "\n\n"

        for result in self.results:
            report += result.summary() + "\n"

        report += "=" * 80 + "\n"
        report += "PHASE 2 EXIT CRITERIA\n"
        report += "=" * 80 + "\n\n"

        all_passed = all(r.status == StressTestStatus.PASSED for r in self.results)

        if all_passed:
            report += "✅ All stress tests PASSED\n"
            report += "✅ System ready for derivatives deployment\n"
        else:
            report += "❌ Some stress tests FAILED\n"
            report += "❌ System NOT ready for derivatives deployment\n"
            report += "\nReview failures above and adjust strategy/risk parameters.\n"

        return report

    def passed_all(self) -> bool:
        """Check if all stress tests passed."""
        return all(r.status == StressTestStatus.PASSED for r in self.results)


__all__ = ["StressTestValidator", "StressTestCriteria", "StressTestResult", "StressTestStatus"]
