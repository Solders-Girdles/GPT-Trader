"""
Main Stress Testing Framework

Orchestrates all stress testing components and provides a unified interface for:
- Running comprehensive stress test suites
- Generating stress test reports
- Managing stress scenarios
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .historical import HistoricalStressTester
from .monte_carlo import MonteCarloEngine
from .sensitivity import SensitivityAnalyzer
from .types import ScenarioType, StressScenario, StressTestResult, StressTestType

logger = logging.getLogger(__name__)


class StressTestingFramework:
    """Main stress testing framework"""

    def __init__(self):
        """Initialize stress testing framework"""
        self.monte_carlo = MonteCarloEngine()
        self.historical_tester = HistoricalStressTester()
        self.sensitivity_analyzer = SensitivityAnalyzer()

        # Results storage
        self.test_results: list[StressTestResult] = []
        self.scenario_library: dict[str, StressScenario] = {}

        # Initialize default scenarios
        self._initialize_scenarios()

    def _initialize_scenarios(self):
        """Initialize default stress scenarios"""
        # Market crash scenario
        self.scenario_library["severe_crash"] = StressScenario(
            name="Severe Market Crash",
            scenario_type=ScenarioType.MARKET_CRASH,
            description="30% market decline over 5 days",
            market_shock=-0.30,
            volatility_multiplier=3.0,
            duration_days=5,
            shock_speed="accelerating",
        )

        # Volatility spike
        self.scenario_library["vol_spike"] = StressScenario(
            name="Volatility Spike",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            description="Volatility triples overnight",
            market_shock=-0.05,
            volatility_multiplier=3.0,
            duration_days=1,
            shock_speed="instant",
        )

        # Liquidity crisis
        self.scenario_library["liquidity_crisis"] = StressScenario(
            name="Liquidity Crisis",
            scenario_type=ScenarioType.LIQUIDITY_CRISIS,
            description="Market liquidity dries up",
            market_shock=-0.10,
            volatility_multiplier=2.0,
            liquidity_factor=0.2,
            duration_days=10,
            shock_speed="gradual",
        )

        # Correlation breakdown
        self.scenario_library["correlation_break"] = StressScenario(
            name="Correlation Breakdown",
            scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
            description="Diversification failure",
            market_shock=-0.15,
            correlation_adjustment=0.8,
            duration_days=3,
            shock_speed="instant",
        )

    def run_monte_carlo_stress(
        self,
        portfolio_value: float,
        expected_return: float,
        portfolio_volatility: float,
        stress_multiplier: float = 2.0,
        include_jumps: bool = True,
    ) -> StressTestResult:
        """
        Run Monte Carlo stress test.

        Args:
            portfolio_value: Current portfolio value
            expected_return: Expected portfolio return
            portfolio_volatility: Portfolio volatility
            stress_multiplier: Stress multiplier for volatility
            include_jumps: Whether to include jump diffusion

        Returns:
            Stress test results
        """
        # Run simulations
        if include_jumps:
            paths = self.monte_carlo.simulate_jump_diffusion(
                initial_price=portfolio_value,
                drift=expected_return,
                volatility=portfolio_volatility * stress_multiplier,
                jump_intensity=0.1,
                jump_mean=-0.05,
                jump_std=0.03,
            )
        else:
            paths = self.monte_carlo.simulate_gbm(
                initial_price=portfolio_value,
                drift=expected_return,
                volatility=portfolio_volatility * stress_multiplier,
            )

        # Calculate metrics
        final_values = paths[:, -1]
        var, cvar = self.monte_carlo.calculate_var_cvar(final_values)

        # Calculate losses
        losses = portfolio_value - final_values
        max_loss = np.max(losses)
        avg_loss = np.mean(losses[losses > 0])

        # Create scenario
        scenario = StressScenario(
            name="Monte Carlo Stress Test",
            scenario_type=ScenarioType.CUSTOM,
            description=f"MC simulation with {stress_multiplier}x volatility",
            volatility_multiplier=stress_multiplier,
        )

        # Create result
        result = StressTestResult(
            scenario=scenario,
            test_type=StressTestType.MONTE_CARLO,
            portfolio_loss=avg_loss,
            max_drawdown=max_loss / portfolio_value,
            var_impact=var * portfolio_value,
            expected_shortfall=cvar * portfolio_value,
            new_var=var,
            new_cvar=cvar,
            new_sharpe=expected_return / (portfolio_volatility * stress_multiplier),
            position_losses={},
            worst_positions=[],
            liquidation_cost=avg_loss * 0.02,
            days_to_liquidate=1,
        )

        self.test_results.append(result)
        return result

    def run_historical_stress(
        self, portfolio_value: float, positions: pd.DataFrame, scenario_name: str
    ) -> StressTestResult:
        """
        Run historical scenario stress test.

        Args:
            portfolio_value: Current portfolio value
            positions: DataFrame of positions
            scenario_name: Name of historical scenario

        Returns:
            Stress test results
        """
        result = self.historical_tester.apply_historical_scenario(
            portfolio_value, positions, scenario_name
        )

        self.test_results.append(result)
        return result

    def run_sensitivity_analysis(
        self, portfolio_value: float, factor_ranges: dict[str, np.ndarray]
    ) -> dict[str, pd.DataFrame]:
        """
        Run comprehensive sensitivity analysis.

        Args:
            portfolio_value: Current portfolio value
            factor_ranges: Dictionary of factor ranges to test

        Returns:
            Dictionary of sensitivity results
        """
        results = {}

        for factor_name, factor_range in factor_ranges.items():
            # Simple linear sensitivity for demonstration
            def calc_func(pv, factor_val):
                return pv * factor_val / 100

            results[factor_name] = self.sensitivity_analyzer.analyze_single_factor(
                portfolio_value, factor_name, factor_range, calc_func
            )

        return results

    def run_reverse_stress_test(self, portfolio_value: float, target_loss: float) -> StressScenario:
        """
        Run reverse stress test to find scenario causing target loss.

        Args:
            portfolio_value: Current portfolio value
            target_loss: Target loss amount

        Returns:
            Scenario that would cause target loss
        """
        loss_percentage = target_loss / portfolio_value

        # Find required market shock
        required_shock = -loss_percentage

        # Find implied volatility multiplier
        implied_vol = abs(required_shock) / 0.10  # Rough approximation

        scenario = StressScenario(
            name="Reverse Stress Test",
            scenario_type=ScenarioType.CUSTOM,
            description=f"Scenario causing ${target_loss:,.0f} loss",
            market_shock=required_shock,
            volatility_multiplier=max(1.0, implied_vol),
            duration_days=1,
        )

        return scenario

    def run_comprehensive_stress_suite(
        self,
        portfolio_value: float,
        positions: pd.DataFrame,
        expected_return: float = 0.08,
        portfolio_volatility: float = 0.15,
    ) -> dict[str, Any]:
        """
        Run comprehensive stress test suite.

        Args:
            portfolio_value: Current portfolio value
            positions: DataFrame of positions
            expected_return: Expected portfolio return
            portfolio_volatility: Portfolio volatility

        Returns:
            Dictionary of all stress test results
        """
        results = {
            "monte_carlo": [],
            "historical": [],
            "sensitivity": {},
            "reverse": None,
            "summary": {},
        }

        # Run Monte Carlo with different stress levels
        for stress_level in [1.5, 2.0, 3.0]:
            mc_result = self.run_monte_carlo_stress(
                portfolio_value, expected_return, portfolio_volatility, stress_level
            )
            results["monte_carlo"].append(mc_result)

        # Run historical scenarios
        for scenario in ["black_monday_1987", "financial_crisis_2008", "covid_2020"]:
            hist_result = self.run_historical_stress(portfolio_value, positions, scenario)
            results["historical"].append(hist_result)

        # Run sensitivity analysis
        factor_ranges = {
            "market_shock": np.linspace(-30, 0, 31),
            "volatility": np.linspace(10, 50, 21),
            "correlation": np.linspace(0, 100, 11),
        }
        results["sensitivity"] = self.run_sensitivity_analysis(portfolio_value, factor_ranges)

        # Run reverse stress test
        target_loss = portfolio_value * 0.25  # 25% loss target
        results["reverse"] = self.run_reverse_stress_test(portfolio_value, target_loss)

        # Calculate summary statistics
        all_losses = []
        for mc in results["monte_carlo"]:
            all_losses.append(mc.portfolio_loss)
        for hist in results["historical"]:
            all_losses.append(hist.portfolio_loss)

        results["summary"] = {
            "avg_loss": np.mean(all_losses),
            "max_loss": np.max(all_losses),
            "min_loss": np.min(all_losses),
            "loss_std": np.std(all_losses),
            "worst_scenario": (
                max(results["historical"], key=lambda x: x.portfolio_loss).scenario.name
                if results["historical"]
                else None
            ),
        }

        return results

    def generate_stress_report(self, results: dict[str, Any]) -> str:
        """
        Generate stress testing report.

        Args:
            results: Stress test results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("STRESS TESTING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Monte Carlo results
        report.append("MONTE CARLO STRESS TESTS")
        report.append("-" * 40)
        for mc_result in results.get("monte_carlo", []):
            report.append(f"  Stress Level: {mc_result.scenario.volatility_multiplier}x")
            report.append(f"    Portfolio Loss: ${mc_result.portfolio_loss:,.2f}")
            report.append(f"    Max Drawdown: {mc_result.max_drawdown:.2%}")
            report.append(f"    VaR (95%): ${mc_result.new_var:,.2f}")
            report.append(f"    CVaR (95%): ${mc_result.new_cvar:,.2f}")
            report.append("")

        # Historical scenarios
        report.append("HISTORICAL STRESS SCENARIOS")
        report.append("-" * 40)
        for hist_result in results.get("historical", []):
            report.append(f"  {hist_result.scenario.name}")
            report.append(f"    Description: {hist_result.scenario.description}")
            report.append(f"    Market Shock: {hist_result.scenario.market_shock:.2%}")
            report.append(f"    Portfolio Loss: ${hist_result.portfolio_loss:,.2f}")
            report.append(f"    Liquidation Cost: ${hist_result.liquidation_cost:,.2f}")
            report.append("")

        # Sensitivity analysis
        report.append("SENSITIVITY ANALYSIS")
        report.append("-" * 40)
        for factor, df in results.get("sensitivity", {}).items():
            if not df.empty:
                report.append(f"  {factor.upper()}")
                report.append(f"    Min Impact: {df['percentage_impact'].min():.2%}")
                report.append(f"    Max Impact: {df['percentage_impact'].max():.2%}")
                report.append(f"    Avg Impact: {df['percentage_impact'].mean():.2%}")
                report.append("")

        # Reverse stress test
        if results.get("reverse"):
            report.append("REVERSE STRESS TEST")
            report.append("-" * 40)
            reverse = results["reverse"]
            report.append(f"  Target Loss: {reverse.description}")
            report.append(f"  Required Market Shock: {reverse.market_shock:.2%}")
            report.append(f"  Implied Volatility: {reverse.volatility_multiplier}x")
            report.append("")

        # Summary
        if results.get("summary"):
            report.append("SUMMARY STATISTICS")
            report.append("-" * 40)
            summary = results["summary"]
            report.append(f"  Average Loss: ${summary['avg_loss']:,.2f}")
            report.append(f"  Maximum Loss: ${summary['max_loss']:,.2f}")
            report.append(f"  Minimum Loss: ${summary['min_loss']:,.2f}")
            report.append(f"  Loss Std Dev: ${summary['loss_std']:,.2f}")
            if summary.get("worst_scenario"):
                report.append(f"  Worst Scenario: {summary['worst_scenario']}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
