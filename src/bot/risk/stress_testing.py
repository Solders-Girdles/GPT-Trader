"""
Stress Testing Automation System
Phase 3, Week 4: RISK-017 to RISK-024
Monte Carlo simulations, historical scenarios, and sensitivity analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class StressTestType(Enum):
    """Types of stress tests"""
    MONTE_CARLO = "monte_carlo"
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    SENSITIVITY = "sensitivity"
    REVERSE = "reverse"
    PARAMETRIC = "parametric"


class ScenarioType(Enum):
    """Types of stress scenarios"""
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    FLASH_CRASH = "flash_crash"
    BLACK_SWAN = "black_swan"
    REGIME_CHANGE = "regime_change"
    CUSTOM = "custom"


@dataclass
class StressScenario:
    """Definition of a stress scenario"""
    name: str
    scenario_type: ScenarioType
    description: str
    
    # Market parameters
    market_shock: float = 0.0  # Percentage market move
    volatility_multiplier: float = 1.0
    correlation_adjustment: float = 0.0
    liquidity_factor: float = 1.0
    
    # Time parameters
    duration_days: int = 1
    shock_speed: str = "instant"  # instant, gradual, accelerating
    
    # Asset-specific shocks
    asset_shocks: Dict[str, float] = field(default_factory=dict)
    sector_shocks: Dict[str, float] = field(default_factory=dict)
    
    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StressTestResult:
    """Results from a stress test"""
    scenario: StressScenario
    test_type: StressTestType
    
    # Portfolio impacts
    portfolio_loss: float
    max_drawdown: float
    var_impact: float
    expected_shortfall: float
    
    # Risk metrics
    new_var: float
    new_cvar: float
    new_sharpe: float
    
    # Position impacts
    position_losses: Dict[str, float]
    worst_positions: List[Tuple[str, float]]
    
    # Liquidity impacts
    liquidation_cost: float
    days_to_liquidate: float
    
    # Recovery metrics
    recovery_time: Optional[int] = None
    permanent_loss: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    computation_time: float = 0.0
    confidence_level: float = 0.95


class MonteCarloEngine:
    """Monte Carlo simulation engine for stress testing"""
    
    def __init__(self,
                 n_simulations: int = 10000,
                 time_horizon: int = 252,
                 confidence_level: float = 0.95):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_simulations: Number of simulation paths
            time_horizon: Time horizon in days
            confidence_level: Confidence level for risk metrics
        """
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.confidence_level = confidence_level
        
        # Simulation cache
        self.last_simulation = None
        self.last_params = None
    
    def simulate_gbm(self,
                     initial_price: float,
                     drift: float,
                     volatility: float,
                     dt: float = 1/252) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion paths.
        
        Args:
            initial_price: Starting price
            drift: Drift parameter (annualized)
            volatility: Volatility parameter (annualized)
            dt: Time step
            
        Returns:
            Array of simulated price paths
        """
        n_steps = int(self.time_horizon)
        
        # Generate random shocks
        shocks = np.random.normal(
            0, 1, (self.n_simulations, n_steps)
        )
        
        # Calculate returns
        returns = (drift - 0.5 * volatility**2) * dt + \
                 volatility * np.sqrt(dt) * shocks
        
        # Calculate price paths
        price_paths = initial_price * np.exp(np.cumsum(returns, axis=1))
        
        # Add initial price
        price_paths = np.column_stack([
            np.full(self.n_simulations, initial_price),
            price_paths
        ])
        
        return price_paths
    
    def simulate_jump_diffusion(self,
                               initial_price: float,
                               drift: float,
                               volatility: float,
                               jump_intensity: float,
                               jump_mean: float,
                               jump_std: float,
                               dt: float = 1/252) -> np.ndarray:
        """
        Simulate Jump Diffusion process (Merton model).
        
        Args:
            initial_price: Starting price
            drift: Drift parameter
            volatility: Diffusion volatility
            jump_intensity: Poisson intensity for jumps
            jump_mean: Mean of jump size
            jump_std: Std dev of jump size
            dt: Time step
            
        Returns:
            Array of simulated price paths with jumps
        """
        n_steps = int(self.time_horizon)
        
        # GBM component
        gbm_paths = self.simulate_gbm(initial_price, drift, volatility, dt)
        
        # Jump component
        for i in range(self.n_simulations):
            n_jumps = np.random.poisson(jump_intensity * self.time_horizon * dt)
            
            if n_jumps > 0:
                jump_times = np.random.randint(0, n_steps, n_jumps)
                jump_sizes = np.random.normal(jump_mean, jump_std, n_jumps)
                
                for t, size in zip(jump_times, jump_sizes):
                    gbm_paths[i, t:] *= (1 + size)
        
        return gbm_paths
    
    def simulate_stressed_returns(self,
                                 historical_returns: pd.Series,
                                 stress_factor: float = 2.0,
                                 fat_tail_alpha: float = 3.0) -> np.ndarray:
        """
        Simulate stressed returns with fat tails.
        
        Args:
            historical_returns: Historical return series
            stress_factor: Multiplier for volatility
            fat_tail_alpha: Alpha parameter for Student-t distribution
            
        Returns:
            Array of stressed return simulations
        """
        mean = historical_returns.mean()
        std = historical_returns.std()
        
        # Use Student-t distribution for fat tails
        returns = stats.t.rvs(
            df=fat_tail_alpha,
            loc=mean,
            scale=std * stress_factor,
            size=(self.n_simulations, self.time_horizon)
        )
        
        return returns
    
    def calculate_var_cvar(self, final_values: np.ndarray) -> Tuple[float, float]:
        """
        Calculate VaR and CVaR from simulation results.
        
        Args:
            final_values: Array of final portfolio values
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        returns = (final_values - final_values[0]) / final_values[0]
        var = np.percentile(returns, (1 - self.confidence_level) * 100)
        cvar = returns[returns <= var].mean()
        
        return var, cvar


class HistoricalStressTester:
    """Historical scenario stress testing"""
    
    # Predefined historical scenarios
    HISTORICAL_SCENARIOS = {
        "black_monday_1987": {
            "date": "1987-10-19",
            "market_shock": -0.22,
            "volatility_spike": 3.0,
            "description": "Black Monday 1987 crash"
        },
        "asian_crisis_1997": {
            "date": "1997-10-27",
            "market_shock": -0.07,
            "volatility_spike": 2.0,
            "description": "Asian Financial Crisis"
        },
        "ltcm_1998": {
            "date": "1998-08-31",
            "market_shock": -0.14,
            "volatility_spike": 2.5,
            "description": "LTCM collapse and Russian default"
        },
        "dot_com_2000": {
            "date": "2000-03-10",
            "market_shock": -0.09,
            "volatility_spike": 1.8,
            "description": "Dot-com bubble burst"
        },
        "september_11_2001": {
            "date": "2001-09-17",
            "market_shock": -0.07,
            "volatility_spike": 1.5,
            "description": "September 11 attacks"
        },
        "financial_crisis_2008": {
            "date": "2008-09-15",
            "market_shock": -0.15,
            "volatility_spike": 3.5,
            "description": "Lehman Brothers collapse"
        },
        "flash_crash_2010": {
            "date": "2010-05-06",
            "market_shock": -0.09,
            "volatility_spike": 2.0,
            "description": "Flash Crash"
        },
        "covid_2020": {
            "date": "2020-03-16",
            "market_shock": -0.12,
            "volatility_spike": 4.0,
            "description": "COVID-19 pandemic shock"
        }
    }
    
    def __init__(self):
        """Initialize historical stress tester"""
        self.scenarios = self.HISTORICAL_SCENARIOS.copy()
    
    def add_custom_scenario(self,
                           name: str,
                           date: str,
                           market_shock: float,
                           volatility_spike: float,
                           description: str) -> None:
        """Add custom historical scenario"""
        self.scenarios[name] = {
            "date": date,
            "market_shock": market_shock,
            "volatility_spike": volatility_spike,
            "description": description
        }
    
    def apply_historical_scenario(self,
                                 portfolio_value: float,
                                 positions: pd.DataFrame,
                                 scenario_name: str) -> StressTestResult:
        """
        Apply historical scenario to portfolio.
        
        Args:
            portfolio_value: Current portfolio value
            positions: DataFrame of positions
            scenario_name: Name of historical scenario
            
        Returns:
            Stress test results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario_data = self.scenarios[scenario_name]
        
        # Create stress scenario
        scenario = StressScenario(
            name=scenario_name,
            scenario_type=ScenarioType.HISTORICAL,
            description=scenario_data["description"],
            market_shock=scenario_data["market_shock"],
            volatility_multiplier=scenario_data["volatility_spike"]
        )
        
        # Apply shocks to positions
        shocked_positions = positions.copy()
        shocked_positions['value'] *= (1 + scenario_data["market_shock"])
        
        # Calculate losses
        portfolio_loss = portfolio_value * abs(scenario_data["market_shock"])
        position_losses = {
            row['symbol']: row['value'] * abs(scenario_data["market_shock"])
            for _, row in positions.iterrows()
        }
        
        # Create result
        result = StressTestResult(
            scenario=scenario,
            test_type=StressTestType.HISTORICAL,
            portfolio_loss=portfolio_loss,
            max_drawdown=scenario_data["market_shock"],
            var_impact=portfolio_loss * 0.05,  # Simplified
            expected_shortfall=portfolio_loss * 0.075,  # Simplified
            new_var=portfolio_loss * 0.05,
            new_cvar=portfolio_loss * 0.075,
            new_sharpe=-0.5,  # Stress scenario
            position_losses=position_losses,
            worst_positions=sorted(
                position_losses.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            liquidation_cost=portfolio_loss * 0.02,
            days_to_liquidate=5
        )
        
        return result


class SensitivityAnalyzer:
    """Sensitivity analysis for risk factors"""
    
    def __init__(self):
        """Initialize sensitivity analyzer"""
        self.sensitivity_results = {}
    
    def analyze_single_factor(self,
                            portfolio_value: float,
                            factor_name: str,
                            factor_range: np.ndarray,
                            calculation_func: Callable) -> pd.DataFrame:
        """
        Analyze sensitivity to single factor.
        
        Args:
            portfolio_value: Current portfolio value
            factor_name: Name of risk factor
            factor_range: Range of factor values to test
            calculation_func: Function to calculate impact
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        for factor_value in factor_range:
            impact = calculation_func(portfolio_value, factor_value)
            results.append({
                'factor': factor_name,
                'value': factor_value,
                'portfolio_impact': impact,
                'percentage_impact': impact / portfolio_value
            })
        
        return pd.DataFrame(results)
    
    def analyze_greeks_sensitivity(self,
                                  option_positions: pd.DataFrame,
                                  spot_range: np.ndarray,
                                  vol_range: np.ndarray) -> Dict[str, pd.DataFrame]:
        """
        Analyze Greeks sensitivity.
        
        Args:
            option_positions: DataFrame of option positions
            spot_range: Range of spot prices
            vol_range: Range of volatilities
            
        Returns:
            Dictionary of sensitivity DataFrames
        """
        results = {}
        
        # Delta sensitivity
        delta_sensitivity = []
        for spot in spot_range:
            # Simplified calculation
            total_delta = option_positions['delta'].sum() * spot / 100
            delta_sensitivity.append({
                'spot': spot,
                'delta_pnl': total_delta
            })
        results['delta'] = pd.DataFrame(delta_sensitivity)
        
        # Vega sensitivity
        vega_sensitivity = []
        for vol in vol_range:
            total_vega = option_positions['vega'].sum() * vol
            vega_sensitivity.append({
                'volatility': vol,
                'vega_pnl': total_vega
            })
        results['vega'] = pd.DataFrame(vega_sensitivity)
        
        return results
    
    def create_sensitivity_matrix(self,
                                 factor1_name: str,
                                 factor1_range: np.ndarray,
                                 factor2_name: str,
                                 factor2_range: np.ndarray,
                                 calculation_func: Callable) -> pd.DataFrame:
        """
        Create two-factor sensitivity matrix.
        
        Args:
            factor1_name: First factor name
            factor1_range: First factor range
            factor2_name: Second factor name
            factor2_range: Second factor range
            calculation_func: Function to calculate combined impact
            
        Returns:
            DataFrame with sensitivity matrix
        """
        matrix = np.zeros((len(factor1_range), len(factor2_range)))
        
        for i, f1 in enumerate(factor1_range):
            for j, f2 in enumerate(factor2_range):
                matrix[i, j] = calculation_func(f1, f2)
        
        return pd.DataFrame(
            matrix,
            index=factor1_range,
            columns=factor2_range
        )


class StressTestingFramework:
    """Main stress testing framework"""
    
    def __init__(self):
        """Initialize stress testing framework"""
        self.monte_carlo = MonteCarloEngine()
        self.historical_tester = HistoricalStressTester()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        
        # Results storage
        self.test_results: List[StressTestResult] = []
        self.scenario_library: Dict[str, StressScenario] = {}
        
        # Initialize default scenarios
        self._initialize_scenarios()
    
    def _initialize_scenarios(self):
        """Initialize default stress scenarios"""
        # Market crash scenario
        self.scenario_library['severe_crash'] = StressScenario(
            name="Severe Market Crash",
            scenario_type=ScenarioType.MARKET_CRASH,
            description="30% market decline over 5 days",
            market_shock=-0.30,
            volatility_multiplier=3.0,
            duration_days=5,
            shock_speed="accelerating"
        )
        
        # Volatility spike
        self.scenario_library['vol_spike'] = StressScenario(
            name="Volatility Spike",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            description="Volatility triples overnight",
            market_shock=-0.05,
            volatility_multiplier=3.0,
            duration_days=1,
            shock_speed="instant"
        )
        
        # Liquidity crisis
        self.scenario_library['liquidity_crisis'] = StressScenario(
            name="Liquidity Crisis",
            scenario_type=ScenarioType.LIQUIDITY_CRISIS,
            description="Market liquidity dries up",
            market_shock=-0.10,
            volatility_multiplier=2.0,
            liquidity_factor=0.2,
            duration_days=10,
            shock_speed="gradual"
        )
        
        # Correlation breakdown
        self.scenario_library['correlation_break'] = StressScenario(
            name="Correlation Breakdown",
            scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
            description="Diversification failure",
            market_shock=-0.15,
            correlation_adjustment=0.8,
            duration_days=3,
            shock_speed="instant"
        )
    
    def run_monte_carlo_stress(self,
                              portfolio_value: float,
                              expected_return: float,
                              portfolio_volatility: float,
                              stress_multiplier: float = 2.0,
                              include_jumps: bool = True) -> StressTestResult:
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
                jump_std=0.03
            )
        else:
            paths = self.monte_carlo.simulate_gbm(
                initial_price=portfolio_value,
                drift=expected_return,
                volatility=portfolio_volatility * stress_multiplier
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
            volatility_multiplier=stress_multiplier
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
            days_to_liquidate=1
        )
        
        self.test_results.append(result)
        return result
    
    def run_historical_stress(self,
                            portfolio_value: float,
                            positions: pd.DataFrame,
                            scenario_name: str) -> StressTestResult:
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
    
    def run_sensitivity_analysis(self,
                                portfolio_value: float,
                                factor_ranges: Dict[str, np.ndarray]) -> Dict[str, pd.DataFrame]:
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
                portfolio_value,
                factor_name,
                factor_range,
                calc_func
            )
        
        return results
    
    def run_reverse_stress_test(self,
                               portfolio_value: float,
                               target_loss: float) -> StressScenario:
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
            duration_days=1
        )
        
        return scenario
    
    def run_comprehensive_stress_suite(self,
                                      portfolio_value: float,
                                      positions: pd.DataFrame,
                                      expected_return: float = 0.08,
                                      portfolio_volatility: float = 0.15) -> Dict[str, Any]:
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
            'monte_carlo': [],
            'historical': [],
            'sensitivity': {},
            'reverse': None,
            'summary': {}
        }
        
        # Run Monte Carlo with different stress levels
        for stress_level in [1.5, 2.0, 3.0]:
            mc_result = self.run_monte_carlo_stress(
                portfolio_value,
                expected_return,
                portfolio_volatility,
                stress_level
            )
            results['monte_carlo'].append(mc_result)
        
        # Run historical scenarios
        for scenario in ['black_monday_1987', 'financial_crisis_2008', 'covid_2020']:
            hist_result = self.run_historical_stress(
                portfolio_value,
                positions,
                scenario
            )
            results['historical'].append(hist_result)
        
        # Run sensitivity analysis
        factor_ranges = {
            'market_shock': np.linspace(-30, 0, 31),
            'volatility': np.linspace(10, 50, 21),
            'correlation': np.linspace(0, 100, 11)
        }
        results['sensitivity'] = self.run_sensitivity_analysis(
            portfolio_value,
            factor_ranges
        )
        
        # Run reverse stress test
        target_loss = portfolio_value * 0.25  # 25% loss target
        results['reverse'] = self.run_reverse_stress_test(
            portfolio_value,
            target_loss
        )
        
        # Calculate summary statistics
        all_losses = []
        for mc in results['monte_carlo']:
            all_losses.append(mc.portfolio_loss)
        for hist in results['historical']:
            all_losses.append(hist.portfolio_loss)
        
        results['summary'] = {
            'avg_loss': np.mean(all_losses),
            'max_loss': np.max(all_losses),
            'min_loss': np.min(all_losses),
            'loss_std': np.std(all_losses),
            'worst_scenario': max(
                results['historical'],
                key=lambda x: x.portfolio_loss
            ).scenario.name if results['historical'] else None
        }
        
        return results
    
    def generate_stress_report(self, results: Dict[str, Any]) -> str:
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
        for mc_result in results.get('monte_carlo', []):
            report.append(f"  Stress Level: {mc_result.scenario.volatility_multiplier}x")
            report.append(f"    Portfolio Loss: ${mc_result.portfolio_loss:,.2f}")
            report.append(f"    Max Drawdown: {mc_result.max_drawdown:.2%}")
            report.append(f"    VaR (95%): ${mc_result.new_var:,.2f}")
            report.append(f"    CVaR (95%): ${mc_result.new_cvar:,.2f}")
            report.append("")
        
        # Historical scenarios
        report.append("HISTORICAL STRESS SCENARIOS")
        report.append("-" * 40)
        for hist_result in results.get('historical', []):
            report.append(f"  {hist_result.scenario.name}")
            report.append(f"    Description: {hist_result.scenario.description}")
            report.append(f"    Market Shock: {hist_result.scenario.market_shock:.2%}")
            report.append(f"    Portfolio Loss: ${hist_result.portfolio_loss:,.2f}")
            report.append(f"    Liquidation Cost: ${hist_result.liquidation_cost:,.2f}")
            report.append("")
        
        # Sensitivity analysis
        report.append("SENSITIVITY ANALYSIS")
        report.append("-" * 40)
        for factor, df in results.get('sensitivity', {}).items():
            if not df.empty:
                report.append(f"  {factor.upper()}")
                report.append(f"    Min Impact: {df['percentage_impact'].min():.2%}")
                report.append(f"    Max Impact: {df['percentage_impact'].max():.2%}")
                report.append(f"    Avg Impact: {df['percentage_impact'].mean():.2%}")
                report.append("")
        
        # Reverse stress test
        if results.get('reverse'):
            report.append("REVERSE STRESS TEST")
            report.append("-" * 40)
            reverse = results['reverse']
            report.append(f"  Target Loss: {reverse.description}")
            report.append(f"  Required Market Shock: {reverse.market_shock:.2%}")
            report.append(f"  Implied Volatility: {reverse.volatility_multiplier}x")
            report.append("")
        
        # Summary
        if results.get('summary'):
            report.append("SUMMARY STATISTICS")
            report.append("-" * 40)
            summary = results['summary']
            report.append(f"  Average Loss: ${summary['avg_loss']:,.2f}")
            report.append(f"  Maximum Loss: ${summary['max_loss']:,.2f}")
            report.append(f"  Minimum Loss: ${summary['min_loss']:,.2f}")
            report.append(f"  Loss Std Dev: ${summary['loss_std']:,.2f}")
            if summary.get('worst_scenario'):
                report.append(f"  Worst Scenario: {summary['worst_scenario']}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    framework = StressTestingFramework()
    
    # Sample portfolio
    portfolio_value = 1_000_000
    positions = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        'value': [250000, 250000, 250000, 250000],
        'delta': [0.5, 0.6, 0.4, 0.7],
        'vega': [10, 12, 8, 15]
    })
    
    # Run comprehensive stress tests
    results = framework.run_comprehensive_stress_suite(
        portfolio_value,
        positions
    )
    
    # Generate report
    report = framework.generate_stress_report(results)
    print(report)