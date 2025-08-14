"""
Historical Scenario Stress Testing

Provides stress testing using historical market scenarios including:
- Predefined historical crises
- Custom scenario support
- Portfolio impact analysis
"""

import logging

import pandas as pd

from .types import ScenarioType, StressScenario, StressTestResult, StressTestType

logger = logging.getLogger(__name__)


class HistoricalStressTester:
    """Historical scenario stress testing"""

    # Predefined historical scenarios
    HISTORICAL_SCENARIOS = {
        "black_monday_1987": {
            "date": "1987-10-19",
            "market_shock": -0.22,
            "volatility_spike": 3.0,
            "description": "Black Monday 1987 crash",
        },
        "asian_crisis_1997": {
            "date": "1997-10-27",
            "market_shock": -0.07,
            "volatility_spike": 2.0,
            "description": "Asian Financial Crisis",
        },
        "ltcm_1998": {
            "date": "1998-08-31",
            "market_shock": -0.14,
            "volatility_spike": 2.5,
            "description": "LTCM collapse and Russian default",
        },
        "dot_com_2000": {
            "date": "2000-03-10",
            "market_shock": -0.09,
            "volatility_spike": 1.8,
            "description": "Dot-com bubble burst",
        },
        "september_11_2001": {
            "date": "2001-09-17",
            "market_shock": -0.07,
            "volatility_spike": 1.5,
            "description": "September 11 attacks",
        },
        "financial_crisis_2008": {
            "date": "2008-09-15",
            "market_shock": -0.15,
            "volatility_spike": 3.5,
            "description": "Lehman Brothers collapse",
        },
        "flash_crash_2010": {
            "date": "2010-05-06",
            "market_shock": -0.09,
            "volatility_spike": 2.0,
            "description": "Flash Crash",
        },
        "covid_2020": {
            "date": "2020-03-16",
            "market_shock": -0.12,
            "volatility_spike": 4.0,
            "description": "COVID-19 pandemic shock",
        },
    }

    def __init__(self):
        """Initialize historical stress tester"""
        self.scenarios = self.HISTORICAL_SCENARIOS.copy()

    def add_custom_scenario(
        self, name: str, date: str, market_shock: float, volatility_spike: float, description: str
    ) -> None:
        """Add custom historical scenario"""
        self.scenarios[name] = {
            "date": date,
            "market_shock": market_shock,
            "volatility_spike": volatility_spike,
            "description": description,
        }

    def apply_historical_scenario(
        self, portfolio_value: float, positions: pd.DataFrame, scenario_name: str
    ) -> StressTestResult:
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
            volatility_multiplier=scenario_data["volatility_spike"],
        )

        # Apply shocks to positions
        shocked_positions = positions.copy()
        shocked_positions["value"] *= 1 + scenario_data["market_shock"]

        # Calculate losses
        portfolio_loss = portfolio_value * abs(scenario_data["market_shock"])
        position_losses = dict(
            zip(
                positions["symbol"],
                positions["value"] * abs(scenario_data["market_shock"]),
                strict=False,
            )
        )
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
            worst_positions=sorted(position_losses.items(), key=lambda x: x[1], reverse=True)[:5],
            liquidation_cost=portfolio_loss * 0.02,
            days_to_liquidate=5,
        )

        return result
