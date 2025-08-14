"""
Sensitivity Analysis for Risk Factors

Provides comprehensive sensitivity analysis capabilities:
- Single factor analysis
- Greeks sensitivity (for options portfolios)
- Multi-factor sensitivity matrices
"""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """Sensitivity analysis for risk factors"""

    def __init__(self):
        """Initialize sensitivity analyzer"""
        self.sensitivity_results = {}

    def analyze_single_factor(
        self,
        portfolio_value: float,
        factor_name: str,
        factor_range: np.ndarray,
        calculation_func: Callable,
    ) -> pd.DataFrame:
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
            results.append(
                {
                    "factor": factor_name,
                    "value": factor_value,
                    "portfolio_impact": impact,
                    "percentage_impact": impact / portfolio_value,
                }
            )

        return pd.DataFrame(results)

    def analyze_greeks_sensitivity(
        self, option_positions: pd.DataFrame, spot_range: np.ndarray, vol_range: np.ndarray
    ) -> dict[str, pd.DataFrame]:
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
            total_delta = option_positions["delta"].sum() * spot / 100
            delta_sensitivity.append({"spot": spot, "delta_pnl": total_delta})
        results["delta"] = pd.DataFrame(delta_sensitivity)

        # Vega sensitivity
        vega_sensitivity = []
        for vol in vol_range:
            total_vega = option_positions["vega"].sum() * vol
            vega_sensitivity.append({"volatility": vol, "vega_pnl": total_vega})
        results["vega"] = pd.DataFrame(vega_sensitivity)

        return results

    def create_sensitivity_matrix(
        self,
        factor1_name: str,
        factor1_range: np.ndarray,
        factor2_name: str,
        factor2_range: np.ndarray,
        calculation_func: Callable,
    ) -> pd.DataFrame:
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

        return pd.DataFrame(matrix, index=factor1_range, columns=factor2_range)
