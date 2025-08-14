"""
Monte Carlo Engine for Stress Testing

Provides Monte Carlo simulation capabilities including:
- Geometric Brownian Motion (GBM)
- Jump Diffusion (Merton model)
- Stressed returns with fat tails
- VaR/CVaR calculations
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """Monte Carlo simulation engine for stress testing"""

    def __init__(
        self, n_simulations: int = 10000, time_horizon: int = 252, confidence_level: float = 0.95
    ):
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

    def simulate_gbm(
        self, initial_price: float, drift: float, volatility: float, dt: float = 1 / 252
    ) -> np.ndarray:
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
        shocks = np.random.normal(0, 1, (self.n_simulations, n_steps))

        # Calculate returns
        returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks

        # Calculate price paths
        price_paths = initial_price * np.exp(np.cumsum(returns, axis=1))

        # Add initial price
        price_paths = np.column_stack([np.full(self.n_simulations, initial_price), price_paths])

        return price_paths

    def simulate_jump_diffusion(
        self,
        initial_price: float,
        drift: float,
        volatility: float,
        jump_intensity: float,
        jump_mean: float,
        jump_std: float,
        dt: float = 1 / 252,
    ) -> np.ndarray:
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

                for t, size in zip(jump_times, jump_sizes, strict=False):
                    gbm_paths[i, t:] *= 1 + size

        return gbm_paths

    def simulate_stressed_returns(
        self, historical_returns: pd.Series, stress_factor: float = 2.0, fat_tail_alpha: float = 3.0
    ) -> np.ndarray:
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
            size=(self.n_simulations, self.time_horizon),
        )

        return returns

    def calculate_var_cvar(self, final_values: np.ndarray) -> tuple[float, float]:
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
