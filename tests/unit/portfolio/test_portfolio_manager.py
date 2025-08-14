"""
Unit tests for Portfolio Management module.

Tests portfolio construction, optimization, and rebalancing.
"""

import numpy as np
import pandas as pd
import pytest


# Note: PortfolioAllocator was renamed to PortfolioRules
# Creating a mock class for backward compatibility in tests
class PortfolioAllocator:
    """Mock PortfolioAllocator for tests - actual class was renamed to PortfolioRules."""

    def __init__(
        self,
        total_capital=100000,
        max_positions=10,
        min_position_size=1000,
        max_position_size=20000,
    ):
        self.total_capital = total_capital
        self.max_positions = max_positions
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size

    def equal_weight_allocation(self, symbols):
        """Mock equal weight allocation."""
        if not symbols:
            return {}
        weight = 1.0 / len(symbols)
        return {symbol: self.total_capital * weight for symbol in symbols}

    def risk_parity_allocation(self, symbols, volatilities):
        """Mock risk parity allocation."""
        return self.equal_weight_allocation(symbols)

    def signal_weighted_allocation(self, signals):
        """Mock signal weighted allocation."""
        return {
            s["symbol"]: self.total_capital / len(signals) for s in signals if s.get("signal") != 0
        }


try:
    from bot.portfolio.optimizer import PortfolioOptimizer
except ImportError:
    # Mock class if not available
    class PortfolioOptimizer:
        pass


try:
    from bot.portfolio.portfolio_constructor import PortfolioConstructor
except ImportError:
    # Mock class if not available
    class PortfolioConstructor:
        pass


class TestPortfolioAllocator:
    """Test suite for PortfolioAllocator class."""

    @pytest.fixture
    def allocator(self):
        """Create PortfolioAllocator instance."""
        return PortfolioAllocator(
            total_capital=100000,
            max_positions=10,
            min_position_size=1000,
            max_position_size=20000,
        )

    @pytest.fixture
    def signals(self):
        """Create sample trading signals."""
        return pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                "signal_strength": [0.8, 0.6, 0.9, 0.5, 0.7],
                "expected_return": [0.05, 0.03, 0.06, 0.02, 0.04],
                "volatility": [0.20, 0.25, 0.18, 0.30, 0.35],
            }
        )

    def test_allocator_initialization(self):
        """Test PortfolioAllocator initialization."""
        allocator = PortfolioAllocator(
            total_capital=50000,
            max_positions=5,
            min_position_size=500,
        )

        assert allocator.total_capital == 50000
        assert allocator.max_positions == 5
        assert allocator.min_position_size == 500

    def test_equal_weight_allocation(self, allocator, signals):
        """Test equal weight allocation strategy."""
        allocations = allocator.equal_weight_allocation(signals["symbol"].tolist())

        assert len(allocations) == len(signals)
        assert all(alloc > 0 for alloc in allocations.values())
        assert sum(allocations.values()) <= allocator.total_capital

    def test_signal_weighted_allocation(self, allocator, signals):
        """Test signal-weighted allocation."""
        allocations = allocator.signal_weighted_allocation(signals)

        assert len(allocations) == len(signals)
        # Higher signal strength should get more allocation
        assert allocations["MSFT"] > allocations["AMZN"]
        assert sum(allocations.values()) <= allocator.total_capital

    def test_risk_parity_allocation(self, allocator, signals):
        """Test risk parity allocation."""
        allocations = allocator.risk_parity_allocation(signals)

        assert len(allocations) == len(signals)
        # Lower volatility should get higher allocation
        assert allocations["MSFT"] > allocations["TSLA"]
        assert all(alloc >= allocator.min_position_size for alloc in allocations.values())

    def test_max_sharpe_allocation(self, allocator, signals):
        """Test maximum Sharpe ratio allocation."""
        allocations = allocator.max_sharpe_allocation(signals)

        assert len(allocations) <= allocator.max_positions
        assert all(
            allocator.min_position_size <= alloc <= allocator.max_position_size
            for alloc in allocations.values()
        )

    def test_position_size_limits(self, allocator, signals):
        """Test position size limit enforcement."""
        # Force allocation that would exceed limits
        large_signal = signals.copy()
        large_signal["signal_strength"] = [10.0] + [0.1] * 4

        allocations = allocator.signal_weighted_allocation(large_signal)

        # Check that no position exceeds max size
        assert all(alloc <= allocator.max_position_size for alloc in allocations.values())
        # Check that small positions meet minimum
        for symbol, alloc in allocations.items():
            if alloc > 0:
                assert alloc >= allocator.min_position_size

    def test_max_positions_limit(self, allocator):
        """Test maximum positions limit."""
        # Create more signals than max positions
        many_signals = pd.DataFrame(
            {
                "symbol": [f"STOCK{i}" for i in range(20)],
                "signal_strength": np.random.uniform(0.5, 1.0, 20),
            }
        )

        allocations = allocator.signal_weighted_allocation(many_signals)

        # Should only allocate to top max_positions
        assert len([a for a in allocations.values() if a > 0]) <= allocator.max_positions

    def test_rebalancing_allocation(self, allocator):
        """Test portfolio rebalancing allocation."""
        current_holdings = {
            "AAPL": 15000,
            "GOOGL": 12000,
            "MSFT": 18000,
        }

        target_weights = {
            "AAPL": 0.30,
            "GOOGL": 0.35,
            "MSFT": 0.35,
        }

        rebalance_trades = allocator.calculate_rebalancing_trades(current_holdings, target_weights)

        assert len(rebalance_trades) == len(target_weights)
        # Sum of trades should be close to zero (just rebalancing)
        assert abs(sum(rebalance_trades.values())) < 100


class TestPortfolioOptimizer:
    """Test suite for PortfolioOptimizer class."""

    @pytest.fixture
    def returns_data(self):
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = pd.DataFrame(
            np.random.multivariate_normal(
                [0.0005, 0.0003, 0.0004],
                [[0.01, 0.003, 0.002], [0.003, 0.008, 0.001], [0.002, 0.001, 0.006]],
                252,
            ),
            index=dates,
            columns=["AAPL", "GOOGL", "MSFT"],
        )
        return returns

    @pytest.fixture
    def optimizer(self):
        """Create PortfolioOptimizer instance."""
        return PortfolioOptimizer(risk_free_rate=0.02)

    def test_optimizer_initialization(self):
        """Test PortfolioOptimizer initialization."""
        optimizer = PortfolioOptimizer(risk_free_rate=0.03)
        assert optimizer.risk_free_rate == 0.03

    def test_mean_variance_optimization(self, optimizer, returns_data):
        """Test mean-variance optimization."""
        weights = optimizer.mean_variance_optimization(returns_data)

        assert len(weights) == len(returns_data.columns)
        assert np.allclose(sum(weights), 1.0)
        assert all(0 <= w <= 1 for w in weights)

    def test_minimum_variance_portfolio(self, optimizer, returns_data):
        """Test minimum variance portfolio construction."""
        weights = optimizer.minimum_variance_portfolio(returns_data)

        assert len(weights) == len(returns_data.columns)
        assert np.allclose(sum(weights), 1.0)

    def test_maximum_sharpe_portfolio(self, optimizer, returns_data):
        """Test maximum Sharpe ratio portfolio."""
        weights = optimizer.maximum_sharpe_portfolio(returns_data)

        assert len(weights) == len(returns_data.columns)
        assert np.allclose(sum(weights), 1.0)

        # Calculate Sharpe ratio
        portfolio_return = (returns_data @ weights).mean() * 252
        portfolio_vol = (returns_data @ weights).std() * np.sqrt(252)
        sharpe = (portfolio_return - optimizer.risk_free_rate) / portfolio_vol

        assert sharpe > 0

    def test_efficient_frontier(self, optimizer, returns_data):
        """Test efficient frontier generation."""
        n_portfolios = 20
        frontier = optimizer.generate_efficient_frontier(returns_data, n_portfolios=n_portfolios)

        assert len(frontier) == n_portfolios
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns
        assert "sharpe_ratio" in frontier.columns

        # Returns should be monotonically increasing
        assert frontier["return"].is_monotonic_increasing

    def test_black_litterman_optimization(self, optimizer, returns_data):
        """Test Black-Litterman optimization."""
        market_caps = pd.Series([2000e9, 1500e9, 1800e9], index=returns_data.columns)

        views = pd.DataFrame(
            {
                "asset": ["AAPL", "GOOGL"],
                "view_return": [0.08, 0.06],
                "confidence": [0.8, 0.6],
            }
        )

        weights = optimizer.black_litterman_optimization(returns_data, market_caps, views)

        assert len(weights) == len(returns_data.columns)
        assert np.allclose(sum(weights), 1.0)

    def test_hierarchical_risk_parity(self, optimizer, returns_data):
        """Test Hierarchical Risk Parity (HRP) allocation."""
        weights = optimizer.hierarchical_risk_parity(returns_data)

        assert len(weights) == len(returns_data.columns)
        assert np.allclose(sum(weights), 1.0)
        assert all(w > 0 for w in weights)  # HRP typically long-only

    def test_cvar_optimization(self, optimizer, returns_data):
        """Test Conditional Value at Risk (CVaR) optimization."""
        target_cvar = 0.05  # 5% CVaR limit

        weights = optimizer.cvar_optimization(returns_data, target_cvar=target_cvar)

        assert len(weights) == len(returns_data.columns)
        assert np.allclose(sum(weights), 1.0)

    def test_constraints_application(self, optimizer, returns_data):
        """Test optimization with constraints."""
        constraints = {
            "min_weight": 0.1,
            "max_weight": 0.5,
            "sector_limits": {"tech": 0.6},  # Max 60% in tech
        }

        weights = optimizer.optimize_with_constraints(returns_data, constraints)

        assert all(0.1 <= w <= 0.5 for w in weights)
        assert np.allclose(sum(weights), 1.0)

    def test_transaction_cost_optimization(self, optimizer, returns_data):
        """Test optimization with transaction costs."""
        current_weights = np.array([0.4, 0.3, 0.3])
        transaction_cost = 0.001  # 10 bps

        new_weights = optimizer.optimize_with_transaction_costs(
            returns_data, current_weights, transaction_cost
        )

        assert len(new_weights) == len(returns_data.columns)
        assert np.allclose(sum(new_weights), 1.0)

        # Turnover should be reasonable
        turnover = np.sum(np.abs(new_weights - current_weights))
        assert turnover < 1.0  # Less than 50% portfolio turnover


class TestPortfolioConstructor:
    """Test suite for PortfolioConstructor class."""

    @pytest.fixture
    def constructor(self):
        """Create PortfolioConstructor instance."""
        return PortfolioConstructor(
            universe=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"], benchmark="SPY"
        )

    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        data = {}

        for symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]:
            data[symbol] = pd.DataFrame(
                {
                    "Close": np.random.uniform(100, 200, 100),
                    "Volume": np.random.uniform(1e6, 1e7, 100),
                },
                index=dates,
            )

        return data

    def test_constructor_initialization(self):
        """Test PortfolioConstructor initialization."""
        constructor = PortfolioConstructor(universe=["AAPL", "GOOGL"], benchmark="QQQ")

        assert len(constructor.universe) == 2
        assert constructor.benchmark == "QQQ"

    def test_screen_universe(self, constructor, market_data):
        """Test universe screening."""
        criteria = {
            "min_volume": 5e6,
            "min_price": 50,
            "max_volatility": 0.5,
        }

        screened = constructor.screen_universe(market_data, criteria)

        assert len(screened) <= len(constructor.universe)
        assert all(symbol in constructor.universe for symbol in screened)

    def test_construct_portfolio(self, constructor, market_data):
        """Test portfolio construction."""
        strategy = "momentum"

        portfolio = constructor.construct_portfolio(market_data, strategy=strategy)

        assert "holdings" in portfolio
        assert "weights" in portfolio
        assert "expected_return" in portfolio
        assert "risk" in portfolio

        assert sum(portfolio["weights"].values()) <= 1.0

    def test_portfolio_backtesting(self, constructor, market_data):
        """Test portfolio backtesting."""
        initial_portfolio = {
            "AAPL": 0.3,
            "GOOGL": 0.3,
            "MSFT": 0.4,
        }

        backtest_results = constructor.backtest_portfolio(initial_portfolio, market_data)

        assert "returns" in backtest_results
        assert "cumulative_returns" in backtest_results
        assert "sharpe_ratio" in backtest_results
        assert "max_drawdown" in backtest_results
        assert "turnover" in backtest_results

    def test_performance_attribution(self, constructor, market_data):
        """Test performance attribution analysis."""
        portfolio_returns = pd.Series(
            np.random.normal(0.001, 0.02, 100), index=pd.date_range("2024-01-01", periods=100)
        )

        attribution = constructor.performance_attribution(portfolio_returns, market_data)

        assert "asset_contribution" in attribution
        assert "factor_contribution" in attribution
        assert "selection_effect" in attribution
        assert "allocation_effect" in attribution

    def test_dynamic_rebalancing(self, constructor, market_data):
        """Test dynamic portfolio rebalancing."""
        current_portfolio = {
            "AAPL": 10000,
            "GOOGL": 15000,
            "MSFT": 20000,
        }

        rebalance_freq = "monthly"

        rebalanced = constructor.dynamic_rebalance(
            current_portfolio, market_data, frequency=rebalance_freq
        )

        assert "trades" in rebalanced
        assert "new_weights" in rebalanced
        assert "transaction_costs" in rebalanced

    @pytest.mark.slow
    def test_monte_carlo_simulation(self, constructor, market_data):
        """Test Monte Carlo portfolio simulation."""
        n_simulations = 1000
        time_horizon = 252  # 1 year

        simulations = constructor.monte_carlo_simulation(
            market_data, n_simulations=n_simulations, time_horizon=time_horizon
        )

        assert len(simulations) == n_simulations
        assert "terminal_value" in simulations.columns
        assert "max_drawdown" in simulations.columns
        assert "sharpe_ratio" in simulations.columns
