"""
Integration tests for backtesting engine.

Tests the complete backtesting workflow including data loading,
strategy execution, and performance calculation.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from bot.backtest.engine_portfolio import BacktestEngine as PortfolioBacktestEngine
from bot.risk.manager import RiskManager
from bot.strategy.demo_ma import DemoMAStrategy


class TestBacktestIntegration:
    """Integration tests for the backtesting system."""

    @pytest.fixture
    def sample_universe(self):
        """Create sample universe of stocks."""
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    @pytest.fixture
    def backtest_config(self):
        """Create backtesting configuration."""
        return {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.001,
            "data_frequency": "1d",
            "rebalance_frequency": "monthly",
        }

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data = {}

        for symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]:
            np.random.seed(hash(symbol) % 1000)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))

            data[symbol] = pd.DataFrame(
                {
                    "Open": prices * np.random.uniform(0.98, 1.02, len(dates)),
                    "High": prices * np.random.uniform(1.01, 1.05, len(dates)),
                    "Low": prices * np.random.uniform(0.95, 0.99, len(dates)),
                    "Close": prices,
                    "Volume": np.random.uniform(1e6, 1e8, len(dates)),
                },
                index=dates,
            )

        return data

    @pytest.fixture
    def backtest_engine(self, backtest_config, sample_universe):
        """Create backtest engine instance."""
        return PortfolioBacktestEngine(universe=sample_universe, config=backtest_config)

    def test_end_to_end_backtest(self, backtest_engine, mock_market_data):
        """Test complete backtest workflow."""
        # Initialize strategy
        strategy = DemoMAStrategy(short_window=20, long_window=50, confidence_threshold=0.6)

        # Run backtest
        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            results = backtest_engine.run(strategy)

        # Verify results structure
        assert "returns" in results
        assert "positions" in results
        assert "metrics" in results
        assert "trades" in results

        # Verify metrics
        metrics = results["metrics"]
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "number_of_trades" in metrics

    def test_data_pipeline_integration(self, backtest_engine):
        """Test data loading and preprocessing pipeline."""
        with patch(
            "src.bot.dataflow.sources.yfinance_source.YFinanceSource.fetch_data"
        ) as mock_fetch:
            # Setup mock data
            mock_data = pd.DataFrame(
                {
                    "Open": [100, 101, 102],
                    "High": [105, 106, 107],
                    "Low": [99, 100, 101],
                    "Close": [103, 104, 105],
                    "Volume": [1e6, 1.1e6, 1.2e6],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
            mock_fetch.return_value = mock_data

            # Load data through pipeline
            data = backtest_engine.load_data()

            # Verify data structure
            assert isinstance(data, dict)
            for symbol in backtest_engine.universe:
                assert symbol in data
                assert isinstance(data[symbol], pd.DataFrame)
                assert all(
                    col in data[symbol].columns
                    for col in ["Open", "High", "Low", "Close", "Volume"]
                )

    def test_strategy_signal_generation(self, mock_market_data):
        """Test strategy signal generation with real data flow."""
        strategy = DemoMAStrategy(short_window=10, long_window=20)

        for symbol, data in mock_market_data.items():
            # Calculate indicators
            data_with_indicators = strategy.calculate_indicators(data)

            # Generate signals
            signals = strategy.generate_signals(data_with_indicators)

            # Verify signals
            assert len(signals) == len(data)
            assert all(signal in [-1, 0, 1] for signal in signals.unique())
            assert not signals.isna().any()

    def test_portfolio_allocation_integration(self, mock_market_data):
        """Test portfolio allocation with signals."""
        # Create allocator
        allocator = PortfolioAllocator(
            total_capital=100000, max_positions=5, min_position_size=5000
        )

        # Generate mock signals
        signals = pd.DataFrame(
            {
                "symbol": list(mock_market_data.keys()),
                "signal": [1, 1, -1, 0, 1],
                "signal_strength": [0.8, 0.6, 0.7, 0.0, 0.9],
            }
        )

        # Allocate capital
        allocations = allocator.signal_weighted_allocation(signals[signals["signal"] != 0])

        # Verify allocations
        assert sum(allocations.values()) <= allocator.total_capital
        assert all(
            alloc >= allocator.min_position_size or alloc == 0 for alloc in allocations.values()
        )

    def test_risk_management_integration(self, backtest_engine, mock_market_data):
        """Test risk management integration in backtesting."""
        # Create risk manager
        risk_manager = RiskManager(
            {
                "max_position_size": 20000,
                "max_portfolio_risk": 0.02,
                "stop_loss_pct": 0.05,
            }
        )

        # Attach to backtest engine
        backtest_engine.risk_manager = risk_manager

        # Create strategy
        strategy = DemoMAStrategy()

        # Run backtest with risk management
        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            results = backtest_engine.run(strategy)

        # Verify risk limits were applied
        positions = results["positions"]
        for position in positions.values():
            if position > 0:
                assert position <= risk_manager.max_position_size

    def test_performance_calculation(self, backtest_engine):
        """Test performance metrics calculation."""
        # Create mock portfolio returns
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        # Calculate metrics
        metrics = backtest_engine.calculate_performance_metrics(returns)

        # Verify all metrics present
        required_metrics = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "avg_win",
            "avg_loss",
        ]

        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.number))

    def test_transaction_cost_modeling(self, backtest_engine, mock_market_data):
        """Test transaction cost impact on results."""
        strategy = DemoMAStrategy()

        # Run without transaction costs
        backtest_engine.config["commission"] = 0
        backtest_engine.config["slippage"] = 0

        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            results_no_costs = backtest_engine.run(strategy)

        # Run with transaction costs
        backtest_engine.config["commission"] = 0.001
        backtest_engine.config["slippage"] = 0.001

        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            results_with_costs = backtest_engine.run(strategy)

        # Returns should be lower with costs
        assert (
            results_with_costs["metrics"]["total_return"]
            < results_no_costs["metrics"]["total_return"]
        )

    def test_multi_strategy_backtest(self, backtest_engine, mock_market_data):
        """Test running multiple strategies in parallel."""
        strategies = [
            DemoMAStrategy(short_window=10, long_window=30),
            DemoMAStrategy(short_window=20, long_window=50),
            DemoMAStrategy(short_window=5, long_window=20),
        ]

        results = []

        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            for strategy in strategies:
                result = backtest_engine.run(strategy)
                results.append(result)

        # Each strategy should produce different results
        returns = [r["metrics"]["total_return"] for r in results]
        assert len(set(returns)) > 1  # Not all the same

    def test_walk_forward_analysis(self, backtest_engine, mock_market_data):
        """Test walk-forward optimization."""
        strategy = DemoMAStrategy()

        # Define walk-forward parameters
        window_size = 180  # days
        step_size = 30  # days

        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            wf_results = backtest_engine.walk_forward_analysis(
                strategy, window_size=window_size, step_size=step_size
            )

        assert "in_sample_results" in wf_results
        assert "out_sample_results" in wf_results
        assert "optimal_parameters" in wf_results

    def test_monte_carlo_validation(self, backtest_engine, mock_market_data):
        """Test Monte Carlo simulation for result validation."""
        strategy = DemoMAStrategy()
        n_simulations = 100

        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            base_results = backtest_engine.run(strategy)

            mc_results = backtest_engine.monte_carlo_validation(
                strategy, n_simulations=n_simulations
            )

        assert len(mc_results) == n_simulations
        assert "confidence_intervals" in mc_results[0]
        assert "percentiles" in mc_results[0]

    def test_data_quality_checks(self, backtest_engine):
        """Test data quality validation in pipeline."""
        # Create data with quality issues
        bad_data = pd.DataFrame(
            {
                "Open": [100, np.nan, 102],
                "High": [105, 106, 107],
                "Low": [99, 100, 101],
                "Close": [103, 104, -105],  # Negative price
                "Volume": [1e6, 0, 1.2e6],  # Zero volume
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        # Validation should catch issues
        with pytest.raises(ValueError, match="Data quality"):
            backtest_engine.validate_data({"TEST": bad_data})

    def test_result_persistence(self, backtest_engine, mock_market_data, tmp_path):
        """Test saving and loading backtest results."""
        strategy = DemoMAStrategy()

        # Run backtest
        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            results = backtest_engine.run(strategy)

        # Save results
        results_file = tmp_path / "backtest_results.joblib"
        backtest_engine.save_results(results, results_file)
        assert results_file.exists()

        # Load results
        loaded_results = backtest_engine.load_results(results_file)

        # Verify loaded results match original
        assert loaded_results["metrics"]["total_return"] == results["metrics"]["total_return"]
        assert len(loaded_results["trades"]) == len(results["trades"])

    @pytest.mark.slow
    def test_large_universe_backtest(self, backtest_config):
        """Test backtesting with large universe of stocks."""
        # Create large universe
        large_universe = [f"STOCK_{i}" for i in range(100)]

        # Create engine
        engine = PortfolioBacktestEngine(universe=large_universe, config=backtest_config)

        # Create mock data for large universe
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        large_data = {}

        for symbol in large_universe:
            np.random.seed(hash(symbol) % 10000)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.015, len(dates))))
            large_data[symbol] = pd.DataFrame(
                {"Close": prices, "Volume": np.random.uniform(1e5, 1e7, len(dates))}, index=dates
            )

        strategy = DemoMAStrategy()

        # Run backtest
        with patch.object(engine, "load_data", return_value=large_data):
            results = engine.run(strategy)

        assert results is not None
        assert "metrics" in results

    def test_benchmark_comparison(self, backtest_engine, mock_market_data):
        """Test strategy performance against benchmark."""
        strategy = DemoMAStrategy()

        # Add benchmark data
        benchmark_data = pd.DataFrame(
            {
                "Close": 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 252))),
            },
            index=pd.date_range("2023-01-01", periods=252),
        )

        with patch.object(backtest_engine, "load_data", return_value=mock_market_data):
            with patch.object(backtest_engine, "load_benchmark", return_value=benchmark_data):
                results = backtest_engine.run_with_benchmark(strategy, benchmark="SPY")

        assert "benchmark_comparison" in results
        assert "alpha" in results["benchmark_comparison"]
        assert "beta" in results["benchmark_comparison"]
        assert "tracking_error" in results["benchmark_comparison"]
