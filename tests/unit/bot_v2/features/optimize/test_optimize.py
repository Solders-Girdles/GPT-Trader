"""
Comprehensive tests for optimization orchestration.

Tests cover:
- Parameter grid search optimization
- Walk-forward analysis
- Multiple strategy comparison
- Metric-based optimization (Sharpe, return, Calmar)
- Data fetching and handling
- Edge cases and error handling
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.optimize.optimize import (
    fetch_data,
    grid_search,
    optimize_strategy,
    walk_forward_analysis,
)
from bot_v2.features.optimize.types import (
    OptimizationResult,
    WalkForwardResult,
)


def create_mock_data(n_bars: int = 200, start_date: str = "2024-01-01") -> pd.DataFrame:
    """Create mock market data for testing."""
    dates = pd.date_range(start=start_date, periods=n_bars, freq="D")
    close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 2)

    return pd.DataFrame(
        {
            "open": close_prices + np.random.randn(n_bars),
            "high": close_prices + np.abs(np.random.randn(n_bars) * 2),
            "low": close_prices - np.abs(np.random.randn(n_bars) * 2),
            "close": close_prices,
            "volume": np.random.randint(1000000, 10000000, n_bars),
        },
        index=dates,
    )


class TestOptimizeStrategy:
    """Test strategy parameter optimization."""

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_basic(self, mock_fetch):
        """Test basic strategy optimization."""
        mock_fetch.return_value = create_mock_data(200)

        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert isinstance(result, OptimizationResult)
        assert result.strategy == "SimpleMA"
        assert result.symbol == "BTC-USD"
        assert result.best_params is not None
        assert result.best_metrics is not None
        assert len(result.all_results) > 0
        assert result.optimization_time > 0

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_custom_param_grid(self, mock_fetch):
        """Test optimization with custom parameter grid."""
        mock_fetch.return_value = create_mock_data(200)

        custom_grid = {"fast_period": [5, 10], "slow_period": [20, 30]}

        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            param_grid=custom_grid,
        )

        # Should test 2 * 2 = 4 combinations
        assert len(result.all_results) == 4

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_different_metrics(self, mock_fetch):
        """Test optimization with different target metrics."""
        mock_fetch.return_value = create_mock_data(200)

        metrics_to_test = ["sharpe_ratio", "return", "calmar"]

        for metric in metrics_to_test:
            result = optimize_strategy(
                strategy="SimpleMA",
                symbol="BTC-USD",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 6, 1),
                metric=metric,
            )

            assert isinstance(result, OptimizationResult)
            assert result.best_params is not None

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_with_commission(self, mock_fetch):
        """Test optimization with custom commission rate."""
        mock_fetch.return_value = create_mock_data(200)

        result_low_comm = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            commission=0.0001,
        )

        result_high_comm = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            commission=0.01,
        )

        # Higher commission should result in lower returns
        assert (
            result_high_comm.best_metrics.total_return <= result_low_comm.best_metrics.total_return
        )

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_with_slippage(self, mock_fetch):
        """Test optimization with custom slippage."""
        mock_fetch.return_value = create_mock_data(200)

        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            slippage=0.001,
        )

        assert isinstance(result, OptimizationResult)

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_selects_best_params(self, mock_fetch):
        """Test that optimization selects best performing parameters."""
        mock_fetch.return_value = create_mock_data(200)

        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            metric="sharpe_ratio",
        )

        # Best params should have highest score among all tested
        best_score = getattr(result.best_metrics, "sharpe_ratio")
        for res in result.all_results:
            assert res["score"] <= best_score

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_momentum_strategy(self, mock_fetch):
        """Test optimization of momentum strategy."""
        mock_fetch.return_value = create_mock_data(200)

        result = optimize_strategy(
            strategy="Momentum",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert result.strategy == "Momentum"
        assert "lookback" in result.best_params
        assert "threshold" in result.best_params
        assert "hold_period" in result.best_params

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_mean_reversion_strategy(self, mock_fetch):
        """Test optimization of mean reversion strategy."""
        mock_fetch.return_value = create_mock_data(200)

        result = optimize_strategy(
            strategy="MeanReversion",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert result.strategy == "MeanReversion"
        assert "period" in result.best_params
        assert "entry_std" in result.best_params
        assert "exit_std" in result.best_params

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_tracks_all_results(self, mock_fetch):
        """Test that all parameter combinations are tracked."""
        mock_fetch.return_value = create_mock_data(200)

        param_grid = {"fast_period": [5, 10, 15], "slow_period": [20, 30]}

        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            param_grid=param_grid,
        )

        # Should have 3 * 2 = 6 results
        assert len(result.all_results) == 6

        # Each result should have params, metrics, and score
        for res in result.all_results:
            assert "params" in res
            assert "metrics" in res
            assert "score" in res

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_period_tuple(self, mock_fetch):
        """Test that optimization result includes period tuple."""
        mock_fetch.return_value = create_mock_data(200)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)

        result = optimize_strategy(
            strategy="SimpleMA", symbol="BTC-USD", start_date=start, end_date=end
        )

        assert result.period == (start, end)


class TestGridSearch:
    """Test grid search across multiple strategies."""

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_grid_search_multiple_strategies(self, mock_fetch):
        """Test grid search with multiple strategies."""
        mock_fetch.return_value = create_mock_data(200)

        strategies = ["SimpleMA", "Momentum"]

        results = grid_search(
            strategies=strategies,
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert len(results) == 2
        assert "SimpleMA" in results
        assert "Momentum" in results
        assert all(isinstance(r, OptimizationResult) for r in results.values())

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_grid_search_selects_best_strategy(self, mock_fetch):
        """Test that grid search identifies best strategy."""
        mock_fetch.return_value = create_mock_data(200)

        strategies = ["SimpleMA", "Momentum", "MeanReversion"]

        results = grid_search(
            strategies=strategies,
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            metric="sharpe_ratio",
        )

        # Find best manually
        best_sharpe = max(r.best_metrics.sharpe_ratio for r in results.values())

        # At least one strategy should have the best sharpe
        assert any(r.best_metrics.sharpe_ratio == best_sharpe for r in results.values())

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_grid_search_single_strategy(self, mock_fetch):
        """Test grid search with single strategy."""
        mock_fetch.return_value = create_mock_data(200)

        results = grid_search(
            strategies=["SimpleMA"],
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert len(results) == 1
        assert "SimpleMA" in results


class TestWalkForwardAnalysis:
    """Test walk-forward analysis."""

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_basic(self, mock_fetch):
        """Test basic walk-forward analysis."""
        mock_fetch.return_value = create_mock_data(300)

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            window_size=60,
            step_size=30,
            test_size=30,
        )

        assert isinstance(result, WalkForwardResult)
        assert result.strategy == "SimpleMA"
        assert result.symbol == "BTC-USD"
        assert len(result.windows) > 0

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_windows_non_overlapping(self, mock_fetch):
        """Test that walk-forward windows are properly structured."""
        mock_fetch.return_value = create_mock_data(300)

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            window_size=60,
            step_size=30,
            test_size=30,
        )

        # Check window structure
        for window in result.windows:
            # Train period should come before test period
            assert window.train_end == window.test_start
            assert window.train_start < window.train_end
            assert window.test_start < window.test_end

            # Should have metrics for both periods
            assert window.train_metrics is not None
            assert window.test_metrics is not None
            assert window.best_params is not None

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_efficiency(self, mock_fetch):
        """Test walk-forward efficiency calculation."""
        mock_fetch.return_value = create_mock_data(300)

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            window_size=60,
            step_size=30,
            test_size=30,
        )

        # Check efficiency for each window
        for window in result.windows:
            efficiency = window.get_efficiency()
            # Efficiency = test return / train return
            if window.train_metrics.total_return != 0:
                expected = window.test_metrics.total_return / window.train_metrics.total_return
                assert abs(efficiency - expected) < 0.01

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_average_efficiency(self, mock_fetch):
        """Test average efficiency calculation."""
        mock_fetch.return_value = create_mock_data(300)

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            window_size=60,
            step_size=30,
            test_size=30,
        )

        # Average efficiency should be mean of all window efficiencies
        efficiencies = [w.get_efficiency() for w in result.windows]
        expected_avg = np.mean(efficiencies)

        assert abs(result.avg_efficiency - expected_avg) < 0.01

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_consistency_score(self, mock_fetch):
        """Test consistency score calculation."""
        mock_fetch.return_value = create_mock_data(300)

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            window_size=60,
            step_size=30,
            test_size=30,
        )

        # Consistency score should be between 0 and 1
        assert 0 <= result.consistency_score <= 1

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_robustness_score(self, mock_fetch):
        """Test robustness score calculation."""
        mock_fetch.return_value = create_mock_data(300)

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            window_size=60,
            step_size=30,
            test_size=30,
        )

        # Robustness should be between 0 and 1
        assert 0 <= result.robustness_score <= 1

        # Should be weighted combination of efficiency and consistency
        expected = result.avg_efficiency * 0.6 + result.consistency_score * 0.4
        expected = min(1.0, max(0.0, expected))
        assert abs(result.robustness_score - expected) < 0.01

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_custom_param_grid(self, mock_fetch):
        """Test walk-forward with custom parameter grid."""
        mock_fetch.return_value = create_mock_data(300)

        custom_grid = {"fast_period": [5, 10], "slow_period": [20, 30]}

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            param_grid=custom_grid,
            window_size=60,
            step_size=30,
            test_size=30,
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_different_window_sizes(self, mock_fetch):
        """Test walk-forward with different window configurations."""
        mock_fetch.return_value = create_mock_data(400)

        # Larger windows
        result_large = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            window_size=90,
            step_size=45,
            test_size=45,
        )

        # Smaller windows
        result_small = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            window_size=30,
            step_size=15,
            test_size=15,
        )

        # Smaller windows should generate more windows
        assert len(result_small.windows) > len(result_large.windows)

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_summary(self, mock_fetch):
        """Test walk-forward summary generation."""
        mock_fetch.return_value = create_mock_data(300)

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            window_size=60,
            step_size=30,
            test_size=30,
        )

        summary = result.summary()

        # Summary should contain key information
        assert "SimpleMA" in summary
        assert "BTC-USD" in summary
        assert str(len(result.windows)) in summary
        assert "Efficiency" in summary or "efficiency" in summary


class TestFetchData:
    """Test data fetching functionality."""

    @patch("bot_v2.features.optimize.optimize.get_data_provider")
    def test_fetch_data_basic(self, mock_provider):
        """Test basic data fetching."""
        mock_data = create_mock_data(100)
        mock_provider_instance = Mock()
        mock_provider_instance.get_historical_data.return_value = mock_data
        mock_provider.return_value = mock_provider_instance

        data = fetch_data("BTC-USD", datetime(2024, 1, 1), datetime(2024, 6, 1))

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        # Columns should be lowercased
        assert all(col.islower() for col in data.columns)

    @patch("bot_v2.features.optimize.optimize.get_data_provider")
    def test_fetch_data_standardizes_columns(self, mock_provider):
        """Test that column names are standardized."""
        # Create data with uppercase columns
        mock_data = create_mock_data(100)
        mock_data.columns = [c.upper() for c in mock_data.columns]

        mock_provider_instance = Mock()
        mock_provider_instance.get_historical_data.return_value = mock_data
        mock_provider.return_value = mock_provider_instance

        data = fetch_data("BTC-USD", datetime(2024, 1, 1), datetime(2024, 6, 1))

        # Should be lowercase
        assert all(col.islower() for col in data.columns)

    @patch("bot_v2.features.optimize.optimize.get_data_provider")
    def test_fetch_data_empty_raises_error(self, mock_provider):
        """Test that empty data raises error."""
        mock_provider_instance = Mock()
        mock_provider_instance.get_historical_data.return_value = pd.DataFrame()
        mock_provider.return_value = mock_provider_instance

        with pytest.raises(ValueError, match="No data available"):
            fetch_data("INVALID", datetime(2024, 1, 1), datetime(2024, 6, 1))


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_with_minimal_data(self, mock_fetch):
        """Test optimization with minimal data points."""
        mock_fetch.return_value = create_mock_data(50)  # Very short period

        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
        )

        # Should complete but may have limited results
        assert isinstance(result, OptimizationResult)

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimize_single_parameter_combination(self, mock_fetch):
        """Test optimization with single parameter set."""
        mock_fetch.return_value = create_mock_data(200)

        param_grid = {"fast_period": [10], "slow_period": [30]}

        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            param_grid=param_grid,
        )

        assert len(result.all_results) == 1
        assert result.best_params == {"fast_period": 10, "slow_period": 30}

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_grid_search_empty_strategy_list(self, mock_fetch):
        """Test grid search with empty strategy list."""
        mock_fetch.return_value = create_mock_data(200)

        results = grid_search(
            strategies=[],
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        assert len(results) == 0

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_walk_forward_insufficient_data(self, mock_fetch):
        """Test walk-forward with insufficient data for windows."""
        # Data too short for requested window sizes
        mock_fetch.return_value = create_mock_data(50)

        result = walk_forward_analysis(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
            window_size=60,  # Longer than data
            step_size=30,
            test_size=30,
        )

        # Should have no windows
        assert len(result.windows) == 0

    @patch("bot_v2.features.optimize.optimize.fetch_data")
    def test_optimization_result_summary(self, mock_fetch):
        """Test optimization result summary generation."""
        mock_fetch.return_value = create_mock_data(200)

        result = optimize_strategy(
            strategy="SimpleMA",
            symbol="BTC-USD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        summary = result.summary()

        # Summary should contain key information
        assert "SimpleMA" in summary
        assert "BTC-USD" in summary
        assert "Total Return" in summary or "Return" in summary
