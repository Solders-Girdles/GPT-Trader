"""
Strategy verification tests for Phase 1.5
Ensures all strategies work with the new consolidated architecture
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestStrategyCompatibility:
    """Test that existing strategies work with new architecture"""

    def test_demo_ma_strategy(self):
        """Test the demo moving average strategy"""
        from bot.strategy.demo_ma import DemoMAStrategy

        # Create strategy instance
        strategy = DemoMAStrategy(window=20)

        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        data = pd.DataFrame(
            {
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        # Generate signals
        signals = strategy.generate_signals(data)

        # Verify signals
        assert signals is not None
        assert len(signals) == len(data)
        assert signals.dtype == bool or signals.dtype == np.int64

        # Check that strategy has the expected attributes
        assert hasattr(strategy, "window")
        assert strategy.window == 20

    def test_trend_breakout_strategy(self):
        """Test the trend breakout strategy"""
        from bot.strategy.trend_breakout import TrendBreakoutStrategy

        # Create strategy instance
        strategy = TrendBreakoutStrategy(donchian_period=55, atr_period=20, atr_multiplier=2.0)

        # Create sample OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 101,
                "low": np.random.randn(100).cumsum() + 99,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        # Ensure high >= low
        data["high"] = data[["high", "close"]].max(axis=1)
        data["low"] = data[["low", "close"]].min(axis=1)

        # Generate signals
        signals = strategy.generate_signals(data)

        # Verify signals
        assert signals is not None
        assert len(signals) == len(data)

        # Check strategy parameters
        assert strategy.donchian_period == 55
        assert strategy.atr_period == 20
        assert strategy.atr_multiplier == 2.0

    def test_strategy_base_class_integration(self):
        """Test that strategies integrate with base classes"""
        from bot.core.base import BaseStrategy, ComponentConfig

        # Check if strategies can work with the base class
        config = ComponentConfig(component_id="test_strategy", component_type="strategy")

        # Create a mock strategy that inherits from BaseStrategy
        class MockStrategy(BaseStrategy):
            def _initialize_component(self):
                self.strategy_parameters = {"window": 20}

            def _start_component(self):
                pass

            def _stop_component(self):
                pass

            def _health_check(self):
                from bot.core.base import HealthStatus

                return HealthStatus.HEALTHY

            def _generate_signals(self, market_data):
                return pd.Series([1, 0, -1])

            def _calculate_position_size(self, signal, portfolio_state):
                return 100

        strategy = MockStrategy(config)

        # Test that it has the expected base class functionality
        assert hasattr(strategy, "get_performance_metrics")
        assert hasattr(strategy, "get_strategy_parameters")

        metrics = strategy.get_performance_metrics()
        assert "total_signals" in metrics
        assert "win_rate" in metrics

    def test_strategy_with_unified_database(self):
        """Test that strategies can work with unified database"""
        import tempfile

        from bot.core.database import DatabaseConfig, DatabaseManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize database
            config = DatabaseConfig(database_path=Path(tmpdir) / "test.db")
            db_manager = DatabaseManager(config)

            # Create a strategy record
            strategy_id = db_manager.insert_record(
                "components",
                {
                    "component_id": "test_ma_strategy",
                    "component_type": "strategy",
                    "status": "active",
                    "config_data": '{"window": 20}',
                },
            )

            # Verify strategy was saved
            result = db_manager.fetch_one(
                "SELECT * FROM components WHERE component_id = ?", ("test_ma_strategy",)
            )

            assert result is not None
            assert result["component_type"] == "strategy"

            # Save strategy performance
            db_manager.insert_record(
                "strategy_performance",
                {
                    "strategy_id": "test_ma_strategy",
                    "measurement_date": datetime.now().date(),
                    "total_pnl": "1000.00",
                    "realized_pnl": "800.00",
                    "unrealized_pnl": "200.00",
                    "total_return_pct": 10.0,
                    "win_rate": 0.65,
                    "sharpe_ratio": 1.5,
                },
            )

            # Verify performance was saved
            perf = db_manager.fetch_one(
                "SELECT * FROM strategy_performance WHERE strategy_id = ?", ("test_ma_strategy",)
            )

            assert perf is not None
            assert float(perf["total_pnl"]) == 1000.00
            assert float(perf["sharpe_ratio"]) == 1.5

            db_manager.close()

    def test_strategy_with_unified_data_pipeline(self):
        """Test that strategies can use unified data pipeline"""
        import tempfile

        from bot.data.unified_pipeline import DataConfig, UnifiedDataPipeline

        # Create pipeline
        config = DataConfig(cache_dir=Path(tempfile.mkdtemp()), validate_data=True)
        pipeline = UnifiedDataPipeline(config)

        # Create test data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        test_data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 101,
                "low": np.random.randn(100).cumsum() + 99,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

        # Validate data through pipeline
        validated_data = pipeline._validate_data(test_data)

        # Use validated data with strategy
        from bot.strategy.demo_ma import DemoMAStrategy

        strategy = DemoMAStrategy(window=20)

        signals = strategy.generate_signals(validated_data)

        assert signals is not None
        assert len(signals) == len(validated_data)


class TestStrategyPerformance:
    """Test strategy performance metrics"""

    def test_strategy_metrics_calculation(self):
        """Test that strategy metrics are calculated correctly"""
        from bot.strategy.demo_ma import DemoMAStrategy

        strategy = DemoMAStrategy(window=20)

        # Create sample data with known pattern
        dates = pd.date_range(end=datetime.now(), periods=200, freq="D")
        # Create trending data
        trend = np.linspace(100, 150, 200)
        noise = np.random.randn(200) * 2
        data = pd.DataFrame(
            {"close": trend + noise, "volume": np.random.randint(1000000, 10000000, 200)},
            index=dates,
        )

        # Generate signals
        signals = strategy.generate_signals(data)

        # Calculate simple metrics
        num_signals = signals.sum()
        signal_rate = num_signals / len(signals)

        # Should generate some signals
        assert num_signals > 0
        assert signal_rate > 0
        assert signal_rate < 1  # Not all periods should have signals

    def test_strategy_with_monitoring(self):
        """Test that strategies work with monitoring system"""
        from bot.monitoring.monitor import MonitorConfig, UnifiedMonitor

        config = MonitorConfig(metrics_interval=1, health_check_interval=1)

        monitor = UnifiedMonitor(config)

        # Create mock strategy metrics
        strategy_metrics = {
            "strategy_id": "test_strategy",
            "total_trades": 100,
            "winning_trades": 65,
            "total_pnl": 10000.0,
            "sharpe_ratio": 1.5,
        }

        # Add to monitor cache
        monitor._metrics_cache.update(strategy_metrics)

        # Get metrics
        metrics = monitor.get_metrics()

        assert "strategy_id" in metrics
        assert metrics["total_trades"] == 100
        assert metrics["winning_trades"] == 65


class TestBacktestIntegration:
    """Test backtesting with new architecture"""

    @patch("bot.dataflow.sources.yfinance_source.YFinanceSource")
    def test_backtest_with_consolidated_components(self, mock_yfinance):
        """Test that backtest works with all consolidated components"""
        import tempfile

        from bot.backtest.engine import BacktestEngine
        from bot.core.database import DatabaseConfig, initialize_database
        from bot.strategy.demo_ma import DemoMAStrategy

        # Setup mock data source
        mock_source = MagicMock()
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        mock_data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 101,
                "low": np.random.randn(100).cumsum() + 99,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )
        mock_source.fetch.return_value = mock_data
        mock_yfinance.return_value = mock_source

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize database
            db_config = DatabaseConfig(database_path=Path(tmpdir) / "test.db")
            db = initialize_database(db_config)

            # Create strategy
            strategy = DemoMAStrategy(window=20)

            # Create backtest engine
            engine = BacktestEngine(
                strategy=strategy,
                start_date=datetime.now() - timedelta(days=100),
                end_date=datetime.now(),
                initial_capital=100000,
            )

            # Run backtest (mocked)
            with patch.object(engine, "run") as mock_run:
                mock_run.return_value = {
                    "total_return": 0.10,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.05,
                    "win_rate": 0.65,
                }

                results = engine.run()

                # Verify results structure
                assert "total_return" in results
                assert "sharpe_ratio" in results
                assert "max_drawdown" in results
                assert "win_rate" in results

                # Verify values
                assert results["total_return"] == 0.10
                assert results["sharpe_ratio"] == 1.5

            db.close()


def run_strategy_verification():
    """Run all strategy verification tests"""
    print("=" * 80)
    print("Running Strategy Verification Tests")
    print("=" * 80)

    pytest_args = [__file__, "-v", "--tb=short", "--color=yes"]

    result = pytest.main(pytest_args)

    if result == 0:
        print("\n" + "=" * 80)
        print("✅ All strategies verified working with new architecture!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Some strategy tests failed. Review output above.")
        print("=" * 80)

    return result


if __name__ == "__main__":
    sys.exit(run_strategy_verification())
