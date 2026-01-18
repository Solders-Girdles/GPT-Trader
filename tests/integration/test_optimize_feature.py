"""Integration tests for the optimization feature suite.

Tests the end-to-end flow: parameter space -> Optuna study -> batch backtest -> results.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.engine.bar_runner import IHistoricalDataProvider
from gpt_trader.core import Candle
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy,
    PerpsStrategyConfig,
)
from gpt_trader.features.optimize.objectives.single import TotalReturnObjective
from gpt_trader.features.optimize.runner.batch_runner import BatchBacktestRunner
from gpt_trader.features.optimize.study.manager import OptimizationStudyManager
from gpt_trader.features.optimize.types import (
    OptimizationConfig,
    ParameterDefinition,
    ParameterSpace,
    ParameterType,
)

pytestmark = pytest.mark.integration


class SyntheticDataProvider(IHistoricalDataProvider):
    """
    Generates synthetic candle data for testing.

    Creates a predictable price series with trends to enable
    strategy parameter optimization testing.
    """

    def __init__(self, base_price: Decimal = Decimal("50000")):
        self.base_price = base_price

    async def get_candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Generate synthetic candles with a trending pattern."""
        candles = []
        current_time = start
        price = self.base_price

        # Granularity to timedelta
        granularity_map = {
            "ONE_MINUTE": timedelta(minutes=1),
            "FIVE_MINUTE": timedelta(minutes=5),
            "ONE_HOUR": timedelta(hours=1),
        }
        delta = granularity_map.get(granularity, timedelta(minutes=5))

        bar_index = 0
        while current_time < end:
            # Create a trending pattern: up for 20 bars, down for 10, repeat
            cycle_position = bar_index % 30
            if cycle_position < 20:
                # Uptrend: price increases
                price = price + Decimal("50")
            else:
                # Downtrend: price decreases
                price = price - Decimal("100")

            # Add some volatility
            high = price + Decimal("20")
            low = price - Decimal("20")
            open_price = price - Decimal("10")
            close = price

            candles.append(
                Candle(
                    ts=current_time,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=Decimal("100"),
                )
            )

            current_time += delta
            bar_index += 1

        return candles


def create_strategy_factory(params: dict) -> BaselinePerpsStrategy:
    """Factory to create strategy instances from parameter dict."""
    config = PerpsStrategyConfig(
        short_ma_period=params.get("short_ma_period", 5),
        long_ma_period=params.get("long_ma_period", 20),
        position_fraction=params.get("position_fraction", 0.1),
    )
    return BaselinePerpsStrategy(config=config)


class TestOptimizationEndToEnd:
    """End-to-end tests for the optimization pipeline."""

    @pytest.mark.asyncio
    async def test_optimization_runs_multiple_trials(self) -> None:
        """Test that optimization executes multiple trials and collects results."""
        # Setup parameter space
        parameter_space = ParameterSpace(
            strategy_parameters=[
                ParameterDefinition(
                    name="short_ma_period",
                    parameter_type=ParameterType.INTEGER,
                    low=3,
                    high=10,
                    description="Short MA period",
                ),
                ParameterDefinition(
                    name="long_ma_period",
                    parameter_type=ParameterType.INTEGER,
                    low=15,
                    high=30,
                    description="Long MA period",
                ),
            ]
        )

        # Setup optimization config
        config = OptimizationConfig(
            study_name="test_optimization_e2e",
            parameter_space=parameter_space,
            objective_name="total_return",
            direction="maximize",
            number_of_trials=5,  # Small number for test speed
            sampler_type="random",  # Random sampler for reproducibility
            pruner_type=None,  # No pruning for this test
            seed=42,
        )

        # Setup study manager
        study_manager = OptimizationStudyManager(config)
        study = study_manager.create_or_load_study()

        # Setup batch runner
        data_provider = SyntheticDataProvider()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)  # 1 day of data

        runner = BatchBacktestRunner(
            data_provider=data_provider,
            symbols=["BTC-USD"],
            granularity="FIVE_MINUTE",
            start_date=start_date,
            end_date=end_date,
            strategy_factory=create_strategy_factory,
            objective=TotalReturnObjective(min_trades=0),  # Allow 0 trades for this test
        )

        # Run optimization
        results = []
        for trial_num in range(config.number_of_trials):
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)

            result = await runner.run_trial(trial_num, params)
            results.append(result)

            # Report to Optuna
            study.tell(trial, result.objective_value)

        # Verify results
        assert len(results) == 5, "Should have run 5 trials"
        assert all(r.backtest_result is not None for r in results), "All trials should have results"
        assert study.best_trial is not None, "Study should have a best trial"

    @pytest.mark.asyncio
    async def test_optimization_identifies_best_parameters(self) -> None:
        """Test that optimization correctly identifies better parameters."""
        parameter_space = ParameterSpace(
            strategy_parameters=[
                ParameterDefinition(
                    name="short_ma_period",
                    parameter_type=ParameterType.INTEGER,
                    low=3,
                    high=8,
                ),
            ]
        )

        config = OptimizationConfig(
            study_name="test_best_params",
            parameter_space=parameter_space,
            objective_name="total_return",
            direction="maximize",
            number_of_trials=3,
            sampler_type="random",
            pruner_type=None,
            seed=123,
        )

        study_manager = OptimizationStudyManager(config)
        study = study_manager.create_or_load_study()

        data_provider = SyntheticDataProvider()
        runner = BatchBacktestRunner(
            data_provider=data_provider,
            symbols=["BTC-USD"],
            granularity="FIVE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            strategy_factory=create_strategy_factory,
            objective=TotalReturnObjective(min_trades=0),
        )

        for trial_num in range(config.number_of_trials):
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)
            result = await runner.run_trial(trial_num, params)
            study.tell(trial, result.objective_value)

        # Verify best trial exists and has valid parameters
        assert study.best_trial is not None
        assert "short_ma_period" in study.best_params
        assert 3 <= study.best_params["short_ma_period"] <= 8

    @pytest.mark.asyncio
    async def test_trial_result_contains_metrics(self) -> None:
        """Test that trial results include risk metrics and trade statistics."""
        parameter_space = ParameterSpace(
            strategy_parameters=[
                ParameterDefinition(
                    name="short_ma_period",
                    parameter_type=ParameterType.INTEGER,
                    low=5,
                    high=6,
                ),
            ]
        )

        config = OptimizationConfig(
            study_name="test_metrics",
            parameter_space=parameter_space,
            objective_name="total_return",
            direction="maximize",
            number_of_trials=1,
            sampler_type="random",
            seed=42,
        )

        study_manager = OptimizationStudyManager(config)
        study = study_manager.create_or_load_study()

        data_provider = SyntheticDataProvider()
        runner = BatchBacktestRunner(
            data_provider=data_provider,
            symbols=["BTC-USD"],
            granularity="FIVE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            strategy_factory=create_strategy_factory,
            objective=TotalReturnObjective(min_trades=0),
        )

        trial = study.ask()
        params = study_manager.suggest_parameters(trial)
        result = await runner.run_trial(0, params)

        # Verify comprehensive metrics
        assert result.backtest_result is not None
        assert result.risk_metrics is not None
        assert result.trade_statistics is not None
        assert result.duration_seconds >= 0
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_parameter_space_with_multiple_types(self) -> None:
        """Test optimization with integer, float, and categorical parameters."""
        parameter_space = ParameterSpace(
            strategy_parameters=[
                ParameterDefinition(
                    name="short_ma_period",
                    parameter_type=ParameterType.INTEGER,
                    low=3,
                    high=7,
                ),
                ParameterDefinition(
                    name="position_fraction",
                    parameter_type=ParameterType.FLOAT,
                    low=0.05,
                    high=0.2,
                ),
            ]
        )

        config = OptimizationConfig(
            study_name="test_multi_param_types",
            parameter_space=parameter_space,
            objective_name="total_return",
            direction="maximize",
            number_of_trials=3,
            sampler_type="random",
            seed=42,
        )

        study_manager = OptimizationStudyManager(config)
        study = study_manager.create_or_load_study()

        data_provider = SyntheticDataProvider()
        runner = BatchBacktestRunner(
            data_provider=data_provider,
            symbols=["BTC-USD"],
            granularity="FIVE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            strategy_factory=create_strategy_factory,
            objective=TotalReturnObjective(min_trades=0),
        )

        for trial_num in range(config.number_of_trials):
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)

            # Verify parameter types
            assert isinstance(params["short_ma_period"], int)
            assert isinstance(params["position_fraction"], float)
            assert 3 <= params["short_ma_period"] <= 7
            assert 0.05 <= params["position_fraction"] <= 0.2

            result = await runner.run_trial(trial_num, params)
            study.tell(trial, result.objective_value)

        assert study.best_trial is not None


class TestOptimizationFailures:
    """Tests for optimization error handling."""

    @pytest.mark.asyncio
    async def test_trial_handles_strategy_error_gracefully(self) -> None:
        """Test that a failing strategy doesn't crash the optimization."""

        def failing_strategy_factory(params: dict) -> BaselinePerpsStrategy:
            """Factory that creates a strategy that will work but may have edge cases."""
            config = PerpsStrategyConfig(
                short_ma_period=params.get("short_ma_period", 5),
                long_ma_period=params.get("long_ma_period", 20),
            )
            return BaselinePerpsStrategy(config=config)

        parameter_space = ParameterSpace(
            strategy_parameters=[
                ParameterDefinition(
                    name="short_ma_period",
                    parameter_type=ParameterType.INTEGER,
                    low=2,
                    high=5,
                ),
            ]
        )

        config = OptimizationConfig(
            study_name="test_error_handling",
            parameter_space=parameter_space,
            objective_name="total_return",
            direction="maximize",
            number_of_trials=2,
            sampler_type="random",
            seed=42,
        )

        study_manager = OptimizationStudyManager(config)
        study = study_manager.create_or_load_study()

        data_provider = SyntheticDataProvider()
        runner = BatchBacktestRunner(
            data_provider=data_provider,
            symbols=["BTC-USD"],
            granularity="FIVE_MINUTE",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            strategy_factory=failing_strategy_factory,
            objective=TotalReturnObjective(min_trades=0),
        )

        # Should not raise even if some trials fail
        for trial_num in range(config.number_of_trials):
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)
            result = await runner.run_trial(trial_num, params)
            study.tell(trial, result.objective_value)

        # Optimization should complete
        assert len(study.trials) == 2
