"""Integration tests for optimization trial results and parameter typing."""

from __future__ import annotations

import pytest

from gpt_trader.features.optimize.study.manager import OptimizationStudyManager
from gpt_trader.features.optimize.types import (
    OptimizationConfig,
    ParameterDefinition,
    ParameterSpace,
    ParameterType,
)
from tests.integration.optimize_feature_test_base import make_batch_runner

pytestmark = pytest.mark.integration


class TestOptimizationResultsAndParameters:
    """Tests for optimization result objects and parameter spaces."""

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

        runner = make_batch_runner()

        trial = study.ask()
        params = study_manager.suggest_parameters(trial)
        result = await runner.run_trial(0, params)

        assert result.backtest_result is not None
        assert result.risk_metrics is not None
        assert result.trade_statistics is not None
        assert result.duration_seconds >= 0
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_parameter_space_with_multiple_types(self) -> None:
        """Test optimization with integer and float parameters."""
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

        runner = make_batch_runner()

        for trial_num in range(config.number_of_trials):
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)

            assert isinstance(params["short_ma_period"], int)
            assert isinstance(params["position_fraction"], float)
            assert 3 <= params["short_ma_period"] <= 7
            assert 0.05 <= params["position_fraction"] <= 0.2

            result = await runner.run_trial(trial_num, params)
            study.tell(trial, result.objective_value)

        assert study.best_trial is not None
