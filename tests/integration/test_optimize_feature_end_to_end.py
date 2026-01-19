"""Integration tests for the optimization feature suite (end-to-end flow)."""

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


class TestOptimizationEndToEnd:
    """End-to-end tests for the optimization pipeline."""

    @pytest.mark.asyncio
    async def test_optimization_runs_multiple_trials(self) -> None:
        """Test that optimization executes multiple trials and collects results."""
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

        config = OptimizationConfig(
            study_name="test_optimization_e2e",
            parameter_space=parameter_space,
            objective_name="total_return",
            direction="maximize",
            number_of_trials=5,
            sampler_type="random",
            pruner_type=None,
            seed=42,
        )

        study_manager = OptimizationStudyManager(config)
        study = study_manager.create_or_load_study()

        runner = make_batch_runner()

        results = []
        for trial_num in range(config.number_of_trials):
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)

            result = await runner.run_trial(trial_num, params)
            results.append(result)

            study.tell(trial, result.objective_value)

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

        runner = make_batch_runner()

        for trial_num in range(config.number_of_trials):
            trial = study.ask()
            params = study_manager.suggest_parameters(trial)
            result = await runner.run_trial(trial_num, params)
            study.tell(trial, result.objective_value)

        assert study.best_trial is not None
        assert "short_ma_period" in study.best_params
        assert 3 <= study.best_params["short_ma_period"] <= 8
