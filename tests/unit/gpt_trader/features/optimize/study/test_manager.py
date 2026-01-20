"""Unit tests for OptimizationStudyManager."""

from unittest.mock import Mock

import optuna
import pytest

from gpt_trader.features.optimize.study.manager import OptimizationStudyManager
from gpt_trader.features.optimize.types import (
    OptimizationConfig,
    ParameterDefinition,
    ParameterSpace,
    ParameterType,
)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    space = ParameterSpace(
        strategy_parameters=[ParameterDefinition("p1", ParameterType.INTEGER, low=1, high=10)]
    )
    return OptimizationConfig(
        study_name="test_study",
        parameter_space=space,
        objective_name="sharpe",
    )


class TestOptimizationStudyManager:
    def test_create_study(self, mock_config, monkeypatch: pytest.MonkeyPatch):
        """Test study creation."""
        manager = OptimizationStudyManager(mock_config)

        mock_create = Mock()
        monkeypatch.setattr(optuna, "create_study", mock_create)
        _study = manager.create_or_load_study()

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["study_name"] == "test_study"
        assert call_kwargs["direction"] == "maximize"

    def test_suggest_parameters(self, mock_config):
        """Test parameter suggestion."""
        manager = OptimizationStudyManager(mock_config)

        mock_trial = Mock(spec=optuna.Trial)
        mock_trial.suggest_int.return_value = 5

        params = manager.suggest_parameters(mock_trial)

        assert params["p1"] == 5
        mock_trial.suggest_int.assert_called_with("p1", 1, 10, step=1, log=False)
