"""Unit tests for OptimizationStudyManager."""

import builtins
import importlib
import sys
from unittest.mock import Mock

import pytest

from gpt_trader.features.optimize.study.manager import OptimizationStudyManager
from gpt_trader.features.optimize.types import (
    OptimizationConfig,
    ParameterDefinition,
    ParameterSpace,
    ParameterType,
)

# optuna is an optional extra (gpt-trader[optimize]); skip this module when absent
# rather than failing collection. The source module (manager.py) guards the import.
optuna = pytest.importorskip("optuna")


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

    def test_missing_optuna_raises_install_extra_message(
        self, mock_config, monkeypatch: pytest.MonkeyPatch
    ):
        """Study manager imports cleanly without Optuna and fails with install guidance."""
        import gpt_trader.features.optimize.study.manager as manager_module

        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "optuna" or name.startswith("optuna."):
                raise ImportError("blocked optuna for test")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with monkeypatch.context() as context:
                for module_name in list(sys.modules):
                    if module_name == "optuna" or module_name.startswith("optuna."):
                        context.delitem(sys.modules, module_name, raising=False)
                context.setattr(builtins, "__import__", blocked_import)

                missing_module = importlib.reload(manager_module)

                with pytest.raises(
                    missing_module.MissingOptimizeDependencyError,
                    match=r"gpt-trader\[optimize\]",
                ):
                    missing_module.OptimizationStudyManager(mock_config)
        finally:
            importlib.reload(manager_module)
