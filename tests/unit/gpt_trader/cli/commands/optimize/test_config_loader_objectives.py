"""Unit tests for optimize CLI config loader objectives."""

from __future__ import annotations

import pytest

from gpt_trader.cli.commands.optimize.config_loader import (
    OBJECTIVE_PRESETS,
    ConfigValidationError,
    create_objective_from_preset,
    get_objective_direction,
)


class TestCreateObjectiveFromPreset:
    def test_creates_simple_objective(self):
        """Test creating simple objective."""
        objective = create_objective_from_preset("sharpe")
        assert objective.name == "sharpe_ratio"

    def test_creates_composite_objective(self):
        """Test creating composite objective with factory."""
        objective = create_objective_from_preset("risk_averse")
        assert objective.name == "risk_averse"

    def test_passes_kwargs_to_factory(self):
        """Test kwargs are passed to factory."""
        objective = create_objective_from_preset("risk_averse", max_drawdown_pct=10.0)
        constraint_names = [constraint.name for constraint in objective.constraints]
        assert "max_drawdown" in constraint_names

    def test_raises_for_unknown_preset(self):
        """Test raises error for unknown preset."""
        with pytest.raises(ConfigValidationError, match="Unknown objective"):
            create_objective_from_preset("nonexistent")


class TestGetObjectiveDirection:
    def test_maximize_presets(self):
        """Test maximize direction presets."""
        assert get_objective_direction("sharpe") == "maximize"
        assert get_objective_direction("sortino") == "maximize"
        assert get_objective_direction("total_return") == "maximize"

    def test_minimize_presets(self):
        """Test minimize direction presets."""
        assert get_objective_direction("max_drawdown") == "minimize"

    def test_raises_for_unknown_preset(self):
        """Test raises error for unknown preset."""
        with pytest.raises(ConfigValidationError):
            get_objective_direction("nonexistent")


class TestObjectivePresets:
    def test_all_presets_have_direction(self):
        """Test all presets have a direction defined."""
        for preset_name in OBJECTIVE_PRESETS:
            factory, direction = OBJECTIVE_PRESETS[preset_name]
            assert direction in ("maximize", "minimize")

    def test_all_presets_can_be_created(self):
        """Test all presets can be instantiated."""
        for preset_name in OBJECTIVE_PRESETS:
            objective = create_objective_from_preset(preset_name)
            assert objective is not None
