"""Unit tests for optimization types."""

import pytest
from gpt_trader.features.optimize.types import (
    OptimizationConfig,
    ParameterDefinition,
    ParameterSpace,
    ParameterType,
)


class TestParameterDefinition:
    def test_integer_validation(self):
        """Test integer parameter validation."""
        # Valid
        param = ParameterDefinition(
            name="test_int",
            parameter_type=ParameterType.INTEGER,
            low=1,
            high=10,
        )
        assert param.name == "test_int"
        
        # Missing bounds
        with pytest.raises(ValueError, match="requires low and high"):
            ParameterDefinition(
                name="test_int",
                parameter_type=ParameterType.INTEGER,
                low=1,
            )
            
        # Invalid bounds
        with pytest.raises(ValueError, match="low must be less than high"):
            ParameterDefinition(
                name="test_int",
                parameter_type=ParameterType.INTEGER,
                low=10,
                high=1,
            )

    def test_categorical_validation(self):
        """Test categorical parameter validation."""
        # Valid
        param = ParameterDefinition(
            name="test_cat",
            parameter_type=ParameterType.CATEGORICAL,
            choices=["a", "b"],
        )
        assert param.choices == ["a", "b"]
        
        # Missing choices
        with pytest.raises(ValueError, match="requires at least 2 choices"):
            ParameterDefinition(
                name="test_cat",
                parameter_type=ParameterType.CATEGORICAL,
                choices=["a"],
            )


class TestParameterSpace:
    def test_parameter_aggregation(self):
        """Test that all parameters are aggregated correctly."""
        p1 = ParameterDefinition("p1", ParameterType.INTEGER, low=1, high=10)
        p2 = ParameterDefinition("p2", ParameterType.FLOAT, low=0.1, high=1.0)
        p3 = ParameterDefinition("p3", ParameterType.CATEGORICAL, choices=["x", "y"])
        
        space = ParameterSpace(
            strategy_parameters=[p1],
            risk_parameters=[p2],
            simulation_parameters=[p3],
        )
        
        assert len(space.all_parameters) == 3
        assert space.parameter_count == 3
        assert space.get_parameter("p1") == p1
        assert space.get_parameter("p2") == p2
        assert space.get_parameter("p3") == p3
        assert space.get_parameter("missing") is None


class TestOptimizationConfig:
    def test_validation(self):
        """Test configuration validation."""
        space = ParameterSpace()
        
        # Valid
        config = OptimizationConfig(
            study_name="test",
            parameter_space=space,
            objective_name="sharpe",
        )
        assert config.direction == "maximize"
        
        # Invalid direction
        with pytest.raises(ValueError, match="direction must be"):
            OptimizationConfig(
                study_name="test",
                parameter_space=space,
                objective_name="sharpe",
                direction="invalid",
            )
            
        # Invalid sampler
        with pytest.raises(ValueError, match="Unknown sampler_type"):
            OptimizationConfig(
                study_name="test",
                parameter_space=space,
                objective_name="sharpe",
                sampler_type="invalid",
            )
