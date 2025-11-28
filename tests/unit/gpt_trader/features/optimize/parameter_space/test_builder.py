"""Unit tests for ParameterSpaceBuilder."""

from gpt_trader.features.optimize.parameter_space.builder import ParameterSpaceBuilder
from gpt_trader.features.optimize.types import ParameterType


class TestParameterSpaceBuilder:
    def test_build_empty(self):
        """Test building empty space."""
        space = ParameterSpaceBuilder().build()
        assert space.parameter_count == 0

    def test_add_integer(self):
        """Test adding integer parameter."""
        space = (
            ParameterSpaceBuilder()
            .add_integer("p1", 1, 10, category="strategy")
            .build()
        )
        assert space.parameter_count == 1
        param = space.get_parameter("p1")
        assert param.parameter_type == ParameterType.INTEGER
        assert param.low == 1
        assert param.high == 10
        assert len(space.strategy_parameters) == 1

    def test_add_float(self):
        """Test adding float parameter."""
        space = (
            ParameterSpaceBuilder()
            .add_float("p1", 0.1, 1.0, category="risk")
            .build()
        )
        assert space.parameter_count == 1
        param = space.get_parameter("p1")
        assert param.parameter_type == ParameterType.FLOAT
        assert param.low == 0.1
        assert param.high == 1.0
        assert len(space.risk_parameters) == 1

    def test_add_categorical(self):
        """Test adding categorical parameter."""
        space = (
            ParameterSpaceBuilder()
            .add_categorical("p1", ["a", "b"], category="simulation")
            .build()
        )
        assert space.parameter_count == 1
        param = space.get_parameter("p1")
        assert param.parameter_type == ParameterType.CATEGORICAL
        assert param.choices == ["a", "b"]
        assert len(space.simulation_parameters) == 1

    def test_with_defaults(self):
        """Test adding default parameter sets."""
        space = (
            ParameterSpaceBuilder()
            .with_strategy_defaults()
            .with_risk_defaults()
            .with_simulation_defaults()
            .build()
        )
        assert space.parameter_count > 0
        assert len(space.strategy_parameters) > 0
        assert len(space.risk_parameters) > 0
        assert len(space.simulation_parameters) > 0
