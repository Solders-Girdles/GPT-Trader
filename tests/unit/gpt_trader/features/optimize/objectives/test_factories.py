"""Unit tests for objective factory methods."""

from gpt_trader.features.optimize.objectives.composite import WeightedObjective
from gpt_trader.features.optimize.objectives.factories import (
    create_execution_quality_objective,
    create_perpetuals_objective,
    create_risk_averse_objective,
    create_streak_resilient_objective,
    create_tail_risk_aware_objective,
    create_time_efficient_objective,
)


class TestCreateRiskAverseObjective:
    def test_returns_weighted_objective(self):
        """Test factory returns WeightedObjective."""
        objective = create_risk_averse_objective()
        assert isinstance(objective, WeightedObjective)
        assert objective.name == "risk_averse"

    def test_has_correct_components(self):
        """Test objective has expected components."""
        objective = create_risk_averse_objective()
        component_names = [c[0].name for c in objective.components]

        assert "sortino_ratio" in component_names
        assert "var_95_daily" in component_names
        assert "drawdown_recovery" in component_names

    def test_has_constraints(self):
        """Test objective has constraints."""
        objective = create_risk_averse_objective(max_drawdown_pct=15.0, max_var_95=5.0)

        constraint_names = [c.name for c in objective.constraints]
        assert "max_drawdown" in constraint_names
        assert "max_var" in constraint_names

    def test_custom_parameters(self):
        """Test custom parameters are applied."""
        objective = create_risk_averse_objective(
            max_drawdown_pct=10.0, max_var_95=3.0, min_trades=20
        )

        # Find drawdown constraint
        dd_constraint = next(c for c in objective.constraints if c.name == "max_drawdown")
        assert dd_constraint.threshold == 10.0


class TestCreateExecutionQualityObjective:
    def test_returns_weighted_objective(self):
        """Test factory returns WeightedObjective."""
        objective = create_execution_quality_objective()
        assert isinstance(objective, WeightedObjective)
        assert objective.name == "execution_focused"

    def test_has_correct_components(self):
        """Test objective has expected components."""
        objective = create_execution_quality_objective()
        component_names = [c[0].name for c in objective.components]

        assert "sharpe_ratio" in component_names
        assert "execution_quality" in component_names
        assert "cost_adjusted_return" in component_names

    def test_has_execution_constraints(self):
        """Test objective has execution-related constraints."""
        objective = create_execution_quality_objective(max_slippage_bps=10.0, min_fill_rate=80.0)

        constraint_names = [c.name for c in objective.constraints]
        assert "max_slippage" in constraint_names
        assert "min_fill_rate" in constraint_names


class TestCreateTimeEfficientObjective:
    def test_returns_weighted_objective(self):
        """Test factory returns WeightedObjective."""
        objective = create_time_efficient_objective()
        assert isinstance(objective, WeightedObjective)
        assert objective.name == "time_efficient"

    def test_has_correct_components(self):
        """Test objective has expected components."""
        objective = create_time_efficient_objective()
        component_names = [c[0].name for c in objective.components]

        assert "time_efficiency" in component_names
        assert "sharpe_ratio" in component_names
        assert "total_return" in component_names


class TestCreateStreakResilientObjective:
    def test_returns_weighted_objective(self):
        """Test factory returns WeightedObjective."""
        objective = create_streak_resilient_objective()
        assert isinstance(objective, WeightedObjective)
        assert objective.name == "streak_resilient"

    def test_has_correct_components(self):
        """Test objective has expected components."""
        objective = create_streak_resilient_objective()
        component_names = [c[0].name for c in objective.components]

        assert "sharpe_ratio" in component_names
        assert "streak_consistency" in component_names
        assert "win_rate" in component_names

    def test_has_streak_constraints(self):
        """Test objective has streak-related constraints."""
        objective = create_streak_resilient_objective(max_consecutive_losses=5, min_win_rate=50.0)

        constraint_names = [c.name for c in objective.constraints]
        assert "max_streak" in constraint_names
        assert "min_win_rate" in constraint_names


class TestCreatePerpetualsObjective:
    def test_returns_weighted_objective(self):
        """Test factory returns WeightedObjective."""
        objective = create_perpetuals_objective()
        assert isinstance(objective, WeightedObjective)
        assert objective.name == "perpetuals_optimized"

    def test_has_correct_components(self):
        """Test objective has expected components."""
        objective = create_perpetuals_objective()
        component_names = [c[0].name for c in objective.components]

        assert "sharpe_ratio" in component_names
        assert "funding_adjusted_return" in component_names
        assert "leverage_adjusted_return" in component_names
        assert "calmar_ratio" in component_names

    def test_has_leverage_constraints(self):
        """Test objective has leverage constraints."""
        objective = create_perpetuals_objective(max_leverage=5.0)

        constraint_names = [c.name for c in objective.constraints]
        assert "max_leverage" in constraint_names

    def test_circuit_breaker_constraint_default(self):
        """Test circuit breaker constraint is added by default."""
        objective = create_perpetuals_objective(allow_circuit_breakers=False)

        constraint_names = [c.name for c in objective.constraints]
        assert "no_circuit_breakers" in constraint_names

    def test_circuit_breaker_constraint_optional(self):
        """Test circuit breaker constraint can be disabled."""
        objective = create_perpetuals_objective(allow_circuit_breakers=True)

        constraint_names = [c.name for c in objective.constraints]
        assert "no_circuit_breakers" not in constraint_names


class TestCreateTailRiskAwareObjective:
    def test_returns_weighted_objective(self):
        """Test factory returns WeightedObjective."""
        objective = create_tail_risk_aware_objective()
        assert isinstance(objective, WeightedObjective)
        assert objective.name == "tail_risk_aware"

    def test_has_correct_components(self):
        """Test objective has expected components."""
        objective = create_tail_risk_aware_objective()
        component_names = [c[0].name for c in objective.components]

        assert "tail_risk_adjusted_return" in component_names
        assert "sortino_ratio" in component_names
        assert "calmar_ratio" in component_names

    def test_has_tail_risk_constraints(self):
        """Test objective has tail risk constraints."""
        objective = create_tail_risk_aware_objective(max_var_99=8.0, max_drawdown_pct=25.0)

        constraint_names = [c.name for c in objective.constraints]
        assert "max_var_99" in constraint_names
        assert "max_drawdown" in constraint_names
