"""Edge tests for composite objectives."""

from types import SimpleNamespace

import pytest

from gpt_trader.features.optimize.objectives.composite import Constraint, WeightedObjective


class DummyObjective:
    def __init__(self, value: float, *, direction: str = "maximize", feasible: bool = True) -> None:
        self._value = value
        self.direction = direction
        self._feasible = feasible

    def calculate(self, result, risk_metrics, trade_statistics) -> float:
        return self._value

    def is_feasible(self, result, risk_metrics, trade_statistics) -> bool:
        return self._feasible


def _make_payloads(**overrides):
    result = SimpleNamespace(total_return=overrides.get("total_return", 1.0))
    risk_metrics = SimpleNamespace(max_drawdown_pct=overrides.get("max_drawdown_pct", 5.0))
    trade_statistics = SimpleNamespace(total_trades=overrides.get("total_trades", 10))
    return result, risk_metrics, trade_statistics


def test_constraint_invalid_operator_raises() -> None:
    with pytest.raises(ValueError, match="operator must be one of"):
        Constraint(name="bad", metric="total_return", operator="ne", threshold=1.0)


def test_constraint_unknown_metric_raises() -> None:
    constraint = Constraint(name="unknown", metric="not_real", operator="gt", threshold=1.0)
    result, risk_metrics, trade_statistics = _make_payloads()
    with pytest.raises(ValueError, match="Unknown metric"):
        constraint.is_satisfied(result, risk_metrics, trade_statistics)


def test_constraint_returns_false_when_value_none() -> None:
    constraint = Constraint(name="none", metric="total_return", operator="gt", threshold=1.0)
    result = SimpleNamespace(total_return=None)
    risk_metrics = SimpleNamespace()
    trade_statistics = SimpleNamespace()

    assert not constraint.is_satisfied(result, risk_metrics, trade_statistics)


@pytest.mark.parametrize(
    ("operator", "value", "threshold", "expected"),
    [
        ("lt", 1.0, 2.0, True),
        ("le", 2.0, 2.0, True),
        ("gt", 3.0, 2.0, True),
        ("ge", 2.0, 2.0, True),
        ("eq", 2.0, 2.0, True),
    ],
)
def test_constraint_operator_logic(
    operator: str, value: float, threshold: float, expected: bool
) -> None:
    constraint = Constraint(
        name="op", metric="total_return", operator=operator, threshold=threshold
    )
    result = SimpleNamespace(total_return=value)
    risk_metrics = SimpleNamespace()
    trade_statistics = SimpleNamespace()

    assert constraint.is_satisfied(result, risk_metrics, trade_statistics) is expected


def test_weighted_objective_raises_on_empty_components() -> None:
    with pytest.raises(ValueError, match="At least one component objective is required"):
        WeightedObjective(name="empty", components=[])


def test_weighted_objective_flips_sign_for_minimize() -> None:
    objective = DummyObjective(2.0, direction="minimize")
    composite = WeightedObjective(name="min", components=[(objective, 1.0)])
    result, risk_metrics, trade_statistics = _make_payloads()

    assert composite.calculate(result, risk_metrics, trade_statistics) == -2.0


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_weighted_objective_propagates_inf(value: float) -> None:
    objective = DummyObjective(value)
    composite = WeightedObjective(name="inf", components=[(objective, 1.0)])
    result, risk_metrics, trade_statistics = _make_payloads()

    assert composite.calculate(result, risk_metrics, trade_statistics) == value


def test_weighted_objective_infeasible_component() -> None:
    objective = DummyObjective(1.0, feasible=False)
    composite = WeightedObjective(name="bad", components=[(objective, 1.0)])
    result, risk_metrics, trade_statistics = _make_payloads()

    assert not composite.is_feasible(result, risk_metrics, trade_statistics)


def test_weighted_objective_infeasible_constraint() -> None:
    objective = DummyObjective(1.0, feasible=True)
    constraint = Constraint(name="min_return", metric="total_return", operator="gt", threshold=10.0)
    composite = WeightedObjective(
        name="constrained", components=[(objective, 1.0)], constraints=[constraint]
    )
    result, risk_metrics, trade_statistics = _make_payloads(total_return=1.0)

    assert not composite.is_feasible(result, risk_metrics, trade_statistics)
