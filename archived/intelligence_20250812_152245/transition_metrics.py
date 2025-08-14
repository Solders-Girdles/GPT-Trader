from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransitionMetrics:
    turnover_rate: float
    slippage_cost: float
    allocation_churn: float
    transition_smoothness: float


class SlippageModel:
    """Interface for slippage models used by TransitionSmoothnessCalculator."""

    def estimate_slippage(self, trade_size: float) -> float:  # pragma: no cover - simple interface
        raise NotImplementedError


class TransitionSmoothnessCalculator:
    """Calculate measurable transition smoothness."""

    def __init__(self, slippage_model: SlippageModel) -> None:
        self.slippage_model = slippage_model

    def calculate_turnover_rate(
        self, current_allocations: dict[str, float], target_allocations: dict[str, float]
    ) -> float:
        turnover = 0.0
        keys = set(current_allocations) | set(target_allocations)
        for key in keys:
            current = current_allocations.get(key, 0.0)
            target = target_allocations.get(key, 0.0)
            turnover += abs(target - current)
        return turnover / 2.0

    def calculate_allocation_churn(
        self,
        current_allocations: dict[str, float],
        target_allocations: dict[str, float],
        threshold: float = 0.01,
    ) -> float:
        keys = set(current_allocations) | set(target_allocations)
        if not keys:
            return 0.0
        changed = 0
        for key in keys:
            current = current_allocations.get(key, 0.0)
            target = target_allocations.get(key, 0.0)
            if abs(target - current) > threshold:
                changed += 1
        return changed / len(keys)

    def calculate_slippage_cost(
        self,
        current_allocations: dict[str, float],
        target_allocations: dict[str, float],
        portfolio_value: float,
    ) -> float:
        total_cost = 0.0
        keys = set(current_allocations) | set(target_allocations)
        for key in keys:
            current = current_allocations.get(key, 0.0)
            target = target_allocations.get(key, 0.0)
            trade_size = abs(target - current) * float(portfolio_value)
            if trade_size > 0:
                slippage_rate = self.slippage_model.estimate_slippage(trade_size)
                total_cost += trade_size * slippage_rate
        return total_cost

    def calculate_smoothness_score(
        self,
        current_allocations: dict[str, float],
        target_allocations: dict[str, float],
        portfolio_value: float,
    ) -> float:
        turnover_rate = self.calculate_turnover_rate(current_allocations, target_allocations)
        allocation_churn = self.calculate_allocation_churn(current_allocations, target_allocations)
        slippage_cost = self.calculate_slippage_cost(
            current_allocations, target_allocations, portfolio_value
        )

        turnover_score = max(0.0, 1.0 - turnover_rate / 0.1)
        churn_score = max(0.0, 1.0 - allocation_churn / 0.5)
        slippage_score = max(0.0, 1.0 - slippage_cost / (portfolio_value * 0.001))

        smoothness = 0.4 * turnover_score + 0.3 * churn_score + 0.3 * slippage_score
        return max(0.0, min(1.0, smoothness))
