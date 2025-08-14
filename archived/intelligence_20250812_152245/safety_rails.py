from __future__ import annotations


class SafetyRails:
    """Safety rails for strategy selector and allocation validation.

    Provides simple, testable constraints to keep allocations sane before
    running through portfolio optimization or execution layers.
    """

    def __init__(self, config: dict[str, float]) -> None:
        self.max_position_size = float(config.get("max_position_size", 0.2))
        self.max_portfolio_risk = float(config.get("max_portfolio_risk", 0.15))
        self.max_drawdown_limit = float(config.get("max_drawdown_limit", 0.1))
        self.emergency_stop_threshold = float(config.get("emergency_stop_threshold", 0.05))

    def validate_allocations(
        self,
        allocations: dict[str, float],
        risk_scores: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Validate allocations against safety constraints.

        Returns (is_ok, violations).
        """
        violations: list[str] = []

        # Individual caps
        for strategy, allocation in allocations.items():
            if allocation > self.max_position_size:
                violations.append(
                    f"Position size {allocation:.2%} exceeds limit {self.max_position_size:.2%} for {strategy}"
                )

        # Portfolio risk (simple weighted sum)
        portfolio_risk = sum(allocations.get(s, 0.0) * risk_scores.get(s, 0.0) for s in allocations)
        if portfolio_risk > self.max_portfolio_risk:
            violations.append(
                f"Portfolio risk {portfolio_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}"
            )

        # Allocations should approximately sum to 1.0
        total_allocation = sum(allocations.values())
        if abs(total_allocation - 1.0) > 0.01:
            violations.append(f"Total allocation {total_allocation:.2%} does not sum to 100%")

        return len(violations) == 0, violations

    def apply_safety_constraints(
        self,
        target_allocations: dict[str, float],
        risk_scores: dict[str, float],
    ) -> dict[str, float]:
        """Apply safety constraints to target allocations and renormalize.

        - Cap individual position sizes
        - Normalize to 1.0
        - Scale down if portfolio risk exceeds threshold
        """
        safe_allocations = dict(target_allocations)

        # Cap individual sizes
        for strategy in list(safe_allocations.keys()):
            safe_allocations[strategy] = min(safe_allocations[strategy], self.max_position_size)

        # Normalize to sum to 1.0
        total = sum(safe_allocations.values())
        if total > 0:
            safe_allocations = {k: v / total for k, v in safe_allocations.items()}

        # Check portfolio risk and reduce proportionally if needed, then renormalize
        portfolio_risk = sum(
            safe_allocations.get(s, 0.0) * risk_scores.get(s, 0.0) for s in safe_allocations
        )
        if portfolio_risk > self.max_portfolio_risk and portfolio_risk > 0:
            reduction_factor = self.max_portfolio_risk / portfolio_risk
            safe_allocations = {k: v * reduction_factor for k, v in safe_allocations.items()}
            total2 = sum(safe_allocations.values())
            if total2 > 0:
                safe_allocations = {k: v / total2 for k, v in safe_allocations.items()}

        return safe_allocations

    def should_block_trading_due_to_drawdown(self, portfolio_max_drawdown: float) -> bool:
        """Return True if portfolio-level drawdown guard should block trading.

        Uses `max_drawdown_limit` configured at initialization.
        """
        try:
            return float(portfolio_max_drawdown) > float(self.max_drawdown_limit)
        except Exception:
            return False
