"""Golden-path validator for comparing live and simulated decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from gpt_trader.backtesting.types import ValidationDivergence, ValidationReport
from gpt_trader.utilities.datetime_helpers import utc_now

from .decision_logger import StrategyDecision

if TYPE_CHECKING:
    from gpt_trader.backtesting.simulation.broker import SimulatedBroker


@dataclass
class ValidationResult:
    """Result of validating a single decision pair."""

    matches: bool
    live_decision: StrategyDecision
    sim_decision: StrategyDecision | None = None
    divergence: ValidationDivergence | None = None
    confidence: Decimal = Decimal("100")  # 0-100 confidence in match
    details: dict[str, str] = field(default_factory=dict)


class GoldenPathValidator:
    """
    Validates that simulated decisions match live decisions.

    The golden-path validation ensures that a backtest produces
    the same decisions as live trading would, given identical inputs.

    Key validation points:
    1. Action match (BUY/SELL/HOLD)
    2. Quantity match (within tolerance)
    3. Price match (within tolerance)
    4. Risk check outcomes
    """

    def __init__(
        self,
        quantity_tolerance_pct: Decimal = Decimal("0.01"),  # 1%
        price_tolerance_pct: Decimal = Decimal("0.001"),  # 0.1%
        strict_action_match: bool = True,
    ):
        """
        Initialize validator.

        Args:
            quantity_tolerance_pct: Allowed quantity deviation (0.01 = 1%)
            price_tolerance_pct: Allowed price deviation
            strict_action_match: Require exact action match
        """
        self.quantity_tolerance = quantity_tolerance_pct
        self.price_tolerance = price_tolerance_pct
        self.strict_action_match = strict_action_match

        self._divergences: list[ValidationDivergence] = []
        self._total_comparisons = 0
        self._matching_comparisons = 0

    def validate_decision(
        self,
        live: StrategyDecision,
        simulated: StrategyDecision,
    ) -> ValidationResult:
        """
        Validate that a simulated decision matches a live decision.

        Args:
            live: Decision from live trading
            simulated: Decision from backtest simulation

        Returns:
            ValidationResult with match status and details
        """
        self._total_comparisons += 1
        details: dict[str, str] = {}
        confidence = Decimal("100")
        matches = True
        divergence_reason = ""

        # 1. Action match
        if live.action != simulated.action:
            if self.strict_action_match:
                matches = False
                divergence_reason = f"Action mismatch: live={live.action}, sim={simulated.action}"
                confidence = Decimal("0")
            else:
                details["action_warning"] = f"Action differs: {live.action} vs {simulated.action}"
                confidence -= Decimal("30")

        # 2. Quantity match (if action involves a trade)
        if live.action != "HOLD" and simulated.action != "HOLD":
            if live.target_quantity > 0 and simulated.target_quantity > 0:
                quantity_diff = abs(live.target_quantity - simulated.target_quantity)
                quantity_pct = (
                    quantity_diff / live.target_quantity
                    if live.target_quantity > 0
                    else Decimal("0")
                )

                if quantity_pct > self.quantity_tolerance:
                    matches = False
                    divergence_reason = (
                        f"Quantity mismatch: live={live.target_quantity}, "
                        f"sim={simulated.target_quantity} ({quantity_pct * 100:.2f}% diff)"
                    )
                    confidence -= Decimal("25")
                else:
                    details["quantity_check"] = f"Within tolerance: {quantity_pct * 100:.2f}%"

        # 3. Price match (for limit orders)
        if live.target_price and simulated.target_price:
            price_diff = abs(live.target_price - simulated.target_price)
            price_pct = price_diff / live.target_price if live.target_price > 0 else Decimal("0")

            if price_pct > self.price_tolerance:
                matches = False
                divergence_reason = (
                    f"Price mismatch: live={live.target_price}, "
                    f"sim={simulated.target_price} ({price_pct * 100:.4f}% diff)"
                )
                confidence -= Decimal("20")
            else:
                details["price_check"] = f"Within tolerance: {price_pct * 100:.4f}%"

        # 4. Risk check outcome match
        if live.risk_checks_passed != simulated.risk_checks_passed:
            matches = False
            divergence_reason = (
                f"Risk check mismatch: live={live.risk_checks_passed}, "
                f"sim={simulated.risk_checks_passed}"
            )
            confidence -= Decimal("25")

        # 5. Order type match
        if live.order_type != simulated.order_type:
            details["order_type_warning"] = (
                f"Order type differs: {live.order_type} vs {simulated.order_type}"
            )
            confidence -= Decimal("10")

        # Create result
        if matches:
            self._matching_comparisons += 1
            return ValidationResult(
                matches=True,
                live_decision=live,
                sim_decision=simulated,
                confidence=max(Decimal("0"), confidence),
                details=details,
            )

        # Create divergence record
        divergence = ValidationDivergence(
            cycle_id=live.cycle_id,
            symbol=live.symbol,
            timestamp=live.timestamp,
            live_action=live.action,
            live_quantity=live.target_quantity,
            live_price=live.target_price,
            sim_action=simulated.action,
            sim_quantity=simulated.target_quantity,
            sim_price=simulated.target_price,
            reason=divergence_reason,
            impact_pct=self._estimate_impact(live, simulated),
        )
        self._divergences.append(divergence)

        return ValidationResult(
            matches=False,
            live_decision=live,
            sim_decision=simulated,
            divergence=divergence,
            confidence=max(Decimal("0"), confidence),
            details=details,
        )

    def _estimate_impact(
        self,
        live: StrategyDecision,
        simulated: StrategyDecision,
    ) -> Decimal:
        """
        Estimate the financial impact of a divergence.

        Returns estimated impact as a percentage of position value.
        """
        # If actions differ completely (one traded, one didn't)
        if (live.action == "HOLD") != (simulated.action == "HOLD"):
            # Full position value at risk
            return Decimal("100")

        # If both are trades, estimate based on quantity/price differences
        if live.action != "HOLD" and simulated.action != "HOLD":
            quantity_impact = Decimal("0")
            if live.target_quantity > 0:
                quantity_impact = (
                    abs(live.target_quantity - simulated.target_quantity)
                    / live.target_quantity
                    * 100
                )

            price_impact = Decimal("0")
            if live.target_price and simulated.target_price and live.target_price > 0:
                price_impact = (
                    abs(live.target_price - simulated.target_price) / live.target_price * 100
                )

            return quantity_impact + price_impact

        return Decimal("0")

    def generate_report(self, cycle_id: str) -> ValidationReport:
        """
        Generate a validation report for a cycle.

        Args:
            cycle_id: Cycle ID to report on

        Returns:
            ValidationReport with statistics and divergences
        """
        cycle_divergences = [d for d in self._divergences if d.cycle_id == cycle_id]

        return ValidationReport(
            cycle_id=cycle_id,
            timestamp=utc_now(),
            total_decisions=self._total_comparisons,
            matching_decisions=self._matching_comparisons,
            divergences=cycle_divergences,
        )

    def get_match_rate(self) -> Decimal:
        """Get overall match rate as a percentage."""
        if self._total_comparisons == 0:
            return Decimal("100")
        return Decimal(self._matching_comparisons) / Decimal(self._total_comparisons) * 100

    def get_divergences(
        self,
        symbol: str | None = None,
        min_impact: Decimal | None = None,
    ) -> list[ValidationDivergence]:
        """
        Get recorded divergences with optional filtering.

        Args:
            symbol: Filter by symbol
            min_impact: Minimum impact percentage to include

        Returns:
            List of divergences
        """
        divergences = self._divergences

        if symbol:
            divergences = [d for d in divergences if d.symbol == symbol]

        if min_impact is not None:
            divergences = [
                d for d in divergences if d.impact_pct is not None and d.impact_pct >= min_impact
            ]

        return divergences

    def reset(self) -> None:
        """Reset validator state."""
        self._divergences = []
        self._total_comparisons = 0
        self._matching_comparisons = 0

    @property
    def total_comparisons(self) -> int:
        """Get total number of comparisons made."""
        return self._total_comparisons

    @property
    def matching_comparisons(self) -> int:
        """Get number of matching comparisons."""
        return self._matching_comparisons

    @property
    def divergence_count(self) -> int:
        """Get number of divergences recorded."""
        return len(self._divergences)


def replay_decisions_through_simulator(
    decisions: list[StrategyDecision],
    broker: SimulatedBroker,
    validator: GoldenPathValidator,
) -> list[ValidationResult]:
    """
    Replay recorded decisions through a simulator and validate.

    This is the core golden-path validation flow:
    1. Take recorded live decisions
    2. Replay the market conditions in the simulator
    3. Run the same strategy logic
    4. Compare the simulated decisions to live decisions

    Args:
        decisions: Live decisions to replay
        broker: SimulatedBroker to use for replay
        validator: Validator for comparison

    Returns:
        List of validation results
    """
    results: list[ValidationResult] = []

    for live_decision in decisions:
        # Set up market state in broker
        # Note: This requires the broker to have historical data loaded
        # matching the decision timestamps

        # For now, create a simulated decision with the same inputs
        # In full implementation, this would call the actual strategy
        sim_decision = StrategyDecision.create(
            cycle_id=live_decision.cycle_id,
            symbol=live_decision.symbol,
            equity=live_decision.equity,
            position_quantity=live_decision.position_quantity,
            position_side=live_decision.position_side,
            mark_price=live_decision.mark_price,
            recent_marks=live_decision.recent_marks,
        )

        # The strategy would be called here to populate sim_decision
        # For validation, we compare what the strategy *would* do

        result = validator.validate_decision(live_decision, sim_decision)
        results.append(result)

    return results
