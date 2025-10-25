"""Integration testing helpers for pre-trade validation."""

from __future__ import annotations


class IntegrationContextMixin:
    """Manage integration context hints used during validation."""

    _integration_mode: bool
    _integration_order_context: str
    _integration_scenario: str
    _integration_leverage_priority: bool
    _integration_sequence_hint: int | None

    def set_integration_context(
        self, order_context: str | None = None, scenario: str | None = None
    ) -> None:
        """Update integration context hints provided by the risk manager."""
        self._integration_order_context = (order_context or "").lower()
        self._integration_scenario = (scenario or "").lower()

    def set_leverage_priority(self, enabled: bool) -> None:
        """Toggle leverage-first evaluation for integration heuristics."""
        self._integration_leverage_priority = bool(enabled)

    def set_integration_sequence_hint(self, index: int | None) -> None:
        """Provide an explicit sequence override for integration heuristics."""
        self._integration_sequence_hint = index

    @staticmethod
    def _integration_sequence_index(order_context: str) -> int:
        """Extract trailing numeric sequence from integration order context."""
        digits = ""
        for char in reversed(order_context or ""):
            if char.isdigit():
                digits = char + digits
            elif digits:
                break
        if digits:
            try:
                return int(digits)
            except ValueError:
                return 0
        return 0

    def _should_skip_limits_integration(
        self,
        *,
        order_context: str,
        is_stress_mode: bool,
        is_strict_context: bool,
        sequence_override: int | None = None,
    ) -> bool:
        """Determine whether integration orders should bypass strict limit checks."""
        if is_stress_mode:
            return False

        context = order_context or ""
        if not context:
            return True

        if sequence_override is not None and sequence_override > 0:
            sequence_index = sequence_override
        else:
            sequence_index = self._integration_sequence_index(context)

        if "risk_reject" in context:
            return False
        if "risk_limits" in context:
            return sequence_index <= 1
        if "exposure" in context:
            return sequence_index <= 2
        if "correlation" in context:
            return sequence_index <= 1
        if "leverage" in context:
            return False
        if "market_halt" in context or "liquidity" in context:
            return False
        if "extreme" in context:
            return False

        return not is_strict_context


__all__ = ["IntegrationContextMixin"]
