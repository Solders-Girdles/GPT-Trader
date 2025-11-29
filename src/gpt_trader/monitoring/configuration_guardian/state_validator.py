"""State validator monitor for configuration guardian."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from gpt_trader.core import Balance, Position

from .base import ConfigurationMonitor
from .models import BaselineSnapshot, DriftEvent
from .responses import DriftResponse


class StateValidator(ConfigurationMonitor):
    """Validates configuration changes against current trading state."""

    def __init__(self, baseline_snapshot: BaselineSnapshot):
        self.baseline = baseline_snapshot

    def update_baseline(self, baseline_snapshot: BaselineSnapshot) -> None:
        """Refresh validator baseline after intentional config updates."""
        self.baseline = baseline_snapshot

    def check_changes(self) -> list[DriftEvent]:
        """Validate configuration state against trading invariants."""
        return []

    def validate_config_against_state(
        self,
        new_config_dict: dict[str, Any],
        current_balances: list[Balance],
        current_positions: list[Position],
        current_equity: Decimal | None,
    ) -> list[DriftEvent]:
        """Validate proposed config changes against live trading state."""
        events: list[DriftEvent] = []

        new_symbols = set(new_config_dict.get("symbols", []))
        new_max_leverage = new_config_dict.get("max_leverage", 3)
        new_position_size = new_config_dict.get("max_position_size", Decimal("1000"))
        new_profile = new_config_dict.get("profile")

        baseline_symbols = set(self.baseline.active_symbols)
        removed_symbols = baseline_symbols - new_symbols

        active_symbols = {pos.symbol for pos in current_positions if hasattr(pos, "symbol")}
        removed_with_positions = removed_symbols & active_symbols

        if removed_with_positions:
            events.append(
                DriftEvent(
                    timestamp=datetime.now(UTC),
                    component=self.monitor_name,
                    drift_type="symbols_remove_active_positions",
                    severity="critical",
                    details={
                        "removed_symbols": list(removed_with_positions),
                        "message": f"Cannot remove {removed_with_positions} - active positions exist",
                    },
                    suggested_response=DriftResponse.EMERGENCY_SHUTDOWN,
                    applied_response=DriftResponse.EMERGENCY_SHUTDOWN,
                )
            )

        current_leverage = self._calculate_current_leverage(current_positions, current_equity)
        if current_leverage > new_max_leverage:
            events.append(
                DriftEvent(
                    timestamp=datetime.now(UTC),
                    component=self.monitor_name,
                    drift_type="leverage_violation_current_positions",
                    severity="high",
                    details={
                        "current_leverage": float(current_leverage),
                        "new_max_leverage": new_max_leverage,
                        "message": "Current positions exceed new leverage limit",
                    },
                    suggested_response=DriftResponse.REDUCE_ONLY,
                    applied_response=DriftResponse.REDUCE_ONLY,
                )
            )

        current_exposure = sum(
            abs(float(getattr(pos, "size", 0))) * float(getattr(pos, "price", 0))
            for pos in current_positions
            if hasattr(pos, "size") and hasattr(pos, "price")
        )

        if current_exposure > float(new_position_size):
            events.append(
                DriftEvent(
                    timestamp=datetime.now(UTC),
                    component=self.monitor_name,
                    drift_type="position_size_violation_current_exposure",
                    severity="high",
                    details={
                        "current_exposure": current_exposure,
                        "new_max_position_size": float(new_position_size),
                        "message": "Current exposure exceeds new position limit",
                    },
                    suggested_response=DriftResponse.REDUCE_ONLY,
                    applied_response=DriftResponse.REDUCE_ONLY,
                )
            )

        if str(new_profile) != str(self.baseline.profile):
            events.append(
                DriftEvent(
                    timestamp=datetime.now(UTC),
                    component=self.monitor_name,
                    drift_type="profile_changed_during_runtime",
                    severity="critical",
                    details={
                        "old_profile": str(self.baseline.profile),
                        "new_profile": str(new_profile),
                        "message": "Profile changes during runtime not supported",
                    },
                    suggested_response=DriftResponse.EMERGENCY_SHUTDOWN,
                    applied_response=DriftResponse.EMERGENCY_SHUTDOWN,
                )
            )

        return events

    def get_current_state(self) -> dict[str, Any]:
        """Get current state validator status."""
        return {"baseline_snapshot_timestamp": self.baseline.timestamp.isoformat()}

    @property
    def monitor_name(self) -> str:
        return "state_validator"

    def _calculate_current_leverage(
        self, positions: list[Position], equity: Decimal | None
    ) -> Decimal:
        """Calculate current leverage across all positions."""
        if not equity or equity <= 0:
            return Decimal("0")

        total_exposure = Decimal("0")
        for pos in positions:
            if hasattr(pos, "size") and hasattr(pos, "price"):
                size = abs(float(pos.size))
                price = float(pos.price)
                exposure = Decimal(str(size * price))
                total_exposure += exposure

        return (total_exposure / equity) if total_exposure > 0 else Decimal("0")


__all__ = ["StateValidator"]
