from __future__ import annotations

from datetime import datetime
from typing import Any


def record_selection_change(
    orchestrator: Any, old_selection: list[str], new_selection: list[str]
) -> None:
    """Record a selection change audit entry.

    Stores old/new selections and basic diff for traceability.
    """
    removed = [sid for sid in old_selection if sid not in new_selection]
    added = [sid for sid in new_selection if sid not in old_selection]

    data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "old_selection": list(old_selection),
        "new_selection": list(new_selection),
        "added": added,
        "removed": removed,
    }

    orchestrator._record_operation("selection_change", data)

    try:
        orchestrator.observability.log_decision(
            decision_type="selection_change",
            decision_data=data,
            metadata={"mode": orchestrator.config.mode.value, "source": "audit"},
        )
    except Exception:
        pass


def record_rebalance(orchestrator: Any, changes: dict[str, dict[str, float]]) -> None:
    """Record a rebalance audit entry with per-symbol changes."""
    total_abs_change = sum(abs(v.get("change", 0.0)) for v in changes.values())
    data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "changes": changes,
        "total_abs_change": total_abs_change,
    }

    orchestrator._record_operation("rebalance", data)

    try:
        orchestrator.observability.log_decision(
            decision_type="rebalance",
            decision_data=data,
            metadata={"mode": orchestrator.config.mode.value, "source": "audit"},
        )
    except Exception:
        pass


def record_trade_blocked(orchestrator: Any, reason: str, details: dict[str, Any]) -> None:
    """Record a trade-blocked audit entry with reason and details."""
    data = {"timestamp": datetime.now().isoformat(), "reason": reason, **dict(details)}

    orchestrator._record_operation("trade_blocked", data)

    try:
        orchestrator.observability.log_decision(
            decision_type="trade_blocked",
            decision_data=data,
            metadata={"mode": orchestrator.config.mode.value, "source": "audit"},
        )
    except Exception:
        pass
