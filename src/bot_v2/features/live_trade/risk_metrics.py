"""Utilities for aggregating risk metrics emitted by the live risk engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from collections.abc import Callable, Iterable

from ...persistence.event_store import EventStore


@dataclass(frozen=True)
class RiskMetricSnapshot:
    """Single snapshot of risk metrics captured by the event store."""

    timestamp: datetime
    equity: Decimal
    total_notional: Decimal
    exposure_pct: Decimal
    max_leverage: Decimal
    daily_pnl: Decimal
    daily_pnl_pct: Decimal
    reduce_only: bool
    kill_switch: bool


@dataclass(frozen=True)
class RiskMetricsSummary:
    """Aggregated statistics derived from a series of risk metric snapshots."""

    count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    latest: RiskMetricSnapshot | None
    exposure_pct_max: Decimal
    exposure_pct_avg: Decimal
    leverage_max: Decimal
    equity_min: Decimal
    equity_max: Decimal
    total_notional_max: Decimal
    daily_pnl_min: Decimal
    daily_pnl_max: Decimal
    daily_pnl_pct_min: Decimal
    daily_pnl_pct_max: Decimal
    reduce_only_active: bool
    kill_switch_active: bool

    @classmethod
    def empty(cls) -> RiskMetricsSummary:
        zero = Decimal("0")
        return cls(
            count=0,
            first_timestamp=None,
            last_timestamp=None,
            latest=None,
            exposure_pct_max=zero,
            exposure_pct_avg=zero,
            leverage_max=zero,
            equity_min=zero,
            equity_max=zero,
            total_notional_max=zero,
            daily_pnl_min=zero,
            daily_pnl_max=zero,
            daily_pnl_pct_min=zero,
            daily_pnl_pct_max=zero,
            reduce_only_active=False,
            kill_switch_active=False,
        )

    @classmethod
    def from_points(cls, points: Iterable[RiskMetricSnapshot]) -> RiskMetricsSummary:
        points_list = sorted(points, key=lambda p: p.timestamp)
        if not points_list:
            return cls.empty()

        count = len(points_list)
        exposure_values = [p.exposure_pct for p in points_list]
        leverage_values = [p.max_leverage for p in points_list]
        equities = [p.equity for p in points_list]
        notionals = [p.total_notional for p in points_list]
        pnl_values = [p.daily_pnl for p in points_list]
        pnl_pct_values = [p.daily_pnl_pct for p in points_list]

        latest = points_list[-1]

        return cls(
            count=count,
            first_timestamp=points_list[0].timestamp,
            last_timestamp=latest.timestamp,
            latest=latest,
            exposure_pct_max=max(exposure_values),
            exposure_pct_avg=sum(exposure_values) / Decimal(count),
            leverage_max=max(leverage_values),
            equity_min=min(equities),
            equity_max=max(equities),
            total_notional_max=max(notionals),
            daily_pnl_min=min(pnl_values),
            daily_pnl_max=max(pnl_values),
            daily_pnl_pct_min=min(pnl_pct_values),
            daily_pnl_pct_max=max(pnl_pct_values),
            reduce_only_active=latest.reduce_only,
            kill_switch_active=latest.kill_switch,
        )


class RiskMetricsAggregator:
    """Aggregate risk metrics snapshots stored in the event store."""

    def __init__(
        self,
        event_store: EventStore,
        now: Callable[[], datetime] | None = None,
        *,
        default_limit: int = 2000,
    ) -> None:
        self._event_store = event_store
        self._now = now or datetime.utcnow
        self._default_limit = default_limit

    def collect(
        self, *, window: timedelta | None = None, limit: int | None = None
    ) -> list[RiskMetricSnapshot]:
        """Return parsed snapshots, optionally filtered to a trailing window."""
        raw_events = self._event_store.tail(
            bot_id="risk_engine", limit=limit or self._default_limit, types=["metric"]
        )
        cutoff = None
        if window is not None:
            cutoff = self._now() - window

        snapshots: list[RiskMetricSnapshot] = []
        for event in raw_events:
            point = self._coerce_snapshot(event)
            if point is None:
                continue
            if cutoff is not None and point.timestamp < cutoff:
                continue
            snapshots.append(point)
        snapshots.sort(key=lambda p: p.timestamp)
        return snapshots

    def aggregate(
        self, *, window: timedelta | None = None, limit: int | None = None
    ) -> RiskMetricsSummary:
        """Compute summary statistics for the collected snapshots."""
        snapshots = self.collect(window=window, limit=limit)
        return RiskMetricsSummary.from_points(snapshots)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            try:
                return datetime.utcfromtimestamp(float(value))
            except (OverflowError, OSError, ValueError):
                return None
        return None

    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _coerce_snapshot(self, event: dict[str, Any]) -> RiskMetricSnapshot | None:
        if event.get("bot_id") != "risk_engine" or event.get("type") != "metric":
            return None

        timestamp = self._parse_timestamp(event.get("timestamp") or event.get("time"))
        if timestamp is None:
            return None

        required_keys = (
            "equity",
            "total_notional",
            "exposure_pct",
            "max_leverage",
            "daily_pnl",
            "daily_pnl_pct",
        )
        values: dict[str, Decimal] = {}
        for key in required_keys:
            coerced = self._to_decimal(event.get(key))
            if coerced is None:
                return None
            values[key] = coerced

        reduce_only = self._to_bool(event.get("reduce_only"))
        kill_switch = self._to_bool(event.get("kill_switch"))

        return RiskMetricSnapshot(
            timestamp=timestamp,
            equity=values["equity"],
            total_notional=values["total_notional"],
            exposure_pct=values["exposure_pct"],
            max_leverage=values["max_leverage"],
            daily_pnl=values["daily_pnl"],
            daily_pnl_pct=values["daily_pnl_pct"],
            reduce_only=reduce_only,
            kill_switch=kill_switch,
        )
