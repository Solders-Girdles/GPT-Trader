from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from ..persistence.event_store import EventStore


@dataclass
class MetricsCalculator:
    """Compute performance metrics from EventStore snapshots.

    Relies on risk_engine metrics that include 'equity'.
    """

    event_store: EventStore

    @staticmethod
    def _coerce_decimal(value: Any) -> Decimal | None:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

    def _get_equity_series(self, days: int = 30) -> list[tuple[datetime, Decimal]]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        events = self.event_store.tail(
            bot_id="risk_engine", limit=100000, types=["metric"]
        )  # best effort
        series: list[tuple[datetime, Decimal]] = []
        for evt in events:
            timestamp_raw = evt.get("timestamp") or evt.get("time")
            ts: datetime | None = None
            if isinstance(timestamp_raw, str):
                try:
                    ts = datetime.fromisoformat(timestamp_raw)
                except ValueError:
                    continue
            elif isinstance(timestamp_raw, (int, float)):
                ts = datetime.fromtimestamp(timestamp_raw)
            else:
                continue
            if ts is None:
                continue
            if ts < cutoff:
                continue

            equity = self._coerce_decimal(evt.get("equity"))
            if equity is None:
                continue
            series.append((ts, equity))
        series.sort(key=lambda x: x[0])
        return series

    def get_equity_curve(self, days: int = 30) -> list[tuple[datetime, Decimal]]:
        return self._get_equity_series(days)

    def _daily_returns(self, series: list[tuple[datetime, Decimal]]) -> list[Decimal]:
        if len(series) < 2:
            return []
        # Aggregate by date to end-of-day equity
        by_day: dict[date, Decimal] = {}
        for ts, eq in series:
            day = ts.date()
            by_day[day] = eq
        days_sorted = sorted(by_day.keys())
        returns: list[Decimal] = []
        for i in range(1, len(days_sorted)):
            prev = by_day[days_sorted[i - 1]]
            cur = by_day[days_sorted[i]]
            if prev > 0:
                returns.append((cur - prev) / prev)
        return returns

    def calculate_sharpe(self, window_days: int = 30) -> Decimal:
        series = self._get_equity_series(window_days)
        rets = self._daily_returns(series)
        if len(rets) < 2:
            return Decimal("0")
        mean = sum(rets) / Decimal(len(rets))
        # sample std dev
        var = sum((r - mean) * (r - mean) for r in rets) / Decimal(len(rets) - 1)
        if var <= 0:
            return Decimal("0")
        std = var.sqrt()
        # Annualize (365 days for crypto)
        return mean * Decimal(365).sqrt() / std

    def calculate_max_drawdown(self, window_days: int = 90) -> tuple[Decimal, datetime, datetime]:
        curve = self._get_equity_series(window_days)
        if not curve:
            now = datetime.utcnow()
            return Decimal("0"), now, now
        peak = curve[0][1]
        peak_date = curve[0][0]
        max_dd = Decimal("0")
        trough_date = peak_date
        for ts, eq in curve:
            if eq > peak:
                peak = eq
                peak_date = ts
            if peak > 0:
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
                    trough_date = ts
        return max_dd, peak_date, trough_date
