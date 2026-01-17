"""Daily report generator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from gpt_trader.config.path_registry import RUNTIME_DATA_DIR

from .analytics import (
    calculate_health_metrics,
    calculate_pnl_metrics,
    calculate_risk_metrics,
    calculate_symbol_metrics,
    calculate_trade_metrics,
)
from .loaders import (
    load_events_since,
    load_liveness_snapshot,
    load_metrics,
    load_runtime_fingerprint,
    load_unfilled_orders_count,
)
from .logging_utils import logger  # naming: allow
from .models import DailyReport


class DailyReportGenerator:
    """Generates daily performance reports from event store and metrics."""

    def __init__(self, profile: str = "demo", data_dir: Path | None = None) -> None:
        self.profile = profile
        if data_dir is None:
            data_dir = RUNTIME_DATA_DIR / profile
        self.data_dir = Path(data_dir)
        self.events_file = self.data_dir / "events.jsonl"
        self.metrics_file = self.data_dir / "metrics.json"
        self.orders_file = self.data_dir / "orders.db"

    def generate(self, date: datetime | None = None, lookback_hours: int = 24) -> DailyReport:
        if date is None:
            date = datetime.now(UTC)
        elif date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        else:
            date = date.astimezone(UTC)

        logger.info(f"Generating daily report for {date.date()} (profile={self.profile})")

        current_metrics = load_metrics(self.metrics_file)
        cutoff = date - timedelta(hours=lookback_hours)
        events = load_events_since(self.events_file, cutoff)

        pnl_metrics = calculate_pnl_metrics(events, current_metrics)
        trade_metrics = calculate_trade_metrics(events)
        symbol_metrics = calculate_symbol_metrics(events)
        risk_metrics = calculate_risk_metrics(events)
        health_metrics = calculate_health_metrics(events)
        orders_unfilled = load_unfilled_orders_count(self.orders_file, as_of=date)
        if orders_unfilled:
            current_unfilled = int(health_metrics.get("unfilled_orders", 0) or 0)
            if orders_unfilled > current_unfilled:
                health_metrics["unfilled_orders"] = orders_unfilled

        liveness_snapshot = None
        runtime_fingerprint = None
        events_db = self.events_file.with_name("events.db")
        if events_db.exists():
            liveness_snapshot = load_liveness_snapshot(events_db, now=date)
            runtime_fingerprint = load_runtime_fingerprint(events_db)

        return DailyReport(
            date=date.strftime("%Y-%m-%d"),
            profile=self.profile,
            generated_at=datetime.now(UTC).isoformat(),
            symbol_performance=symbol_metrics,
            liveness=liveness_snapshot,
            runtime=runtime_fingerprint,
            **pnl_metrics,  # type: ignore[arg-type]
            **trade_metrics,  # type: ignore[arg-type]
            **risk_metrics,
            **health_metrics,  # type: ignore[arg-type]
        )

    def save_report(self, report: DailyReport, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            output_dir = self.data_dir / "reports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"daily_report_{report.date}.json"
        with open(json_path, "w") as f:
            import json

            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved JSON report to {json_path}")

        text_path = output_dir / f"daily_report_{report.date}.txt"
        with open(text_path, "w") as f:
            f.write(report.to_text())
        logger.info(f"Saved text report to {text_path}")

        return text_path


__all__ = ["DailyReportGenerator"]
