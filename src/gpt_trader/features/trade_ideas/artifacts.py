"""Machine-readable trade-idea evidence artifacts.

These serializers are read-only: they shape already-persisted report, audit,
and closeout data into deterministic JSON/CSV contracts for evidence workflows.
They do not mutate trade-idea records, audit logs, or closeout attribution.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from gpt_trader.features.trade_ideas.audit import AuditEvent
from gpt_trader.features.trade_ideas.closeout import CloseoutAttribution

AUDIT_EXPORT_SCHEMA_VERSION = "gpt-trader.trade_ideas.audit_export.v1"
CLOSEOUT_EXPORT_SCHEMA_VERSION = "gpt-trader.trade_ideas.closeout_export.v1"

_AUDIT_CSV_FIELDS = (
    "event_id",
    "timestamp",
    "decision_id",
    "actor_type",
    "actor_id",
    "action",
    "before_state",
    "after_state",
    "reason",
    "record_hash",
    "evidence",
    "venue",
    "external_order_id",
)
_CLOSEOUT_CSV_FIELDS = (
    "decision_id",
    "timestamp",
    "actor_type",
    "actor_id",
    "resolution",
    "realized_profit_loss_amount",
    "realized_profit_loss_percent",
    "realized_profit_loss_unavailable_reason",
    "max_loss_amount",
    "max_loss_percent_of_account",
    "max_loss_assumptions",
    "evidence",
    "terminal_event_id",
    "terminal_event_timestamp",
    "terminal_action",
    "terminal_state",
    "record_hash",
)


def build_audit_list_payload(
    events: Sequence[AuditEvent],
    *,
    filters: Mapping[str, Any],
    total_count: int,
    limit: int | None,
    offset: int,
) -> dict[str, Any]:
    rows = [event.to_dict() for event in events]
    return {
        "schema_version": AUDIT_EXPORT_SCHEMA_VERSION,
        "filters": dict(filters),
        "pagination": _pagination_payload(
            total_count=total_count,
            returned_count=len(rows),
            limit=limit,
            offset=offset,
        ),
        "events": rows,
    }


def build_audit_export_artifact(
    events: Sequence[AuditEvent],
    *,
    filters: Mapping[str, Any],
    total_count: int,
    limit: int | None,
    offset: int,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    rows = [event.to_dict() for event in events]
    basis = {
        "schema_version": AUDIT_EXPORT_SCHEMA_VERSION,
        "filters": dict(filters),
        "pagination": _pagination_payload(
            total_count=total_count,
            returned_count=len(rows),
            limit=limit,
            offset=offset,
        ),
        "rows": rows,
    }
    return {
        "schema_version": AUDIT_EXPORT_SCHEMA_VERSION,
        "audit_export_id": _stable_artifact_id("tiaudit", basis),
        "artifact_type": "trade_idea_audit_export",
        "generated_at": _generated_at(generated_at),
        "filters": dict(filters),
        "pagination": basis["pagination"],
        "row_count": len(rows),
        "rows": rows,
    }


def build_closeout_list_payload(
    records: Sequence[CloseoutAttribution],
    *,
    terminal_events_by_id: Mapping[str, AuditEvent],
    filters: Mapping[str, Any],
    total_count: int,
    limit: int | None,
    offset: int,
) -> dict[str, Any]:
    rows = [_closeout_row(record, terminal_events_by_id) for record in records]
    return {
        "schema_version": CLOSEOUT_EXPORT_SCHEMA_VERSION,
        "filters": dict(filters),
        "pagination": _pagination_payload(
            total_count=total_count,
            returned_count=len(rows),
            limit=limit,
            offset=offset,
        ),
        "closeouts": rows,
    }


def build_closeout_export_artifact(
    records: Sequence[CloseoutAttribution],
    *,
    terminal_events_by_id: Mapping[str, AuditEvent],
    filters: Mapping[str, Any],
    total_count: int,
    limit: int | None,
    offset: int,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    rows = [_closeout_row(record, terminal_events_by_id) for record in records]
    basis = {
        "schema_version": CLOSEOUT_EXPORT_SCHEMA_VERSION,
        "filters": dict(filters),
        "pagination": _pagination_payload(
            total_count=total_count,
            returned_count=len(rows),
            limit=limit,
            offset=offset,
        ),
        "rows": rows,
    }
    return {
        "schema_version": CLOSEOUT_EXPORT_SCHEMA_VERSION,
        "closeout_export_id": _stable_artifact_id("ticloseout", basis),
        "artifact_type": "trade_idea_closeout_export",
        "generated_at": _generated_at(generated_at),
        "filters": dict(filters),
        "pagination": basis["pagination"],
        "row_count": len(rows),
        "rows": rows,
    }


def trade_idea_report_to_csv(report: Mapping[str, Any]) -> str:
    rows: list[dict[str, str]] = []
    for metric_path, value in _flatten_report(report):
        rows.append(
            {
                "quality_report_id": str(report.get("quality_report_id", "")),
                "schema_version": str(report.get("schema_version", "")),
                "metric_path": metric_path,
                "value": _csv_value(value),
            }
        )
    return _csv_from_rows(
        rows,
        fieldnames=("quality_report_id", "schema_version", "metric_path", "value"),
    )


def audit_events_to_csv(events: Sequence[AuditEvent]) -> str:
    rows = [_audit_csv_row(event) for event in events]
    return _csv_from_rows(rows, fieldnames=_AUDIT_CSV_FIELDS)


def closeout_records_to_csv(
    records: Sequence[CloseoutAttribution],
    *,
    terminal_events_by_id: Mapping[str, AuditEvent],
) -> str:
    rows = [
        _closeout_csv_row(record, terminal_events_by_id=terminal_events_by_id) for record in records
    ]
    return _csv_from_rows(rows, fieldnames=_CLOSEOUT_CSV_FIELDS)


def _closeout_row(
    record: CloseoutAttribution,
    terminal_events_by_id: Mapping[str, AuditEvent],
) -> dict[str, Any]:
    row = record.to_dict()
    terminal_event = terminal_events_by_id.get(record.terminal_event_id)
    row["terminal_event_timestamp"] = (
        terminal_event.timestamp.isoformat() if terminal_event is not None else None
    )
    row["terminal_action"] = terminal_event.action.value if terminal_event is not None else None
    row["terminal_state"] = terminal_event.after_state.value if terminal_event is not None else None
    return row


def _audit_csv_row(event: AuditEvent) -> dict[str, str]:
    payload = event.to_dict()
    return {field: _csv_value(payload[field]) for field in _AUDIT_CSV_FIELDS}


def _closeout_csv_row(
    record: CloseoutAttribution,
    *,
    terminal_events_by_id: Mapping[str, AuditEvent],
) -> dict[str, str]:
    row = _closeout_row(record, terminal_events_by_id)
    max_loss = row["max_loss"]
    return {
        "decision_id": _csv_value(row["decision_id"]),
        "timestamp": _csv_value(row["timestamp"]),
        "actor_type": _csv_value(row["actor_type"]),
        "actor_id": _csv_value(row["actor_id"]),
        "resolution": _csv_value(row["resolution"]),
        "realized_profit_loss_amount": _csv_value(row["realized_profit_loss_amount"]),
        "realized_profit_loss_percent": _csv_value(row["realized_profit_loss_percent"]),
        "realized_profit_loss_unavailable_reason": _csv_value(
            row["realized_profit_loss_unavailable_reason"]
        ),
        "max_loss_amount": _csv_value(max_loss["amount"]),
        "max_loss_percent_of_account": _csv_value(max_loss["percent_of_account"]),
        "max_loss_assumptions": _csv_value(max_loss["assumptions"]),
        "evidence": _csv_value(row["evidence"]),
        "terminal_event_id": _csv_value(row["terminal_event_id"]),
        "terminal_event_timestamp": _csv_value(row["terminal_event_timestamp"]),
        "terminal_action": _csv_value(row["terminal_action"]),
        "terminal_state": _csv_value(row["terminal_state"]),
        "record_hash": _csv_value(row["record_hash"]),
    }


def _flatten_report(payload: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []
    for key in sorted(payload):
        value = payload[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.extend(_flatten_report(value, path))
        else:
            flattened.append((path, value))
    return flattened


def _pagination_payload(
    *,
    total_count: int,
    returned_count: int,
    limit: int | None,
    offset: int,
) -> dict[str, int | None]:
    next_offset = None
    if limit is not None and offset + returned_count < total_count:
        next_offset = offset + returned_count
    return {
        "total_count": total_count,
        "returned_count": returned_count,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
    }


def _stable_artifact_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _generated_at(generated_at: datetime | None) -> str:
    current_time = generated_at or datetime.now(UTC)
    return current_time.isoformat()


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _csv_from_rows(rows: Sequence[Mapping[str, str]], *, fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()
