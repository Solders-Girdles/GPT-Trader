"""Trade-idea CLI commands.

This module is an operator-facing adapter over ``TradeIdeaService``. It never
submits, modifies, or cancels broker orders; submission and fill commands only
append audit records for tickets executed elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from gpt_trader.cli import options
from gpt_trader.cli.response import CliError, CliErrorCode, CliResponse
from gpt_trader.core import Candle
from gpt_trader.errors import ValidationError
from gpt_trader.features.trade_ideas import (
    ActorType,
    AuditIntegrityError,
    BaselineProposer,
    BaselineProposerConfig,
    BudgetIntegrityError,
    CloseoutResolution,
    InvalidTransitionError,
    PolicyViolationError,
    ReplayReport,
    ReplayRunnerConfig,
    TradeIdea,
    TradeIdeaReplayRunner,
    TradeIdeaService,
    TradeIdeaState,
    UnknownTradeIdeaError,
    create_trade_idea_service,
    is_safe_decision_id,
    resolve_trade_idea_actor_id,
)
from gpt_trader.features.trade_ideas.report import (
    build_trade_idea_track_record_report,
    format_trade_idea_track_record_report,
)

VENUE_CHOICES = ("coinbase", "manual")
BUDGET_FIELDS = (
    "max_loss_per_idea_pct",
    "max_daily_loss_pct",
    "max_open_notional_pct",
    "max_concurrent_approved_tickets",
    "max_review_latency_hours",
    "sizing_capped_by_budget",
    "gain_retention_floor_pct",
    "allow_futures_leverage",
    "allow_naked_shorts",
)


class IdeaInputError(ValueError):
    """Raised when a JSON trade-idea payload cannot be parsed."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class CandleInputError(ValueError):
    """Raised when a replay candle fixture cannot be parsed."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def register(subparsers: Any) -> None:
    """Register the ideas command group."""
    parser: ArgumentParser = subparsers.add_parser(
        "ideas",
        help="Review and audit trade ideas without broker execution",
        description=(
            "Thin CLI over the trade-idea approval workflow. "
            "No command places, modifies, or cancels broker orders."
        ),
    )
    ideas_subparsers = parser.add_subparsers(dest="ideas_command", required=True)

    propose = ideas_subparsers.add_parser("propose", help="Propose a new trade idea")
    _add_common_options(propose)
    _add_input_options(propose)
    _add_actor_options(propose)
    propose.add_argument(
        "--actor-type",
        choices=(ActorType.AI.value, ActorType.HUMAN.value),
        default=ActorType.AI.value,
        help="Actor type stamped into the proposed event",
    )
    propose.add_argument("--reason", default="New trade idea proposed", help="Audit reason")
    propose.set_defaults(handler=_handle_propose, subcommand="propose")

    resubmit = ideas_subparsers.add_parser(
        "resubmit", help="Submit a revised record after requested changes"
    )
    _add_common_options(resubmit)
    _add_input_options(resubmit)
    _add_actor_options(resubmit)
    resubmit.add_argument(
        "--actor-type",
        choices=(ActorType.AI.value, ActorType.HUMAN.value),
        default=ActorType.AI.value,
        help="Actor type stamped into the resubmitted event",
    )
    resubmit.add_argument(
        "--reason", default="Revised after requested changes", help="Audit reason"
    )
    resubmit.set_defaults(handler=_handle_resubmit, subcommand="resubmit")

    list_parser = ideas_subparsers.add_parser("list", help="List stored trade ideas")
    _add_common_options(list_parser)
    list_parser.add_argument(
        "--state",
        choices=[state.value for state in TradeIdeaState],
        help="Filter by workflow state",
    )
    list_parser.set_defaults(handler=_handle_list, subcommand="list")

    show = ideas_subparsers.add_parser("show", help="Show one trade idea")
    _add_common_options(show)
    show.add_argument("decision_id", help="Trade idea decision identifier")
    show.add_argument("--events", action="store_true", help="Include audit history")
    show.set_defaults(handler=_handle_show, subcommand="show")

    report = ideas_subparsers.add_parser(
        "report",
        help="Summarize proposal quality, workflow rates, and closeout coverage",
        description=(
            "Read-only track-record report over local trade-idea records, audit events, "
            "and closeout attribution. This command never contacts a broker or account."
        ),
    )
    _add_common_options(report)
    report.set_defaults(handler=_handle_report, subcommand="report")

    replay = ideas_subparsers.add_parser(
        "replay",
        help="Replay proposer calibration against local candle fixtures",
        description=(
            "Read-only calibration commands over local candle fixtures. "
            "Replay commands never contact brokers, accounts, or live market data."
        ),
    )
    replay_subparsers = replay.add_subparsers(dest="replay_command", required=True)
    baseline = replay_subparsers.add_parser(
        "baseline",
        help="Replay the deterministic baseline proposer over candle history",
        description=(
            "Feed a local OHLCV candle fixture to the deterministic baseline proposer "
            "and summarize replay scoring. This command is broker-free and read-only."
        ),
    )
    options.add_output_options(baseline, include_quiet=False)
    baseline.add_argument(
        "--file",
        type=Path,
        required=True,
        help="JSON fixture with a top-level candles array of OHLCV bars",
    )
    baseline.add_argument("--symbol", required=True, help="Symbol represented by the candles")
    baseline.add_argument(
        "--granularity",
        required=True,
        help="Candle granularity, for example ONE_HOUR, 1H, or 1D",
    )
    baseline.add_argument(
        "--source",
        default="fixture:candles",
        help="Source label stamped into point-in-time replay snapshots",
    )
    baseline.add_argument(
        "--min-history",
        type=_positive_int_value,
        help=(
            "Minimum historical candles before evaluating snapshots "
            "(default: max(short-window, long-window) + crossover-lookback)"
        ),
    )
    baseline.add_argument("--short-window", type=_positive_int_value, default=10)
    baseline.add_argument("--long-window", type=_positive_int_value, default=50)
    baseline.add_argument("--crossover-lookback", type=_positive_int_value, default=3)
    baseline.add_argument(
        "--risk-per-idea-pct",
        type=_non_negative_decimal_value,
        default=Decimal("2"),
    )
    baseline.add_argument(
        "--entry-band-pct",
        type=_positive_decimal_value,
        default=Decimal("1"),
    )
    baseline.add_argument(
        "--reward-multiple",
        type=_positive_decimal_value,
        default=Decimal("2"),
    )
    baseline.add_argument("--expiry-hours", type=_positive_int_value, default=48)
    baseline.add_argument("--expected-hold", default="5-15 days")
    baseline.add_argument(
        "--price-precision",
        type=_positive_decimal_value,
        default=Decimal("0.01"),
    )
    baseline.set_defaults(handler=_handle_replay_baseline, subcommand="replay baseline")

    closeout = ideas_subparsers.add_parser(
        "closeout",
        help="Record or inspect terminal closeout attribution",
        description=(
            "Record and inspect broker-neutral closeout attribution for terminal trade "
            "ideas. This command never contacts a broker or account."
        ),
    )
    closeout_subparsers = closeout.add_subparsers(dest="closeout_command", required=True)

    closeout_record = closeout_subparsers.add_parser(
        "record",
        help="Record terminal closeout attribution; does not call a broker API",
        description=(
            "Append closeout attribution for a terminal trade idea. This records external "
            "evidence only and never places, modifies, or cancels broker orders."
        ),
    )
    _add_common_options(closeout_record)
    _add_actor_options(closeout_record)
    closeout_record.add_argument("decision_id", help="Trade idea decision identifier")
    closeout_record.add_argument(
        "--resolution",
        required=True,
        choices=[resolution.value for resolution in CloseoutResolution],
        help="Why the terminal idea resolved",
    )
    closeout_record.add_argument(
        "--realized-profit-loss-amount",
        type=_decimal_value,
        help="Realized profit/loss amount; negative values represent losses",
    )
    closeout_record.add_argument(
        "--realized-profit-loss-percent",
        type=_decimal_value,
        help="Realized profit/loss percent; negative values represent losses",
    )
    closeout_record.add_argument(
        "--realized-profit-loss-unavailable-reason",
        default="",
        help="Why realized profit/loss is unavailable",
    )
    closeout_record.add_argument(
        "--evidence",
        action="append",
        default=None,
        help="Evidence string to attach; repeat for multiple evidence entries",
    )
    closeout_record.add_argument(
        "--actor-type",
        choices=(ActorType.HUMAN.value, ActorType.SYSTEM.value),
        default=ActorType.HUMAN.value,
        help="Actor type stamped into the closeout attribution record",
    )
    closeout_record.set_defaults(handler=_handle_closeout_record, subcommand="closeout record")

    closeout_show = closeout_subparsers.add_parser(
        "show",
        help="Show closeout attribution for one trade idea",
        description="Read closeout attribution from local trade-idea records only.",
    )
    _add_common_options(closeout_show)
    closeout_show.add_argument("decision_id", help="Trade idea decision identifier")
    closeout_show.set_defaults(handler=_handle_closeout_show, subcommand="closeout show")

    approve = ideas_subparsers.add_parser("approve", help="Approve a proposed trade idea")
    _add_common_options(approve)
    _add_actor_options(approve)
    approve.add_argument("decision_id", help="Trade idea decision identifier")
    approve.add_argument("--reason", required=True, help="Human approval reason")
    approve.set_defaults(handler=_handle_approve, subcommand="approve")

    reject = ideas_subparsers.add_parser("reject", help="Reject a proposed trade idea")
    _add_common_options(reject)
    _add_actor_options(reject)
    reject.add_argument("decision_id", help="Trade idea decision identifier")
    reject.add_argument("--reason", required=True, help="Human rejection reason")
    reject.set_defaults(handler=_handle_reject, subcommand="reject")

    request_changes = ideas_subparsers.add_parser(
        "request-changes", help="Request changes to a proposed trade idea"
    )
    _add_common_options(request_changes)
    _add_actor_options(request_changes)
    request_changes.add_argument("decision_id", help="Trade idea decision identifier")
    request_changes.add_argument("--reason", required=True, help="Requested change reason")
    request_changes.set_defaults(handler=_handle_request_changes, subcommand="request-changes")

    cancel = ideas_subparsers.add_parser("cancel", help="Cancel an approved or submitted idea")
    _add_common_options(cancel)
    _add_actor_options(cancel)
    cancel.add_argument("decision_id", help="Trade idea decision identifier")
    cancel.add_argument("--reason", required=True, help="Human cancellation reason")
    cancel.set_defaults(handler=_handle_cancel, subcommand="cancel")

    expire = ideas_subparsers.add_parser("expire", help="Expire one idea or sweep stale ideas")
    _add_common_options(expire)
    _add_actor_options(expire)
    expire.add_argument("decision_id", nargs="?", help="Trade idea decision identifier")
    expire.add_argument("--sweep", action="store_true", help="Expire every stale non-terminal idea")
    expire.add_argument(
        "--reason",
        default="Idea passed its review or execution deadline",
        help="Expiry audit reason",
    )
    expire.set_defaults(handler=_handle_expire, subcommand="expire")

    mark_submitted = ideas_subparsers.add_parser(
        "mark-submitted",
        help="Record a manually submitted approved ticket; does not call a broker API",
        description=(
            "Append an audit event for a ticket submitted outside this CLI. "
            "This command never places, modifies, or cancels broker orders."
        ),
    )
    _add_common_options(mark_submitted)
    _add_actor_options(mark_submitted)
    mark_submitted.add_argument("decision_id", help="Trade idea decision identifier")
    mark_submitted.add_argument("--venue", required=True, choices=VENUE_CHOICES)
    mark_submitted.add_argument("--external-order-id", default="", help="External order id")
    mark_submitted.add_argument("--reason", default="Approved ticket submitted")
    mark_submitted.add_argument(
        "--actor-type",
        choices=(ActorType.SYSTEM.value, ActorType.HUMAN.value),
        default=ActorType.SYSTEM.value,
        help="Actor type stamped into the submitted event",
    )
    mark_submitted.set_defaults(handler=_handle_mark_submitted, subcommand="mark-submitted")

    mark_filled = ideas_subparsers.add_parser(
        "mark-filled",
        help="Record a manually observed fill; does not call a broker API",
        description=(
            "Append an audit event for a fill observed outside this CLI. "
            "This command never contacts a broker or account."
        ),
    )
    _add_common_options(mark_filled)
    _add_actor_options(mark_filled)
    mark_filled.add_argument("decision_id", help="Trade idea decision identifier")
    mark_filled.add_argument("--venue", required=True, choices=VENUE_CHOICES)
    mark_filled.add_argument("--external-order-id", default="", help="External order id")
    mark_filled.add_argument("--reason", default="Venue confirmed fill")
    mark_filled.set_defaults(handler=_handle_mark_filled, subcommand="mark-filled")

    budget = ideas_subparsers.add_parser("budget", help="Inspect or update risk budget")
    budget_subparsers = budget.add_subparsers(dest="budget_command", required=True)

    budget_show = budget_subparsers.add_parser("show", help="Show current risk budget")
    _add_common_options(budget_show)
    budget_show.set_defaults(handler=_handle_budget_show, subcommand="budget show")

    budget_set = budget_subparsers.add_parser("set", help="Set a new risk budget version")
    _add_common_options(budget_set)
    _add_actor_options(budget_set)
    budget_set.add_argument("--reason", required=True, help="Reason for this budget version")
    budget_set.add_argument("--max-loss-per-idea-pct", type=_non_negative_decimal_value)
    budget_set.add_argument("--max-daily-loss-pct", type=_non_negative_decimal_value)
    budget_set.add_argument("--max-open-notional-pct", type=_non_negative_decimal_value)
    budget_set.add_argument("--max-concurrent-approved-tickets", type=_non_negative_int_value)
    budget_set.add_argument("--max-review-latency-hours", type=_non_negative_int_value)
    budget_set.add_argument(
        "--sizing-capped-by-budget",
        choices=("true", "false"),
        help="Whether sizing is capped by budget",
    )
    budget_set.add_argument("--gain-retention-floor-pct", type=_non_negative_decimal_value)
    budget_set.add_argument(
        "--allow-futures-leverage",
        choices=("true", "false"),
        help="Whether futures leverage is permitted",
    )
    budget_set.add_argument(
        "--allow-naked-shorts",
        choices=("true", "false"),
        help="Whether naked shorts are permitted",
    )
    budget_set.set_defaults(handler=_handle_budget_set, subcommand="budget set")

    audit = ideas_subparsers.add_parser("audit", help="Read and verify the audit log")
    audit_subparsers = audit.add_subparsers(dest="audit_command", required=True)

    audit_tail = audit_subparsers.add_parser("tail", help="Show recent audit events")
    _add_common_options(audit_tail)
    audit_tail.add_argument("-n", "--count", type=int, default=20)
    audit_tail.add_argument("--decision-id", help="Filter events by decision id")
    audit_tail.set_defaults(handler=_handle_audit_tail, subcommand="audit tail")

    audit_verify = audit_subparsers.add_parser("verify", help="Verify audit log integrity")
    _add_common_options(audit_verify)
    audit_verify.set_defaults(handler=_handle_audit_verify, subcommand="audit verify")


def _add_common_options(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--ideas-root",
        type=Path,
        help="Trade-idea storage root (default: GPT_TRADER_IDEAS_ROOT, then var/data/trade_ideas)",
    )
    options.add_output_options(parser, include_quiet=False)


def _add_input_options(parser: ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Read TradeIdea JSON from file")
    source.add_argument("--stdin", action="store_true", help="Read TradeIdea JSON from stdin")


def _add_actor_options(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--actor",
        help="Actor id stamped into audit events (default: GPT_TRADER_ACTOR, then OS user)",
    )


def _decimal_value(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value}") from error
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError(f"decimal value must be finite: {value}")
    return parsed


def _non_negative_decimal_value(value: str) -> Decimal:
    parsed = _decimal_value(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"decimal value must be non-negative: {value}")
    return parsed


def _non_negative_int_value(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer value: {value}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"integer value must be non-negative: {value}")
    return parsed


def _positive_decimal_value(value: str) -> Decimal:
    parsed = _decimal_value(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"decimal value must be positive: {value}")
    return parsed


def _positive_int_value(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer value: {value}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"integer value must be positive: {value}")
    return parsed


def _bool_value(value: str) -> bool:
    return value.lower() == "true"


def _service(args: Namespace) -> TradeIdeaService:
    return create_trade_idea_service(getattr(args, "ideas_root", None))


def _actor_id(args: Namespace) -> str:
    return resolve_trade_idea_actor_id(getattr(args, "actor", None))


def _output_format(args: Namespace) -> str:
    return getattr(args, "output_format", "text")


def _load_trade_idea(args: Namespace) -> TradeIdea:
    try:
        if getattr(args, "file", None):
            raw_payload = args.file.read_text(encoding="utf-8")
        else:
            raw_payload = sys.stdin.read()
    except OSError as error:
        raise IdeaInputError(f"Could not read trade idea input: {error}") from error

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as error:
        raise IdeaInputError(f"Malformed trade idea JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise IdeaInputError("Trade idea input must be a JSON object")

    try:
        return TradeIdea.from_dict(payload)
    except KeyError as error:
        field = str(error).strip("'")
        raise IdeaInputError(f"Missing required trade idea field: {field}", field=field) from error
    except (InvalidOperation, TypeError, ValueError) as error:
        raise IdeaInputError(f"Invalid trade idea field: {error}") from error


def _load_candle_fixture(path: Path) -> tuple[Candle, ...]:
    try:
        raw_payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CandleInputError(f"Could not read candle fixture: {error}") from error

    try:
        payload = json.loads(raw_payload, parse_float=Decimal, parse_int=Decimal)
    except json.JSONDecodeError as error:
        raise CandleInputError(f"Malformed candle fixture JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise CandleInputError("Candle fixture must be a JSON object", field="candles")

    raw_candles = payload.get("candles")
    if not isinstance(raw_candles, list):
        raise CandleInputError("Candle fixture must contain a candles array", field="candles")

    return tuple(_parse_candle(row, index) for index, row in enumerate(raw_candles))


def _parse_candle(row: Any, index: int) -> Candle:
    if not isinstance(row, dict):
        raise CandleInputError(
            f"candle at index {index} must be a JSON object",
            field=f"candles[{index}]",
        )
    candle = Candle(
        ts=_parse_candle_timestamp(_required_candle_field(row, "ts", index), index),
        open=_parse_candle_decimal(_required_candle_field(row, "open", index), index, "open"),
        high=_parse_candle_decimal(_required_candle_field(row, "high", index), index, "high"),
        low=_parse_candle_decimal(_required_candle_field(row, "low", index), index, "low"),
        close=_parse_candle_decimal(_required_candle_field(row, "close", index), index, "close"),
        volume=_parse_candle_decimal(_required_candle_field(row, "volume", index), index, "volume"),
    )
    _validate_candle_semantics(candle, index)
    return candle


def _required_candle_field(row: dict[str, Any], field_name: str, index: int) -> Any:
    if field_name not in row:
        raise CandleInputError(
            f"candle at index {index} is missing required field '{field_name}'",
            field=f"candles[{index}].{field_name}",
        )
    return row[field_name]


def _parse_candle_timestamp(value: Any, index: int) -> datetime:
    field = f"candles[{index}].ts"
    if not isinstance(value, str):
        raise CandleInputError("candle ts must be an ISO-8601 string", field=field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CandleInputError(f"Invalid candle timestamp: {value}", field=field) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_candle_decimal(value: Any, index: int, field_name: str) -> Decimal:
    field = f"candles[{index}].{field_name}"
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise CandleInputError(
            f"Invalid decimal candle value for {field_name}: {value}",
            field=field,
        ) from error
    if not parsed.is_finite():
        raise CandleInputError(
            f"Candle decimal value must be finite for {field_name}: {value}",
            field=field,
        )
    return parsed


def _validate_candle_semantics(candle: Candle, index: int) -> None:
    for field_name in ("open", "high", "low", "close"):
        if getattr(candle, field_name) <= 0:
            raise CandleInputError(
                f"candle at index {index} has non-positive {field_name}",
                field=f"candles[{index}].{field_name}",
            )
    if candle.high < candle.low:
        raise CandleInputError(
            f"candle at index {index} has high below low",
            field=f"candles[{index}].high",
        )
    if not candle.low <= candle.open <= candle.high:
        raise CandleInputError(
            f"candle at index {index} has open outside low/high range",
            field=f"candles[{index}].open",
        )
    if not candle.low <= candle.close <= candle.high:
        raise CandleInputError(
            f"candle at index {index} has close outside low/high range",
            field=f"candles[{index}].close",
        )
    if candle.volume < 0:
        raise CandleInputError(
            f"candle at index {index} has negative volume",
            field=f"candles[{index}].volume",
        )


def _default_replay_min_history(config: BaselineProposerConfig) -> int:
    return max(config.short_window, config.long_window) + config.crossover_lookback


def _handle_propose(args: Namespace) -> CliResponse:
    command = "ideas propose"
    try:
        idea = _load_trade_idea(args)
        service = _service(args)
        service.validate_new_proposal(idea)
        violations = service.approval_violations(idea, actor_type=ActorType.HUMAN)
        view = service.propose(
            idea,
            actor_id=_actor_id(args),
            actor_type=ActorType(args.actor_type),
            reason=args.reason,
        )
    except IdeaInputError as error:
        return _input_error(command, args, error)
    except Exception as error:
        return _mapped_error(command, args, error)

    payload = {
        **_view_summary(view),
        "record_hash": view.idea.record_hash(),
        "violations": violations,
    }
    warning_messages = [f"would fail approval: {violation}" for violation in violations]
    text = _status_line(command, "OK", f"{view.idea.decision_id}, state={view.state.value}")
    if violations:
        text += "\n" + "\n".join(f"⚠ would fail approval: {violation}" for violation in violations)
    return _success(command, args, payload, text, warnings=warning_messages)


def _handle_resubmit(args: Namespace) -> CliResponse:
    command = "ideas resubmit"
    try:
        idea = _load_trade_idea(args)
        service = _service(args)
        service.validate_resubmission(idea)
        violations = service.approval_violations(idea, actor_type=ActorType.HUMAN)
        view = service.resubmit(
            idea,
            actor_id=_actor_id(args),
            actor_type=ActorType(args.actor_type),
            reason=args.reason,
        )
    except IdeaInputError as error:
        return _input_error(command, args, error)
    except Exception as error:
        return _mapped_error(command, args, error)

    payload = {
        **_view_summary(view),
        "record_hash": view.idea.record_hash(),
        "violations": violations,
    }
    warning_messages = [f"would fail approval: {violation}" for violation in violations]
    text = _status_line(command, "OK", f"{view.idea.decision_id}, state={view.state.value}")
    if violations:
        text += "\n" + "\n".join(f"⚠ would fail approval: {violation}" for violation in violations)
    return _success(command, args, payload, text, warnings=warning_messages)


def _handle_list(args: Namespace) -> CliResponse:
    command = "ideas list"
    try:
        requested_state = TradeIdeaState(args.state) if args.state else None
        views = _service(args).list_views(requested_state)
    except Exception as error:
        return _mapped_error(command, args, error)

    ideas = [_view_summary(view) for view in views]
    text = _ideas_table(ideas)
    return _success(command, args, {"ideas": ideas}, text, was_noop=not ideas)


def _handle_show(args: Namespace) -> CliResponse:
    command = "ideas show"
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    try:
        view = _service(args).get(args.decision_id)
    except Exception as error:
        return _mapped_error(command, args, error)

    payload = _view_record(view, include_events=args.events)
    text = _record_text(payload, include_events=args.events)
    return _success(command, args, payload, text)


def _handle_report(args: Namespace) -> CliResponse:
    command = "ideas report"
    try:
        payload = build_trade_idea_track_record_report(_service(args))
    except Exception as error:
        return _mapped_error(command, args, error)

    text = format_trade_idea_track_record_report(payload)
    idea_count = payload["proposal_volume"]["idea_count"]
    return _success(command, args, payload, text, was_noop=idea_count == 0)


def _handle_replay_baseline(args: Namespace) -> CliResponse:
    command = "ideas replay baseline"
    try:
        candles = _load_candle_fixture(args.file)
        proposer_config = BaselineProposerConfig(
            short_window=args.short_window,
            long_window=args.long_window,
            crossover_lookback=args.crossover_lookback,
            risk_per_idea_pct=args.risk_per_idea_pct,
            entry_band_pct=args.entry_band_pct,
            reward_multiple=args.reward_multiple,
            expiry_hours=args.expiry_hours,
            expected_hold=args.expected_hold,
            price_precision=args.price_precision,
        )
        min_history = (
            args.min_history
            if args.min_history is not None
            else _default_replay_min_history(proposer_config)
        )
        report = TradeIdeaReplayRunner(
            BaselineProposer(proposer_config),
            config=ReplayRunnerConfig(source=args.source, min_history=min_history),
        ).run_series(symbol=args.symbol, granularity=args.granularity, candles=candles)
    except CandleInputError as error:
        return _input_error(command, args, error)
    except Exception as error:
        return _mapped_error(command, args, error)

    payload = report.to_dict()
    text = _replay_report_text(report)
    return _success(command, args, payload, text, was_noop=report.ideas_proposed == 0)


def _handle_closeout_record(args: Namespace) -> CliResponse:
    command = "ideas closeout record"
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    profit_loss_error = _closeout_profit_loss_error(command, args)
    if profit_loss_error is not None:
        return profit_loss_error
    evidence_error = _evidence_error(command, args)
    if evidence_error is not None:
        return evidence_error

    try:
        record = _service(args).record_closeout_attribution(
            args.decision_id,
            actor_id=_actor_id(args),
            actor_type=ActorType(args.actor_type),
            resolution=args.resolution,
            realized_profit_loss_amount=args.realized_profit_loss_amount,
            realized_profit_loss_percent=args.realized_profit_loss_percent,
            realized_profit_loss_unavailable_reason=(
                args.realized_profit_loss_unavailable_reason.strip()
            ),
            evidence=_evidence(args),
        )
    except Exception as error:
        return _mapped_error(command, args, error)
    return _closeout_success(command, args, record)


def _handle_closeout_show(args: Namespace) -> CliResponse:
    command = "ideas closeout show"
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    try:
        record = _service(args).get_closeout_attribution(args.decision_id)
    except Exception as error:
        return _mapped_error(command, args, error)

    payload = {
        "decision_id": args.decision_id,
        "closeout_attribution": record.to_dict() if record is not None else None,
    }
    if record is None:
        text = _status_line(command, "OK", f"{args.decision_id}, no closeout attribution")
        return _success(command, args, payload, text, was_noop=True)

    text = _closeout_text(command, record.to_dict())
    return _success(command, args, payload, text)


def _handle_approve(args: Namespace) -> CliResponse:
    command = "ideas approve"
    reason_error = _reason_error(command, args)
    if reason_error is not None:
        return reason_error
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    try:
        view = _service(args).approve(
            args.decision_id, actor_id=_actor_id(args), reason=args.reason
        )
    except Exception as error:
        return _mapped_error(command, args, error)
    return _state_change_success(command, args, view)


def _handle_reject(args: Namespace) -> CliResponse:
    command = "ideas reject"
    reason_error = _reason_error(command, args)
    if reason_error is not None:
        return reason_error
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    try:
        view = _service(args).reject(args.decision_id, actor_id=_actor_id(args), reason=args.reason)
    except Exception as error:
        return _mapped_error(command, args, error)
    return _state_change_success(command, args, view)


def _handle_request_changes(args: Namespace) -> CliResponse:
    command = "ideas request-changes"
    reason_error = _reason_error(command, args)
    if reason_error is not None:
        return reason_error
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    try:
        view = _service(args).request_changes(
            args.decision_id,
            actor_id=_actor_id(args),
            reason=args.reason,
        )
    except Exception as error:
        return _mapped_error(command, args, error)
    return _state_change_success(command, args, view)


def _handle_cancel(args: Namespace) -> CliResponse:
    command = "ideas cancel"
    reason_error = _reason_error(command, args)
    if reason_error is not None:
        return reason_error
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    try:
        view = _service(args).cancel(args.decision_id, actor_id=_actor_id(args), reason=args.reason)
    except Exception as error:
        return _mapped_error(command, args, error)
    return _state_change_success(command, args, view)


def _handle_expire(args: Namespace) -> CliResponse:
    command = "ideas expire"
    has_decision_id = bool(args.decision_id)
    if has_decision_id == bool(args.sweep):
        return _failure(
            command,
            args,
            CliErrorCode.MISSING_ARGUMENT,
            "Provide exactly one of DECISION_ID or --sweep",
        )
    if has_decision_id:
        decision_id_error = _decision_id_error(command, args, args.decision_id)
        if decision_id_error is not None:
            return decision_id_error

    try:
        service = _service(args)
        if args.sweep:
            expired_views = service.expire_due_ideas(actor_id=_actor_id(args), reason=args.reason)
            expired = [view.idea.decision_id for view in expired_views]
            text = _status_line(command, "OK", f"expired={len(expired)}")
            payload = {"expired": expired}
            return _success(command, args, payload, text, was_noop=not expired)

        view = service.expire(args.decision_id, actor_id=_actor_id(args), reason=args.reason)
    except Exception as error:
        return _mapped_error(command, args, error)
    return _state_change_success(command, args, view)


def _handle_mark_submitted(args: Namespace) -> CliResponse:
    command = "ideas mark-submitted"
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    try:
        view = _service(args).record_submission(
            args.decision_id,
            actor_id=_actor_id(args),
            venue=args.venue,
            external_order_id=args.external_order_id,
            reason=args.reason,
            actor_type=ActorType(args.actor_type),
        )
    except Exception as error:
        return _mapped_error(command, args, error)
    return _state_change_success(command, args, view)


def _handle_mark_filled(args: Namespace) -> CliResponse:
    command = "ideas mark-filled"
    decision_id_error = _decision_id_error(command, args, args.decision_id)
    if decision_id_error is not None:
        return decision_id_error
    try:
        view = _service(args).record_fill(
            args.decision_id,
            actor_id=_actor_id(args),
            venue=args.venue,
            external_order_id=args.external_order_id,
            reason=args.reason,
            actor_type=ActorType.VENUE,
        )
    except Exception as error:
        return _mapped_error(command, args, error)
    return _state_change_success(command, args, view)


def _handle_budget_show(args: Namespace) -> CliResponse:
    command = "ideas budget show"
    try:
        budget = _service(args).current_budget()
    except Exception as error:
        return _mapped_error(command, args, error)
    payload = budget.to_dict()
    text = _budget_text(payload)
    return _success(command, args, payload, text)


def _handle_budget_set(args: Namespace) -> CliResponse:
    command = "ideas budget set"
    reason_error = _reason_error(command, args)
    if reason_error is not None:
        return reason_error
    overrides = _budget_overrides(args)
    if not overrides:
        return _failure(
            command,
            args,
            CliErrorCode.MISSING_ARGUMENT,
            "At least one budget field flag is required",
        )

    try:
        service = _service(args)
        current = service.current_budget()
        new_budget = replace(
            current,
            version=current.version + 1,
            reason=args.reason,
            **overrides,
        )
        service.update_budget(new_budget, ActorType.HUMAN, _actor_id(args))
    except Exception as error:
        return _mapped_error(command, args, error)

    payload = new_budget.to_dict()
    text = _status_line(command, "OK", f"version={new_budget.version}")
    return _success(command, args, payload, text)


def _handle_audit_tail(args: Namespace) -> CliResponse:
    command = "ideas audit tail"
    try:
        count = max(0, args.count)
        events = _service(args).audit_log.read_events(args.decision_id)
    except Exception as error:
        return _mapped_error(command, args, error)
    selected = events[-count:] if count else []
    payload = {"events": [event.to_dict() for event in selected]}
    text = _events_text(payload["events"])
    return _success(command, args, payload, text, was_noop=not selected)


def _handle_audit_verify(args: Namespace) -> CliResponse:
    command = "ideas audit verify"
    try:
        events = _service(args).audit_log.verify()
    except Exception as error:
        return _mapped_error(command, args, error)
    payload = {"event_count": len(events)}
    text = _status_line(command, "OK", f"{len(events)} events")
    return _success(command, args, payload, text, was_noop=not events)


def _budget_overrides(args: Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for field_name in BUDGET_FIELDS:
        value = getattr(args, field_name)
        if value is None:
            continue
        if field_name in {
            "sizing_capped_by_budget",
            "allow_futures_leverage",
            "allow_naked_shorts",
        }:
            value = _bool_value(value)
        overrides[field_name] = value
    return overrides


def _closeout_profit_loss_error(command: str, args: Namespace) -> CliResponse | None:
    if (
        args.realized_profit_loss_amount is not None
        or args.realized_profit_loss_percent is not None
        or args.realized_profit_loss_unavailable_reason.strip()
    ):
        return None
    return _failure(
        command,
        args,
        CliErrorCode.MISSING_ARGUMENT,
        (
            "Provide at least one of --realized-profit-loss-amount, "
            "--realized-profit-loss-percent, or --realized-profit-loss-unavailable-reason"
        ),
        details={"field": "realized_profit_loss"},
    )


def _evidence(args: Namespace) -> tuple[str, ...]:
    return tuple(item.strip() for item in (getattr(args, "evidence", None) or ()))


def _evidence_error(command: str, args: Namespace) -> CliResponse | None:
    for index, item in enumerate(getattr(args, "evidence", None) or ()):
        if item.strip():
            continue
        return _failure(
            command,
            args,
            CliErrorCode.INVALID_ARGUMENT,
            "--evidence entries must be non-empty",
            details={"field": "evidence", "index": index},
        )
    return None


def _state_change_success(command: str, args: Namespace, view: Any) -> CliResponse:
    payload = _view_summary(view)
    text = _status_line(command, "OK", f"{view.idea.decision_id}, state={view.state.value}")
    return _success(command, args, payload, text)


def _closeout_success(command: str, args: Namespace, record: Any) -> CliResponse:
    record_payload = record.to_dict()
    payload = {
        "decision_id": record.decision_id,
        "closeout_attribution": record_payload,
    }
    return _success(command, args, payload, _closeout_text(command, record_payload))


def _view_summary(view: Any) -> dict[str, Any]:
    idea = view.idea
    expires_at = idea.time_horizon.expires_at
    percent = idea.max_loss.percent_of_account
    return {
        "decision_id": idea.decision_id,
        "state": view.state.value,
        "instrument": idea.instrument,
        "direction": idea.direction.value,
        "max_loss_pct": str(percent) if percent is not None else None,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "confidence": idea.confidence.label.value,
    }


def _view_record(view: Any, *, include_events: bool) -> dict[str, Any]:
    payload: dict[str, Any] = view.idea.to_dict()
    payload["state"] = view.state.value
    if include_events:
        payload["events"] = [event.to_dict() for event in view.events]
    return payload


def _success(
    command: str,
    args: Namespace,
    payload: Any,
    text: str,
    *,
    warnings: list[str] | None = None,
    was_noop: bool = False,
) -> CliResponse:
    data = text if _output_format(args) == "text" else payload
    return CliResponse.success_response(
        command=command,
        data=data,
        warnings=warnings,
        was_noop=was_noop,
    )


def _input_error(
    command: str, args: Namespace, error: IdeaInputError | CandleInputError
) -> CliResponse:
    details = {"field": error.field} if error.field else {}
    return _failure(
        command,
        args,
        CliErrorCode.INVALID_ARGUMENT,
        str(error),
        details=details,
        data={"field": error.field} if error.field else None,
    )


def _decision_id_error(command: str, args: Namespace, decision_id: str) -> CliResponse | None:
    if is_safe_decision_id(decision_id):
        return None
    return _failure(
        command,
        args,
        CliErrorCode.INVALID_ARGUMENT,
        "decision_id must be a safe path segment",
        details={"field": "decision_id", "value": decision_id},
        data={"field": "decision_id"},
    )


def _reason_error(command: str, args: Namespace) -> CliResponse | None:
    reason = getattr(args, "reason", "")
    if reason and reason.strip():
        return None
    return _failure(
        command,
        args,
        CliErrorCode.MISSING_ARGUMENT,
        "--reason must be non-empty",
        details={"field": "reason"},
    )


def _mapped_error(command: str, args: Namespace, error: Exception) -> CliResponse:
    if isinstance(error, PolicyViolationError):
        violations = error.violations or [str(error)]
        message = f"approval refused ({len(violations)} violations)"
        return _failure(
            command,
            args,
            CliErrorCode.POLICY_VIOLATION,
            message,
            details={"violations": violations},
            data={"violations": violations},
            text_lines=[
                f"✗ {command} FAILED: {message}",
                *[f"  - {violation}" for violation in violations],
            ],
        )
    if isinstance(error, UnknownTradeIdeaError):
        return _failure(
            command,
            args,
            CliErrorCode.IDEA_NOT_FOUND,
            str(error),
            details=_error_context(error),
        )
    if isinstance(error, InvalidTransitionError):
        return _failure(
            command,
            args,
            CliErrorCode.VALIDATION_ERROR,
            str(error),
            details=_error_context(error),
        )
    if isinstance(error, (AuditIntegrityError, BudgetIntegrityError)):
        return _failure(
            command,
            args,
            CliErrorCode.OPERATION_FAILED,
            str(error),
            details=_error_context(error),
        )
    if isinstance(error, ValidationError):
        return _failure(
            command,
            args,
            CliErrorCode.VALIDATION_ERROR,
            str(error),
            details=_error_context(error),
        )
    return _failure(
        command,
        args,
        CliErrorCode.OPERATION_FAILED,
        str(error),
        details={"exception_type": type(error).__name__},
    )


def _failure(
    command: str,
    args: Namespace,
    code: CliErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    data: Any = None,
    text_lines: list[str] | None = None,
) -> CliResponse:
    output_format = _output_format(args)
    response_data = data
    if output_format == "text":
        response_data = "\n".join(text_lines or [f"✗ {command} FAILED: {message}"])
    return CliResponse(
        success=False,
        command=command,
        data=response_data,
        errors=[CliError.from_code(code, message, **(details or {}))],
        exit_code=1,
    )


def _error_context(error: Exception) -> dict[str, Any]:
    context = getattr(error, "context", None)
    if isinstance(context, dict):
        return context
    return {}


def _status_line(command: str, status: str, details: str) -> str:
    return f"✓ {command} {status} ({details})"


def _ideas_table(ideas: list[dict[str, Any]]) -> str:
    if not ideas:
        return _status_line("ideas list", "OK", "0 ideas")
    lines = ["DECISION_ID  STATE  INSTRUMENT  DIRECTION  MAX_LOSS%  EXPIRES_AT"]
    for idea in ideas:
        lines.append(
            "{decision_id}  {state}  {instrument}  {direction}  {max_loss_pct}  {expires_at}".format(
                decision_id=idea["decision_id"],
                state=idea["state"],
                instrument=idea["instrument"],
                direction=idea["direction"],
                max_loss_pct=idea["max_loss_pct"] or "",
                expires_at=idea["expires_at"] or "",
            )
        )
    return "\n".join(lines)


def _record_text(payload: dict[str, Any], *, include_events: bool) -> str:
    lines = [
        f"decision_id: {payload['decision_id']}",
        f"state: {payload['state']}",
        f"instrument: {payload['instrument']}",
        f"product_type: {payload['product_type']}",
        f"direction: {payload['direction']}",
        f"thesis: {payload['thesis']}",
        f"invalidation: {payload['invalidation']}",
        f"target_exit: {payload['target_exit']}",
        f"max_loss.percent_of_account: {payload['max_loss']['percent_of_account']}",
        f"expires_at: {payload['time_horizon']['expires_at']}",
        f"confidence: {payload['confidence']['label']}",
    ]
    if include_events:
        lines.append("")
        lines.append("TIMESTAMP  ACTOR  ACTION  TRANSITION  REASON")
        lines.append(_events_text(payload.get("events", [])))
    return "\n".join(lines)


def _events_text(events: list[dict[str, Any]]) -> str:
    if not events:
        return "No audit events found."
    lines: list[str] = []
    for event in events:
        before = event["before_state"] or "none"
        after = event["after_state"]
        lines.append(
            "{timestamp}  {actor_type}/{actor_id}  {action}  {before}->{after}  {reason}".format(
                timestamp=event["timestamp"],
                actor_type=event["actor_type"],
                actor_id=event["actor_id"],
                action=event["action"],
                before=before,
                after=after,
                reason=event["reason"],
            )
        )
    return "\n".join(lines)


def _budget_text(payload: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in payload.items())


def _replay_report_text(report: ReplayReport) -> str:
    average_return_r = report.average_return_r
    return "\n".join(
        [
            _status_line(
                "ideas replay baseline",
                "OK",
                (
                    f"{report.symbol} {report.granularity}, "
                    f"snapshots={report.snapshots_evaluated}, "
                    f"ideas={report.ideas_proposed}"
                ),
            ),
            f"proposer_id: {report.proposer_id}",
            (
                "outcomes: "
                f"target_hits={report.target_hits}, "
                f"stop_hits={report.stop_hits}, "
                f"timed_out={report.timed_out}, "
                f"not_filled={report.not_filled}, "
                f"no_future_data={report.no_future_data}"
            ),
            (
                "hit_rates: "
                f"target={_decimal_pct(report.target_hit_rate)}, "
                f"stop={_decimal_pct(report.stop_hit_rate)}"
            ),
            (
                "average_return_r: "
                f"{average_return_r.normalize() if average_return_r is not None else 'n/a'}"
            ),
        ]
    )


def _decimal_pct(value: Decimal) -> str:
    return f"{(value * Decimal('100')).quantize(Decimal('0.01'))}%"


def _closeout_text(command: str, payload: dict[str, Any]) -> str:
    lines = [
        _status_line(
            command, "OK", f"{payload['decision_id']}, resolution={payload['resolution']}"
        ),
        f"actor: {payload['actor_type']}/{payload['actor_id']}",
        f"realized_profit_loss_amount: {payload['realized_profit_loss_amount']}",
        f"realized_profit_loss_percent: {payload['realized_profit_loss_percent']}",
        "realized_profit_loss_unavailable_reason: "
        f"{payload['realized_profit_loss_unavailable_reason']}",
        f"terminal_event_id: {payload['terminal_event_id']}",
        f"record_hash: {payload['record_hash']}",
    ]
    evidence = payload.get("evidence", [])
    if evidence:
        lines.append("evidence:")
        lines.extend(f"- {item}" for item in evidence)
    return "\n".join(lines)
