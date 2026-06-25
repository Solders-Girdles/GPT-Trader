"""Trade-idea review screen.

This screen is deliberately not wired into ``StateRegistry``/``TuiState``:
trade-idea review is reviewer workflow state backed by the trade-ideas store,
not bot telemetry. The TUI stays a thin adapter over ``TradeIdeaService`` and
does not place, modify, or cancel broker orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Input, Label, Static

from gpt_trader.errors import ValidationError
from gpt_trader.features.trade_ideas import (
    ALLOWED_TRANSITIONS,
    ActorType,
    AuditEvent,
    PolicyViolationError,
    TradeIdea,
    TradeIdeaService,
    TradeIdeaState,
    TradeIdeaView,
    create_trade_idea_service,
    resolve_trade_idea_actor_id,
)

ReviewAction = Literal["approve", "reject", "request_changes", "expire"]


@dataclass(frozen=True, slots=True)
class ReviewActionSpec:
    action: ReviewAction
    title: str
    target_state: TradeIdeaState


ACTION_SPECS: dict[ReviewAction, ReviewActionSpec] = {
    "approve": ReviewActionSpec("approve", "Approve", TradeIdeaState.APPROVED),
    "reject": ReviewActionSpec("reject", "Reject", TradeIdeaState.REJECTED),
    "request_changes": ReviewActionSpec(
        "request_changes", "Request Changes", TradeIdeaState.NEEDS_CHANGES
    ),
    "expire": ReviewActionSpec("expire", "Expire", TradeIdeaState.EXPIRED),
}


class ReasonModal(ModalScreen[str | None]):
    """Collect a required reason before a review mutation."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+s", "submit", "Submit", show=True),
    ]

    def __init__(
        self,
        *,
        title: str,
        decision_id: str,
        policy_violations: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._decision_id = decision_id
        self._policy_violations = policy_violations or []

    def compose(self) -> ComposeResult:
        with Container(id="ideas-reason-modal"):
            yield Label(f"{self._title}: {self._decision_id}", classes="modal-title")
            if self._policy_violations:
                yield Static("Current approval policy refusals:", classes="modal-warning")
                for violation in self._policy_violations:
                    yield Static(f"- {violation}", classes="modal-warning")
            yield Input(placeholder="Required reason", id="ideas-reason-input")
            with Horizontal(classes="button-row"):
                yield Button("Confirm", id="ideas-reason-confirm", variant="primary", disabled=True)
                yield Button("Cancel", id="ideas-reason-cancel", variant="default")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "ideas-reason-input":
            return
        self.query_one("#ideas-reason-confirm", Button).disabled = not bool(event.value.strip())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ideas-reason-cancel":
            self.dismiss(None)
            return
        if event.button.id == "ideas-reason-confirm":
            self.action_submit()

    def action_submit(self) -> None:
        reason = self.query_one("#ideas-reason-input", Input).value.strip()
        if not reason:
            return
        self.dismiss(reason)

    def action_cancel(self) -> None:
        self.dismiss(None)


class IdeasReviewScreen(Screen[None]):
    """Keyboard-first trade-idea review surface."""

    BINDINGS = [
        Binding("a", "review_action('approve')", "Approve", show=True),
        Binding("x", "review_action('reject')", "Reject", show=True),
        Binding("c", "review_action('request_changes')", "Changes", show=True),
        Binding("e", "review_action('expire')", "Expire", show=True),
        Binding("f", "cycle_filter", "Filter", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    FILTERS: tuple[TradeIdeaState | None, ...] = (
        None,
        TradeIdeaState.PROPOSED,
        TradeIdeaState.NEEDS_CHANGES,
        TradeIdeaState.APPROVED,
    )

    def __init__(
        self,
        service: TradeIdeaService | None = None,
        *,
        reviewer_id: str | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._reviewer_id = reviewer_id
        self._views: list[TradeIdeaView] = []
        self._selected_decision_id: str | None = None
        self._filter_index = 0

    @property
    def reviewer_id(self) -> str:
        if self._reviewer_id is None:
            self._reviewer_id = resolve_trade_idea_actor_id()
        return self._reviewer_id

    @property
    def service(self) -> TradeIdeaService:
        if self._service is None:
            self._service = create_trade_idea_service()
        return self._service

    def compose(self) -> ComposeResult:
        with Container(id="ideas-review"):
            yield Label("", id="ideas-review-title")
            yield Label("", id="ideas-review-filter")
            with Horizontal(id="ideas-review-body"):
                with Vertical(id="ideas-review-queue-pane"):
                    yield Label("Queue", classes="pane-title")
                    queue = DataTable(id="ideas-review-table", zebra_stripes=True)
                    queue.cursor_type = "row"
                    yield queue
                with VerticalScroll(id="ideas-review-detail-pane"):
                    yield Static("Select a trade idea", id="ideas-review-detail")
            yield Label(
                "[a]pprove [x]reject [c]hanges [e]xpire [f]ilter [r]efresh [esc]back",
                id="ideas-review-actions",
            )

    def on_mount(self) -> None:
        table = self.query_one("#ideas-review-table", DataTable)
        table.add_columns("ID", "STATE", "INSTR", "DIR", "LOSS%", "EXPIRES")
        self.query_one("#ideas-review-title", Label).update(
            f"Ideas Review - reviewer: {self.reviewer_id}"
        )
        self.action_refresh()
        self.set_interval(30, self.action_refresh)

    def action_refresh(self) -> None:
        self.refresh_views(notify=False)

    def refresh_views(self, *, notify: bool = True) -> None:
        try:
            self._views = self._sorted_views(self.service.list_views())
        except ValidationError as error:
            self.notify(str(error), title="Ideas Review", severity="error", timeout=6)
            return
        self._render_queue()
        if notify:
            self.notify("Trade ideas refreshed", title="Ideas Review", timeout=2)

    def action_cycle_filter(self) -> None:
        self._filter_index = (self._filter_index + 1) % len(self.FILTERS)
        self._render_queue()
        self.notify(
            f"Filter: {self._filter_label(self.FILTERS[self._filter_index])}",
            title="Ideas Review",
            timeout=2,
        )

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_review_action(self, action: ReviewAction) -> None:
        view = self._selected_view()
        if view is None:
            self.notify("Select a trade idea first", title="Ideas Review", severity="warning")
            return
        spec = ACTION_SPECS[action]
        if spec.target_state not in ALLOWED_TRANSITIONS[view.state]:
            self.notify(
                f"{spec.title} not allowed from state {view.state.value}",
                title="Ideas Review",
                severity="warning",
                timeout=3,
            )
            return
        violations = self.service.approval_violations(view.idea) if action == "approve" else []
        self.app.push_screen(
            ReasonModal(
                title=spec.title,
                decision_id=view.idea.decision_id,
                policy_violations=violations,
            ),
            callback=lambda reason: self._handle_reason(action, view.idea.decision_id, reason),
        )

    def _handle_reason(
        self,
        action: ReviewAction,
        decision_id: str,
        reason: str | None,
    ) -> None:
        if reason is None:
            return
        self._mutate(action, decision_id, reason)

    def _mutate(self, action: ReviewAction, decision_id: str, reason: str) -> None:
        try:
            if action == "approve":
                self.service.approve(decision_id, actor_id=self.reviewer_id, reason=reason)
            elif action == "reject":
                self.service.reject(decision_id, actor_id=self.reviewer_id, reason=reason)
            elif action == "request_changes":
                self.service.request_changes(decision_id, actor_id=self.reviewer_id, reason=reason)
            elif action == "expire":
                self.service.expire(
                    decision_id,
                    actor_id=self.reviewer_id,
                    actor_type=ActorType.HUMAN,
                    reason=reason,
                )
        except PolicyViolationError as error:
            details = "\n".join(error.violations) if error.violations else str(error)
            self.notify(details, title="Approval refused", severity="error", timeout=8)
        except ValidationError as error:
            self.notify(str(error), title="Ideas Review", severity="error", timeout=6)
        else:
            self.notify(f"{ACTION_SPECS[action].title} recorded", title="Ideas Review")
        self.refresh_views(notify=False)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "ideas-review-table":
            return
        self._select_row_key(event.row_key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "ideas-review-table":
            return
        self._select_row_key(event.row_key)

    def _select_row_key(self, row_key: object) -> None:
        value = getattr(row_key, "value", row_key)
        self._selected_decision_id = str(value) if value is not None else None
        self._render_detail()

    def _render_queue(self) -> None:
        table = self.query_one("#ideas-review-table", DataTable)
        table.clear()
        visible = self._visible_views()
        for view in visible:
            idea = view.idea
            table.add_row(
                idea.decision_id,
                view.state.value,
                idea.instrument,
                idea.direction.value,
                self._decimal_text(idea.max_loss.percent_of_account),
                self._datetime_text(idea.time_horizon.expires_at),
                key=idea.decision_id,
            )
        self.query_one("#ideas-review-filter", Label).update(
            f"Filter: {self._filter_label(self.FILTERS[self._filter_index])} "
            f"({len(visible)} of {len(self._views)})"
        )
        if visible and self._selected_decision_id not in {
            view.idea.decision_id for view in visible
        }:
            self._selected_decision_id = visible[0].idea.decision_id
            table.move_cursor(row=0)
        elif not visible:
            self._selected_decision_id = None
        self._render_detail()

    def _render_detail(self) -> None:
        detail = self.query_one("#ideas-review-detail", Static)
        view = self._selected_view()
        if view is None:
            detail.update("No trade ideas match the current filter.")
            return
        detail.update(self._detail_text(view))

    def _selected_view(self) -> TradeIdeaView | None:
        if self._selected_decision_id is None:
            return None
        for view in self._views:
            if view.idea.decision_id == self._selected_decision_id:
                return view
        return None

    def _visible_views(self) -> list[TradeIdeaView]:
        active_filter = self.FILTERS[self._filter_index]
        if active_filter is None:
            return self._views
        return [view for view in self._views if view.state is active_filter]

    def _sorted_views(self, views: list[TradeIdeaView]) -> list[TradeIdeaView]:
        state_rank = {
            TradeIdeaState.PROPOSED: 0,
            TradeIdeaState.NEEDS_CHANGES: 1,
            TradeIdeaState.APPROVED: 2,
        }
        return sorted(
            views,
            key=lambda view: (
                state_rank.get(view.state, 9),
                view.idea.time_horizon.expires_at or datetime.max.replace(tzinfo=UTC),
                view.idea.decision_id,
            ),
        )

    def _detail_text(self, view: TradeIdeaView) -> str:
        idea = view.idea
        sections = [
            f"{idea.decision_id}  [{view.state.value}]",
            "",
            self._idea_fields(idea),
            "",
            "Policy check:",
            *self._policy_lines(idea),
            "",
            "History:",
            *self._history_lines(view.events),
        ]
        return "\n".join(sections)

    def _idea_fields(self, idea: TradeIdea) -> str:
        return "\n".join(
            [
                f"Thesis: {idea.thesis}",
                f"Instrument: {idea.instrument}",
                f"Product: {idea.product_type.value}",
                f"Direction: {idea.direction.value}",
                "Entry zone: "
                f"{self._decimal_text(idea.entry_zone.lower)} - "
                f"{self._decimal_text(idea.entry_zone.upper)}"
                + (f" ({idea.entry_zone.trigger})" if idea.entry_zone.trigger else ""),
                f"Invalidation: {idea.invalidation}",
                f"Target/exit: {idea.target_exit}",
                "Max loss: "
                f"{self._decimal_text(idea.max_loss.amount)} / "
                f"{self._decimal_text(idea.max_loss.percent_of_account)}%",
                f"Loss assumptions: {self._list_text(idea.max_loss.assumptions)}",
                "Sizing: "
                f"qty {self._decimal_text(idea.sizing_recommendation.quantity)}, "
                f"notional {self._decimal_text(idea.sizing_recommendation.notional)}",
                f"Sizing rationale: {idea.sizing_recommendation.rationale}",
                f"Expected hold: {idea.time_horizon.expected_hold}",
                f"Expires: {self._datetime_text(idea.time_horizon.expires_at)}",
                f"Data used: {self._list_text(idea.data_used)}",
                f"Confidence: {idea.confidence.label.value} - {idea.confidence.rationale}",
                f"Failure mode: {idea.failure_mode}",
                f"Do not trade if: {self._list_text(idea.do_not_trade_if)}",
                "Broker ticket: "
                f"{idea.broker_ticket.venue.value}/{idea.broker_ticket.status.value}",
            ]
        )

    def _policy_lines(self, idea: TradeIdea) -> list[str]:
        violations = self.service.approval_violations(idea)
        if not violations:
            return ["PASS would pass approval policy"]
        return [f"FAIL {violation}" for violation in violations]

    def _history_lines(self, events: tuple[AuditEvent, ...]) -> list[str]:
        return [
            " ".join(
                [
                    self._datetime_text(event.timestamp),
                    f"{event.actor_type.value}:{event.actor_id}",
                    event.action.value,
                    f"{event.before_state.value if event.before_state else 'none'}->"
                    f"{event.after_state.value}",
                    f"reason={event.reason}",
                ]
            )
            for event in events
        ]

    def _filter_label(self, state: TradeIdeaState | None) -> str:
        return "all" if state is None else state.value

    def _list_text(self, values: tuple[str, ...]) -> str:
        return ", ".join(values) if values else "none"

    def _datetime_text(self, value: datetime | None) -> str:
        return value.isoformat(timespec="minutes") if value is not None else "-"

    def _decimal_text(self, value: Decimal | None) -> str:
        return str(value) if value is not None else "-"
