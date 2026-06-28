---
status: current
workstream: 2 (see TRADE_IDEA_INTERFACES_DESIGN_NOTES.md)
depends-on: Workstream 0 (service factory, actor resolution); independent of the CLI workstream
---

# Implemented Spec: TUI Trade-Idea Review Screen

## Goal

A Textual screen where the human reviewer — the approval gate of
`human_approved_execution` mode — can triage proposed trade ideas: read the
full thesis/risk record, see exactly which policy checks pass or fail, and
approve, reject, or request changes with a recorded reason. This is the
operator decision point the framework requires.

## Implementation status

This workstream is implemented by `IdeasReviewScreen` and its TUI binding. This
document is retained as the implementation record and maintenance guide, not as
future backlog. The screen remains a review/audit adapter over
`TradeIdeaService`; it does not place, modify, cancel, or route broker orders.

## Scope

Review and audit only. The screen never proposes ideas (that is the AI/CLI
lane), never edits records, and never touches a broker. It works in every TUI
mode including demo, because it reads the filesystem-backed idea store, not
the bot runtime.

## Files

| File | Action |
|------|--------|
| `src/gpt_trader/tui/screens/ideas_review_screen.py` | Implemented — `IdeasReviewScreen(Screen)` + reason-input modal |
| `src/gpt_trader/tui/screens/__init__.py` | Implemented — exports the screen |
| `src/gpt_trader/tui/app.py`, `src/gpt_trader/tui/app_actions.py` | Implemented — `I` binding pushes the screen |
| `src/gpt_trader/tui/styles/screens/ideas_review.tcss` | Implemented — CSS module; generated main theme files include it |
| `tests/unit/gpt_trader/tui/test_ideas_review_screen.py` | Implemented — behavior tests |
| `tests/unit/gpt_trader/tui/test_snapshots_ideas_review.py` | Implemented — snapshot tests |

## Architecture decisions

- **Service access:** construct via `create_trade_idea_service()` on mount
  (root from `GPT_TRADER_IDEAS_ROOT` or default), injectable through the
  screen constructor for tests:
  `IdeasReviewScreen(service: TradeIdeaService | None = None)`.
- **Not wired into `StateRegistry`/`TuiState`:** idea review is reviewer
  workflow state, not bot telemetry. This is deliberate; document it in the
  module docstring. Refresh is manual (`r` key) plus an optional
  `set_interval` poll (30s) of `list_view_result(TradeIdeaListQuery(...))`.
- **Reviewer identity:** resolve once at screen mount (same precedence as
  CLI: `GPT_TRADER_ACTOR` env → OS user); display it in the header so the
  reviewer knows what the audit log will record. All actions from this screen
  use `ActorType.HUMAN`.
- **Service calls are synchronous filesystem I/O** — small store, acceptable
  on the UI thread for v1; wrap in `run_worker` only if list rendering
  measurably stutters.

## Layout

```
┌─ Ideas Review ── reviewer: operator ─────────────────────────────────┐
│ ┌─ Queue (DataTable) ──────────────┐ ┌─ Detail (VerticalScroll) ───┐ │
│ │ ID  STATE  INSTR  DIR  LOSS% EXP │ │ thesis, instrument, entry,  │ │
│ │ ...                              │ │ invalidation, target, loss, │ │
│ │                                  │ │ sizing, horizon, confidence,│ │
│ │                                  │ │ failure_mode, do_not_trade  │ │
│ │                                  │ │ ─────────────────────────── │ │
│ │                                  │ │ Policy check: ✓/✗ list      │ │
│ │                                  │ │ History: audit events       │ │
│ └──────────────────────────────────┘ └─────────────────────────────┘ │
│ [a]pprove [x]reject [c]hanges [e]xpire [f]ilter [r]efresh [esc]back  │
└──────────────────────────────────────────────────────────────────────┘
```

- **Queue table:** columns `decision_id`, `state`, `instrument`,
  `direction`, `max_loss %`, `expires_at`. Default service sort is proposed
  first, then `needs_changes`, then `approved`, then everything else; `f`
  cycles all → proposed → needs_changes → approved. `/` opens an instrument
  filter modal, and `F` clears active filters. The queue header summarizes
  active state/instrument filters. Highlight rows whose `expires_at` is within
  `RiskBudget.max_review_latency_hours` (stale-soon) and rows already past
  expiry.
- **Detail pane:** every field of the selected `TradeIdea` (the reviewer must
  be able to check eligibility from recorded data alone, per Decision 2 of
  the framework). Render `data_used` and `do_not_trade_if` as lists.
- **Policy check section:** call
  `ApprovalPolicy().approval_violations(idea, actor_type=ActorType.HUMAN,
  budget=service.current_budget(),
  open_approved_count=service.open_approved_count(), now=...)` read-only on
  selection. Empty → `✓ would pass approval policy`. Otherwise one `✗` line
  per violation. This preview must use the same policy object the service
  uses — no reimplementation.
- **History section:** the view's audit events,
  `timestamp  actor  action  before→after  reason`.

## Actions

| Key | Action | Allowed when state is | Service call |
|-----|--------|----------------------|--------------|
| `a` | Approve | `proposed` | `approve(id, actor_id, reason)` |
| `x` | Reject | `proposed`, `needs_changes` | `reject(...)` |
| `c` | Request changes | `proposed` | `request_changes(...)` |
| `e` | Expire | any non-terminal | `expire(id, actor_id=reviewer, actor_type=HUMAN, reason=...)` |
| `f` | Cycle state filter | — | `list_view_result(...)` |
| `/` | Apply instrument filter | — | `list_view_result(...)` |
| `F` | Clear filters | — | `list_view_result(...)` |
| `r` | Refresh | — | `list_view_result(...)` |

- Every mutating action opens a **reason modal** (`ModalScreen[str | None]`,
  pattern: `api_setup_wizard.py`): a `TextArea`/`Input` for the reason
  (required, non-empty — disable confirm until text present), confirm +
  cancel buttons. The modal title states the action and decision_id, and for
  approve it re-lists current policy violations so the reviewer confirms
  with full context. (Approval with outstanding violations will be refused
  by the service anyway — the modal is informational, not a bypass.)
- On `PolicyViolationError`: show a notification (`notify` /
  `notification_helpers.py` conventions, severity=error) containing **all**
  violations, keep the idea selected, refresh the policy section.
- On success: notification (severity=information), refresh queue + detail.
- Keys for unavailable transitions are no-ops with a short "not allowed from
  state X" notification (the state machine in `workflow.py` is the truth;
  do not duplicate transition logic — attempt the call or check
  `ALLOWED_TRANSITIONS` directly).

## Styling

- CSS module `tui/styles/screens/ideas_review.tcss`; run
  `python scripts/build_tui_css.py` after editing it. Never edit
  `styles/main.tcss` directly.
- Any `DEFAULT_CSS` uses hardcoded hex + `/* $variable */` comments per
  CLAUDE.md.
- State colors should reuse the theme palette conventions in
  `docs/TUI_STYLE_GUIDE.md` (e.g., success-green for `approved`,
  warning-yellow for `needs_changes`/stale-soon, error-red for
  `rejected`/expired).

## Test plan

Behavior tests are implemented in
`tests/unit/gpt_trader/tui/test_ideas_review_screen.py` using Textual's
`App.run_test()` pilot and a `tmp_path` service injected via the constructor:

1. Empty store → queue renders empty-state message, no crash.
2. Seeded proposed idea appears in queue; selecting it renders all record
   fields and a passing policy section for a compliant idea.
3. Non-compliant idea (e.g., `max_loss` over budget) shows each violation
   line in the policy section.
4. Approve flow: `a` → modal → empty reason cannot confirm → with reason,
   service state becomes `approved`, audit event has
   `actor_type=human` and the typed reason.
5. Approve refused: `PolicyViolationError` surfaces as an error notification
   listing every violation; state unchanged.
6. Reject and request-changes flows append the correct events.
7. Action key on an idea in a terminal state → "not allowed" notification,
   no service mutation.
8. Filter cycling changes the visible row set.

Snapshot tests are implemented in `test_snapshots_ideas_review.py`: empty state
and populated queue with detail pane. For intentional visual changes, update via
`uv run pytest tests/unit/gpt_trader/tui/test_snapshots_*.py --snapshot-update`
and review diffs.

## Acceptance criteria

- [x] Reviewer can fully triage with keyboard only: open screen → select →
      read record + policy verdict + history → approve/reject/request
      changes with a reason → see the result reflected.
- [x] Every mutation lands in the audit log with `actor_type=human`, the
      reviewer's id, and the typed reason; nothing mutates without a reason.
- [x] No policy/workflow logic re-implemented in TUI code (imports from
      `features/trade_ideas` only: service, models, workflow, policy, audit
      types). No imports from brokerage or live_trade slices.
- [x] Screen opens and functions in demo mode.
- [x] CSS built via `scripts/build_tui_css.py`; snapshot tests committed.
- [x] TUI behavior and snapshot coverage committed for the implemented screen.
