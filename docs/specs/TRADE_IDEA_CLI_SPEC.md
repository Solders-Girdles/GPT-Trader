---
status: current
last-updated: 2026-06-24
workstream: 1 (see TRADE_IDEA_INTERFACES_DESIGN_NOTES.md)
depends-on: Workstream 0 (implemented service factory, error codes, actor resolution)
---

# Implemented Spec: `gpt-trader ideas` CLI Command Group

## Goal

A complete, scriptable CLI surface over `TradeIdeaService` so that AI agents
can propose trade-idea records and humans can review them, with JSON output
for programmatic use. This is the first working door into the staged-autonomy
workflow and is implemented in `src/gpt_trader/cli/commands/ideas.py`.

This document is now a maintainer reference for the implemented CLI contract,
not a request to build a new command group. Future work should update the
current command and tests rather than duplicating the trade-ideas interface.

## Files

| File | Action |
|------|--------|
| `src/gpt_trader/cli/commands/ideas.py` | Implemented command group |
| `src/gpt_trader/cli/__init__.py` | Registers `ideas.register(subparsers)` alongside existing commands |
| `src/gpt_trader/cli/response.py` | Defines `POLICY_VIOLATION`, `IDEA_NOT_FOUND` in `CliErrorCode` |
| `src/gpt_trader/features/trade_ideas/service.py` | Defines `DEFAULT_IDEAS_ROOT`, `create_trade_idea_service()` factory |
| `tests/unit/gpt_trader/cli/commands/test_ideas_*.py` | Implemented CLI tests split by lifecycle/query/budget and focused integrity cases |

The implementation follows the structure of `cli/commands/controls.py`: a
`register(subparsers)` function, one `_handle_*` per subcommand returning
`CliResponse`, with `options.add_output_options(parser)` on every subcommand.

## Common options

Every subcommand accepts:

- `--format {text,json}` (via `options.add_output_options`)
- `--ideas-root PATH` — override storage root (default: `GPT_TRADER_IDEAS_ROOT`
  env, then `var/data/trade_ideas/`)

Every mutating subcommand accepts:

- `--actor ID` — actor identity (default: `GPT_TRADER_ACTOR` env, then OS user)

Current behavior is discoverable with:

```bash
uv run gpt-trader ideas --help
```

## Command tree

```
gpt-trader ideas
├── propose          --file PATH | --stdin   [--actor-type {ai,human}] [--reason TEXT]
├── resubmit         --file PATH | --stdin   [--actor-type {ai,human}] [--reason TEXT]
├── list             [--state STATE]
├── show             DECISION_ID [--events]
├── approve          DECISION_ID --reason TEXT
├── reject           DECISION_ID --reason TEXT
├── request-changes  DECISION_ID --reason TEXT
├── cancel           DECISION_ID --reason TEXT
├── expire           DECISION_ID [--reason TEXT] | --sweep
├── mark-submitted   DECISION_ID --venue {coinbase,manual} [--external-order-id ID] [--reason TEXT]
├── mark-filled      DECISION_ID --venue {coinbase,manual} [--external-order-id ID] [--reason TEXT]
├── budget
│   ├── show
│   └── set          --reason TEXT [one flag per RiskBudget field]
└── audit
    ├── tail         [-n N] [--decision-id ID]
    └── verify
```

### `ideas propose`

- Input: a `TradeIdea` JSON document (the `to_dict` shape from
  `features/trade_ideas/models.py`) from `--file` or stdin. `--file` and
  `--stdin` are mutually exclusive; one is required.
- Parse via `TradeIdea.from_dict`. `KeyError`/`ValueError` →
  `INVALID_ARGUMENT` naming the missing/invalid field.
- Call `service.propose(idea, actor_id=..., actor_type=...)`.
- On success, `data` contains `decision_id`, `state`, `record_hash`, and the
  eligibility preview: `violations` from
  `ApprovalPolicy().approval_violations(...)` evaluated read-only with the
  current budget, so the proposer immediately sees whether the idea could be
  approved as-is. Preview violations are `warnings`, not errors — propose
  still succeeds (rejection happens at review, per the framework).
- Text: `✓ ideas propose OK (trade-20260612-001, state=proposed)` followed by
  `⚠ would fail approval: <reason>` lines if any.

### `ideas resubmit`

Same input handling as `propose`; calls `service.resubmit`. The record must
already exist (`IDEA_NOT_FOUND` otherwise). Service/audit layer enforces the
`needs_changes → proposed` transition; surface `InvalidTransitionError` as
`VALIDATION_ERROR`.

### `ideas list`

- `--state` choices = `TradeIdeaState` values. Calls
  `service.list_views(state)`.
- Text: aligned table — `DECISION_ID  STATE  INSTRUMENT  DIRECTION
  MAX_LOSS%  EXPIRES_AT`. JSON: `data["ideas"]` = list of
  `{decision_id, state, instrument, direction, max_loss_pct, expires_at,
  confidence}`.
- Empty store is success with an empty list, not an error.

### `ideas show DECISION_ID`

- Calls `service.get`. `data` = full `idea.to_dict()` plus `state`.
- `--events` additionally includes the audit history
  (`[e.to_dict() for e in view.events]`).
- Text: field-per-line record rendering; with `--events`, a chronological
  `timestamp  actor_type/actor_id  action  before→after  reason` table.

### `ideas approve DECISION_ID --reason TEXT`

- `actor_type` hard-coded `HUMAN`. `--reason` required and non-empty.
- `PolicyViolationError` → exit 1, code `POLICY_VIOLATION`,
  `data["violations"]` = full list. Text mode:

  ```
  ✗ ideas approve FAILED: approval refused (3 violations)
    - max_loss 8% exceeds budget cap of 5% per idea
    - 2 tickets already approved; budget allows 2 concurrent approved tickets
    - Idea expired at 2026-06-11T20:00:00+00:00; approve nothing stale
  ```

### `ideas reject` / `request-changes` / `cancel`

Thin wrappers over `service.reject` / `service.request_changes` /
`service.cancel` with `actor_type=HUMAN` and required `--reason`.

### `ideas expire`

- `ideas expire DECISION_ID` — expire one idea (service defaults for
  reason/actor unless overridden).
- `ideas expire --sweep` — list all non-terminal views and expire each idea
  whose `time_horizon.expires_at` is set and `<= now`, or whose review time
  exceeds `RiskBudget.max_review_latency_hours`. Report `data["expired"]` =
  list of decision_ids; success even when zero matched (`was_noop=True`).
  DECISION_ID and `--sweep` are mutually exclusive; one is required.

### `ideas mark-submitted` / `ideas mark-filled`

- Wrap `service.record_submission` / `service.record_fill`.
- **These record manually executed tickets; they never touch a broker API.**
  Help text says so, and this boundary must remain explicit in future changes.
- `--venue` choices: `coinbase`, `manual` (no INTX-specific venue surface).

### `ideas budget show` / `ideas budget set`

- `show`: `data` = `service.current_budget().to_dict()` (this seeds defaults
  on first use — acceptable and documented).
- `set`: one optional flag per `RiskBudget` field
  (`--max-loss-per-idea-pct`, `--max-daily-loss-pct`,
  `--max-open-notional-pct`, `--max-concurrent-approved-tickets`,
  `--max-review-latency-hours`, `--sizing-capped-by-budget {true,false}`,
  `--gain-retention-floor-pct`, `--allow-futures-leverage {true,false}`,
  `--allow-naked-shorts {true,false}`), plus required `--reason`. Build the
  new `RiskBudget` by copying the current one with overrides and
  `version = current.version + 1`; call
  `service.update_budget(budget, ActorType.HUMAN, actor_id)`.
  At least one field flag is required (`MISSING_ARGUMENT` otherwise).
  `PolicyViolationError` → `POLICY_VIOLATION`.

### `ideas audit tail` / `ideas audit verify`

- `tail`: last N events (default 20) from
  `service.audit_log.read_events(decision_id)`, newest last.
- `verify`: full read of the audit log; `AuditIntegrityError` →
  `✗ ideas audit verify FAILED: <reason>` with `OPERATION_FAILED`. Success
  reports event count: `✓ ideas audit verify OK (142 events)`.

## Output standards

Per CLAUDE.md: text mode prints `✓ ideas <action> OK (<details>)` /
`✗ ideas <action> FAILED: <error>`; exit 0 on success, 1 on failure; JSON
mode returns the `CliResponse` envelope with `command="ideas <subcommand>"`.

## Serialization notes

- `Decimal` fields (`MaxLoss`, `RiskBudget`) serialize as strings via the
  models' own `to_dict`; do not re-serialize with `float()`.
- Timestamps render ISO-8601 UTC.
- Never print secrets; the audit contract forbids credentials in any record.

## Implemented test plan (tests/unit/gpt_trader/cli/commands/)

Use a `tmp_path`-rooted service via `--ideas-root` (no monkeypatching of
globals needed; this is why the flag exists). Fixture helper builds a valid
`TradeIdea` dict matching `eligibility` requirements.

Required cases:

1. `propose` from file and stdin → success, record on disk, audit event
   appended, eligibility warnings surfaced for a non-compliant idea.
2. `propose` with missing required field → `INVALID_ARGUMENT`, names field.
3. `list` empty store → success, empty list. `list --state proposed` filters.
4. `show` unknown id → `IDEA_NOT_FOUND`. `show --events` includes history.
5. `approve` happy path → state `approved`, human actor in audit event.
6. `approve` over-budget idea → exit 1, `POLICY_VIOLATION`, all violations in
   `data["violations"]` (assert ≥2 violations both present).
7. `request-changes` → `resubmit` (revised record) → `approve` full loop.
8. `reject`, `cancel`, `expire` single, `expire --sweep` (one stale + one
   fresh idea: only stale expires; `was_noop` when none).
9. `mark-submitted` then `mark-filled` with venue/external id recorded in
   audit events.
10. `budget show` seeds defaults; `budget set --max-loss-per-idea-pct 2
    --reason ...` bumps version; `budget set` with no field flags →
    `MISSING_ARGUMENT`.
11. `audit verify` OK path; tampered line in `audit.jsonl` → failure.
12. JSON mode for at least propose/approve/list asserting the
    `CliResponse` envelope per CLAUDE.md patterns
    (`result.errors[0].code == CliErrorCode.POLICY_VIOLATION.value`).

## Acceptance criteria

- [x] `gpt-trader ideas --help` lists all subcommands with accurate help text.
- [x] Full loop works on a clean checkout with no env vars:
      propose → list → show → approve → mark-submitted → mark-filled,
      and the audit log verifies afterward.
- [x] No import from `features/brokerages/` or `features/live_trade/` in
      `ideas.py` (enforce by review; this surface must stay execution-free).
- [x] All policy violations reach the user; no first-error-only truncation.
- [x] The focused `tests/unit/gpt_trader/cli/commands/test_ideas_*.py` suite
      covers the implemented command group.
- [ ] Re-run the repo-required quality bundle before merging any future change
      that modifies this CLI or its tests.
