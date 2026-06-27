---
status: current
last-updated: 2026-06-27
scope: Operator and agent interfaces over the trade_ideas slice
audience: Implementation agents (Codex) and reviewers
---

# Trade-Idea Interfaces — Design Notes

## Context

The accepted direction (docs/PRE_MIGRATION_DECISION_FRAMEWORK.md) is staged
autonomy: AI drafts complete trade-idea records, a human approves them, and
every state change lands in an append-only audit log. The core slice for this
already exists and is complete at the domain level:

| Module | Provides |
|--------|----------|
| `features/trade_ideas/models.py` | `TradeIdea` record (frozen dataclass, `to_dict`/`from_dict`, `record_hash`) |
| `features/trade_ideas/workflow.py` | `TradeIdeaState`, `ALLOWED_TRANSITIONS`, `validate_transition` |
| `features/trade_ideas/service.py` | `TradeIdeaService` — the one audited code path for every actor |
| `features/trade_ideas/policy.py` | `ApprovalPolicy` — autonomy mode as enforceable checks |
| `features/trade_ideas/budget.py` | `RiskBudget`, versioned `RiskBudgetLog`, seeded defaults |
| `features/trade_ideas/audit.py` | Append-only `TradeIdeaAuditLog` (JSONL), `AuditEvent` |
| `features/trade_ideas/store.py` | `TradeIdeaStore` — versioned records under record hash |
| `features/trade_ideas/baseline.py`, `replay.py` | Baseline proposer and replay scoring |

**Current interface state:** the CLI and TUI review surfaces now exist.
`gpt-trader ideas` constructs `TradeIdeaService` through the trade-ideas factory
and exposes the agent-facing approval workflow: propose, list, show, approve,
reject, request-changes, expire, resubmit, mark-submitted, mark-filled, budget,
audit, and report. It also exposes the read-only Stage 1 baseline calibration
surface, `gpt-trader ideas replay baseline`, which parses a local candle fixture
and formats `TradeIdeaReplayRunner` / `ReplayReport` output without using
`TradeIdeaService` storage. The TUI `IdeasReviewScreen` is the implemented human
review surface: it lists proposed ideas, renders record details, previews policy
violations, and records approve, reject, request-changes, and expire decisions
through `TradeIdeaService`. MCP or other remote surfaces remain future work. The
service docstring still states the adapter boundary explicitly: *"interfaces such
as CLI, TUI, or MCP servers must stay thin adapters over these methods."*

These notes preserve the interface decisions that shaped the implemented CLI and
TUI surfaces. They are not a request to re-promote the TUI review workstream.

## Design Principles

1. **Thin adapters only.** Interfaces parse input, resolve actor identity,
   call one `TradeIdeaService` method, and render the result. No workflow,
   policy, or budget logic in CLI/TUI code. If an interface needs a new
   behavior, it goes into the service first. The read-only replay baseline
   calibration command is the explicit exception: it is not a workflow mutation
   or storage adapter, so it may call `TradeIdeaReplayRunner` directly and
   render `ReplayReport` without constructing `TradeIdeaService`.
2. **Every action is identity-stamped.** No anonymous mutations. Each
   mutating command requires an `actor_id` and an `actor_type`; review
   actions from interactive interfaces are always `ActorType.HUMAN`.
3. **No execution lane.** Nothing in these workstreams places, modifies, or
   cancels broker orders. `mark-submitted` / `mark-filled` are audit
   bookkeeping for manually executed tickets, not order routing.
4. **Policy refusals are first-class UX.** `PolicyViolationError.violations`
   is a list of every reason an approval was refused. Interfaces must show
   the full list, never just the first reason or a generic failure.
5. **JSON-first for agents.** All CLI commands support
   `--format json` via the existing `CliResponse` envelope so Codex/Claude
   and CI can drive the workflow programmatically. Text output follows the
   `✓`/`✗` standards in CLAUDE.md.
6. **Append-only mindset.** Interfaces never edit records in place. A change
   request produces a `needs_changes` event; the revised record is a new
   version saved via `resubmit`.

## Shared Decisions (Workstream 0 — implemented)

The CLI implements these decisions today. The TUI should reuse them instead of
creating a parallel service or storage contract.

### Storage root

- Default root: `var/data/trade_ideas/` (consistent with the existing
  `var/data/status.json` convention). The service derives
  `records/`, `audit.jsonl`, and `risk_budget.jsonl` under it.
- Override: environment variable `GPT_TRADER_IDEAS_ROOT`; CLI also accepts
  `--ideas-root PATH` (highest precedence) for tests and sandboxing.

### Service factory

Implemented in `features/trade_ideas/service.py`:

```python
DEFAULT_IDEAS_ROOT = Path("var/data/trade_ideas")

def create_trade_idea_service(root: Path | None = None) -> TradeIdeaService:
    """Resolve root (arg > GPT_TRADER_IDEAS_ROOT > default) and build the service."""
```

The CLI constructs the service directly through this factory because idea
review has no broker or config dependency. A cached `trade_idea_service`
property on `ApplicationContainer` remains a future option once a proposer loop
runs inside the bot.

### Actor identity resolution

Precedence for `actor_id`: `--actor` flag → `GPT_TRADER_ACTOR` env var →
`getpass.getuser()`. The resolved value is recorded verbatim in the audit
log. `actor_type` rules:

| Action | actor_type |
|--------|-----------|
| `propose`, `resubmit` | `ai` by default; `--actor-type human` allowed |
| `approve`, `reject`, `request-changes`, `cancel` | always `human` (the policy enforces this for approve; interfaces hard-code it for the rest) |
| `expire` (sweep) | `system` |
| `mark-submitted` | `system` (default per service) or `human` |
| `mark-filled` | `venue` (default per service) |
| budget `set` | `human` (policy refuses non-human in current mode) |

### Error mapping

Implemented in `CliErrorCode` in `cli/response.py`:
`POLICY_VIOLATION = "POLICY_VIOLATION"` and
`IDEA_NOT_FOUND = "IDEA_NOT_FOUND"`.

| Exception | CliErrorCode | Exit | Notes |
|-----------|--------------|------|-------|
| `PolicyViolationError` | `POLICY_VIOLATION` | 1 | Put `violations` list in `data["violations"]`; text mode prints each on its own line |
| `UnknownTradeIdeaError` | `IDEA_NOT_FOUND` | 1 | |
| `InvalidTransitionError` | `VALIDATION_ERROR` | 1 | |
| `AuditIntegrityError` / `BudgetIntegrityError` | `OPERATION_FAILED` | 1 | Integrity failures are loud, never swallowed |
| Malformed input JSON / missing fields | `INVALID_ARGUMENT` | 1 | Report the offending field |

## Workstreams

| # | Spec | Depends on | Size |
|---|------|-----------|------|
| 0 | Shared wiring (this doc, "Shared Decisions") | — | Implemented |
| 1 | [`TRADE_IDEA_CLI_SPEC.md`](TRADE_IDEA_CLI_SPEC.md) — `gpt-trader ideas` command group | 0 | Implemented |
| 2 | [`TRADE_IDEA_TUI_REVIEW_SPEC.md`](TRADE_IDEA_TUI_REVIEW_SPEC.md) — Ideas review screen | 0 (not 1) | Implemented |

Workstreams 1 and 2 were independently mergeable and have both shipped.
Workstreams 0+1 provide the agent-facing CLI surface and unblock the
AI-propose -> human-approve loop end to end. Workstream 2 provides the
keyboard-driven human review surface without reimplementing CLI or service
behavior. Future interface work should start from these shipped adapters and
must not treat the TUI review screen as an open backlog item.

## Non-Goals (all workstreams)

- No order submission, modification, or cancellation through any broker API.
- No broker-ticket payload generation (derived artifacts come later, after
  approval-lane experience).
- No bounded-autonomy behavior; `ApprovalPolicy` defaults stand.
- No INTX surfaces (frozen).
- No MCP server (a future workstream; the CLI JSON mode is the agent surface
  for now).
- No editing of historical records or audit events, ever.

## Conventions Codex Must Follow

- Naming: banned abbreviations `cfg`, `svc`, `mgr`, `util`, `utils`, `amt`,
  `calc`, `upd` (see `docs/naming.md`). Run `uv run agent-naming`.
- Tests: prefer `monkeypatch`; assert on `CliResponse` per CLAUDE.md
  (`result.errors[0].code == CliErrorCode.X.value`). Unit tests live under
  `tests/unit/gpt_trader/...` mirroring source paths.
- Quality gate before PR: `uv run agent-check` (or `/quality`), plus
  `uv run ruff check . --fix`, `uv run black .`, `uv run mypy src/gpt_trader`.
- TUI CSS: edit modules under `src/gpt_trader/tui/styles/`, then run
  `python scripts/build_tui_css.py`; never edit `styles/main.tcss` directly.
  `DEFAULT_CSS` blocks use hardcoded hex with `/* $variable */` comments.
- Import boundaries: `scripts/ci/check_import_boundaries.py` runs in CI —
  keep interface code free of cross-slice imports it would flag.
