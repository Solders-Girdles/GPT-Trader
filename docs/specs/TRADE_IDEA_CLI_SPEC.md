---
status: current
last-updated: 2026-06-27
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

Every storage-backed workflow subcommand accepts:

- `--format {text,json}` (via `options.add_output_options`)
- `--ideas-root PATH` — override storage root (default: `GPT_TRADER_IDEAS_ROOT`
  env, then `var/data/trade_ideas/`)

Read-only replay commands accept `--format {text,json}` but do not read or
write `--ideas-root`; they operate only on the local fixture named by `--file`.

Every mutating subcommand accepts:

- `--actor ID` — actor identity (default: `GPT_TRADER_ACTOR` env, then OS user)

Exception: `propose-baseline` defaults to the deterministic baseline proposer
id so generated proposal audit events identify the proposer unless an operator
explicitly overrides `--actor`.

Current behavior is discoverable with:

```bash
uv run gpt-trader ideas --help
```

## Command tree

```
gpt-trader ideas
├── propose          --file PATH | --stdin   [--actor-type {ai,human}] [--reason TEXT]
├── propose-baseline --snapshot PATH         [--actor ID] [--reason TEXT]
├── snapshot
│   └── build        --from-coinbase --symbols BTC-USD,ETH-USD
│                    --granularity GRANULARITY --lookback N --out snapshot.json
│                    [--as-of TIMESTAMP] [--source-label LABEL]
├── resubmit         --file PATH | --stdin   [--actor-type {ai,human}] [--reason TEXT]
├── list             [--state STATE]
├── show             DECISION_ID [--events]
├── report
├── replay
│   └── baseline     --file PATH --symbol SYMBOL --granularity GRANULARITY
│                    [--short-window N] [--long-window N]
│                    [--crossover-lookback N] [--min-history N]
│                    [--risk-per-idea-pct DECIMAL] [--entry-band-pct DECIMAL]
│                    [--reward-multiple DECIMAL] [--expiry-hours N]
│                    [--expected-hold TEXT] [--price-precision DECIMAL]
│                    [--source LABEL]
├── closeout
│   ├── record       DECISION_ID --resolution {thesis_target,invalidation,expiry}
│   │                [--realized-profit-loss-amount DECIMAL]
│   │                [--realized-profit-loss-percent DECIMAL]
│   │                [--realized-profit-loss-unavailable-reason TEXT]
│   │                [--evidence TEXT]... [--actor-type {human,system}]
│   └── show         DECISION_ID
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

### `ideas propose-baseline`

- Input: a local `MarketSnapshot` JSON fixture from `--snapshot PATH`.
- Never calls broker, account, credential, preflight, canary, or live market
  surfaces. The snapshot file is the complete data source for the command.
- Parse the fixture into `MarketSnapshot` / `SymbolSeries`, run
  `BaselineProposer`, preflight every emitted `TradeIdea` with
  `service.validate_new_proposal`, preview approval policy with
  `service.approval_violations`, then persist through
  `service.propose(..., actor_type=ai)`.
- The default actor id is the deterministic proposer id
  (`baseline-ma-10-50` today); `--actor` overrides it when an operator needs a
  different audit identity.
- No-signal snapshots return success with `proposal_count=0` and
  `was_noop=true`; they do not create trade-idea records.
- Duplicate decision ids fail before any proposed record or audit event is
  written for that invocation.
- JSON `data` contains `proposer_id`, snapshot metadata, `proposal_count`, and
  one entry per proposal:

  ```json
  {
    "decision_id": "trade-20350612-btcusd-4c5a9e2d",
    "state": "proposed",
    "instrument": "BTC-USD",
    "direction": "long",
    "record_hash": "sha256...",
    "approval_preview": {
      "violations": [],
      "warnings": []
    }
  }
  ```

  Approval-preview warnings are also mirrored into the top-level
  `CliResponse.warnings` list for agent callers that already consume warning
  envelopes.

#### `propose-baseline` snapshot fixture shape

The JSON file uses strings for timestamps and decimal fields so fixtures remain
stable across languages and shells:

```json
{
  "as_of": "2035-06-12T00:00:00+00:00",
  "source": "local-fixture:coinbase-candles",
  "series": [
    {
      "symbol": "BTC-USD",
      "granularity": "1d",
      "candles": [
        {
          "ts": "2035-04-20T00:00:00+00:00",
          "open": "100",
          "high": "100",
          "low": "100",
          "close": "100",
          "volume": "1000"
        }
      ]
    }
  ]
}
```

All candle timestamps must include a timezone, be strictly ascending within a
series, and be strictly before `as_of`. The command rejects malformed fixtures
as `INVALID_ARGUMENT`.

### `ideas snapshot build`

- Output: a local `MarketSnapshot` JSON file that can be passed directly to
  `ideas propose-baseline --snapshot PATH`.
- The live market fetch path is explicit: `--from-coinbase` is required. The
  command builds from read-only public Coinbase market candles through the
  existing historical candle abstractions; it never reads accounts, performs
  product/account discovery, runs broker readiness checks, preflight, canary, or
  order-affecting commands.
- Required options:
  - `--from-coinbase`
  - `--symbols BTC-USD,ETH-USD` (comma-separated, unique Coinbase product ids)
  - `--granularity GRANULARITY`
  - `--lookback N` (number of completed candles to include per symbol)
  - `--out PATH`
- Optional options:
  - `--as-of TIMESTAMP` (timezone required; defaults to current UTC time)
  - `--source-label LABEL` (default `coinbase:market-candles`)
  - `--coinbase-base-url URL` (default `https://api.coinbase.com`)
- Point-in-time rules:
  - Fetch window is `[as_of - granularity * lookback, as_of)`.
  - Source candles must be strictly ascending by timestamp.
  - Candle selection includes only fully closed bars: for each source candle,
    `candle.ts + granularity <= as_of`. Candles that start before `as_of` but
    are still open are skipped so current or future bars cannot leak into
    proposer input.
  - Each configured symbol must have at least one completed candle in the
    window.
- JSON `data` contains `out` plus snapshot metadata: `as_of`, `source`,
  `symbols`, and per-series `candle_count`, `first_ts`, and `last_ts`.
- Text starts with:

  ```text
  ✓ ideas snapshot build OK (2 series -> var/snapshots/coinbase.json)
  ```

Recommended Stage 1 workflow:

```bash
uv run gpt-trader ideas snapshot build \
  --from-coinbase \
  --symbols BTC-USD,ETH-USD \
  --granularity ONE_HOUR \
  --lookback 53 \
  --out var/snapshots/coinbase-market-snapshot.json

uv run gpt-trader ideas propose-baseline \
  --snapshot var/snapshots/coinbase-market-snapshot.json \
  --format json
```

For replay calibration, keep using `ideas replay baseline --file` with a local
candle fixture. The snapshot build output is the one-shot point-in-time proposer
input; replay consumes a longer historical candle series and constructs many
point-in-time snapshots internally.

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

### `ideas report`

- Read-only track-record report over the local `--ideas-root` records, audit
  events, and closeout attribution log. It does not call broker, account,
  venue, preflight, canary, or live-trading surfaces.
- Empty stores return success with zero counts and `was_noop=True`.
- JSON `data` contains:
  - `proposal_volume`: idea count, proposal event count, resubmission count,
    and monthly volume/approval/closeout buckets.
  - `workflow`: lifecycle event counts, current-state counts, ever-approved /
    submitted / filled counts, and rates.
  - `quality`: eligibility and approval-policy quality counts evaluated
    read-only against stored records and the default risk-budget constants,
    without seeding `risk_budget.jsonl`.
  - `closeouts`: terminal closeout coverage, missing terminal closeout ids,
    resolution counts, profit/loss outcome distribution, and realized P/L
    versus max-loss comparisons when closeout records include numeric data.
- Text starts with:

  ```text
  ✓ ideas report OK (3 ideas, approval_rate=33.33%, closeout_coverage=66.67%)
  ```

  When `proposal_volume.by_month` has buckets, text output also includes a
  compact `Monthly` section with one line per month showing idea count, approval
  rate, closeout coverage, and realized P/L amount. Empty stores and reports
  without monthly buckets omit the section while still returning success.

### `ideas replay baseline`

- Read-only Stage 1 calibration command over local candle fixtures. It
  constructs `BaselineProposer(BaselineProposerConfig(...))`, feeds it through
  `TradeIdeaReplayRunner`, and formats the returned `ReplayReport`.
- It never reads credentials, contacts brokers/accounts/live market data, writes
  trade-idea records, creates tickets, or touches closeout attribution.
- Required options:
  - `--file PATH`
  - `--symbol SYMBOL`
  - `--granularity GRANULARITY`
- Supported baseline/replay options:
  - `--source LABEL` (default `fixture:candles`)
  - `--min-history N` (default `max(short-window, long-window) + crossover-lookback`)
  - `--short-window N` (default `10`)
  - `--long-window N` (default `50`)
  - `--crossover-lookback N` (default `3`)
  - `--risk-per-idea-pct DECIMAL` (default `2`)
  - `--entry-band-pct DECIMAL` (default `1`)
  - `--reward-multiple DECIMAL` (default `2`)
  - `--expiry-hours N` (default `48`)
  - `--expected-hold TEXT` (default `5-15 days`)
  - `--price-precision DECIMAL` (default `0.01`)
- Candle fixture shape:

  ```json
  {
    "candles": [
      {
        "ts": "2026-06-12T12:00:00+00:00",
        "open": "100",
        "high": "102",
        "low": "99",
        "close": "101",
        "volume": "1000"
      }
    ]
  }
  ```

  Timestamps are ISO-8601; naive timestamps are treated as UTC. Decimal values
  may be strings or JSON numbers. Malformed JSON, a missing `candles` array,
  missing candle fields, invalid timestamps, non-finite decimals, and invalid
  OHLCV rows (`high < low`, open/close outside `[low, high]`, or negative
  volume) return `INVALID_ARGUMENT` with the offending field when available.
- JSON `data` is `ReplayReport.to_dict()` and includes `proposer_id`,
  `symbol`, `granularity`, `source`, `snapshots_evaluated`, `ideas_proposed`,
  `target_hits`, `stop_hits`, `timed_out`, `not_filled`, `no_future_data`,
  `resolved_ideas`, `target_hit_rate`, `stop_hit_rate`, `average_return_r`, and
  per-idea replay outcomes.
- Text starts with:

  ```text
  ✓ ideas replay baseline OK (BTC-USD ONE_HOUR, snapshots=2, ideas=1)
  proposer_id: baseline-ma-2-4
  outcomes: target_hits=1, stop_hits=0, timed_out=0, not_filled=0, no_future_data=0
  hit_rates: target=100.00%, stop=0.00%
  average_return_r: 2
  ```

  A replay with zero proposed ideas succeeds with `was_noop=True`.

### `ideas closeout record` / `ideas closeout show`

- `record` wraps `service.record_closeout_attribution`; `show` wraps
  `service.get_closeout_attribution`. Both use only local trade-idea storage.
- `record` requires a terminal idea. The service enforces the terminal-state
  precondition and pins the attribution to the latest terminal audit event and
  record hash.
- `record` accepts:
  - `--resolution {thesis_target,invalidation,expiry}`
  - `--realized-profit-loss-amount DECIMAL` and/or
    `--realized-profit-loss-percent DECIMAL` (negative values represent losses)
  - `--realized-profit-loss-unavailable-reason TEXT` when numeric realized P/L
    is unavailable
  - repeated `--evidence TEXT` strings
  - `--actor-type {human,system}` with default `human`
- At least one realized P/L value or unavailable reason is required.
- JSON `data` for both successful commands is
  `{decision_id, closeout_attribution}`. `closeout_attribution` is the
  persisted closeout record dictionary, or `null` for `show` when the idea is
  known but has no attribution (`was_noop=True`).
- Text starts with:

  ```text
  ✓ ideas closeout record OK (trade-20260612-001, resolution=thesis_target)
  ```

- These commands never call broker, account, venue, preflight, canary, ticket
  payload-generation, or live-trading surfaces. Evidence strings are operator
  references only.

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
  that can legally transition to expired when either:
  - `time_horizon.expires_at` is set and `<= now`; or
  - the review deadline has elapsed under the current
    `RiskBudget.max_review_latency_hours` policy.
  The review-latency path can expire a far-future idea whose `expires_at` is
  still later than `now` when the review queue exceeds the configured budget.
  Report `data["expired"]` = list of decision_ids; success even when zero
  matched (`was_noop=True`). DECISION_ID and `--sweep` are mutually exclusive;
  one is required.

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
5. `report` empty store, normal records, missing closeout coverage, JSON
   output, and read-only behavior that does not create `risk_budget.jsonl`.
6. `replay baseline` text success, malformed fixture input, empty/no-idea
   replay success with `was_noop=True`, JSON output exposing `ReplayReport`
   aggregate fields, precision-preserving JSON-number candle parsing, custom
   moving-average history defaults, semantically invalid OHLCV fixture rows,
   and help text for required read-only flags.
7. `closeout record` for a terminal filled idea with realized amount/percent
   and repeated evidence; `closeout show` returns the persisted attribution.
8. `closeout record` for an expired idea with unavailable P/L reason; proposed
   ideas fail with `VALIDATION_ERROR`; missing realized P/L input fails with
   `MISSING_ARGUMENT`; `closeout show` without attribution succeeds with
   `was_noop=True`.
9. `approve` happy path → state `approved`, human actor in audit event.
10. `approve` over-budget idea → exit 1, `POLICY_VIOLATION`, all violations in
   `data["violations"]` (assert ≥2 violations both present).
11. `request-changes` → `resubmit` (revised record) → `approve` full loop.
12. `reject`, `cancel`, `expire` single, `expire --sweep` with explicit expiry
   coverage (one stale + one fresh idea: only stale expires; `was_noop` when
   none) plus review-latency sweep coverage for a far-future idea whose review
   deadline exceeds `max_review_latency_hours`.
13. `mark-submitted` then `mark-filled` with venue/external id recorded in
   audit events.
14. `budget show` seeds defaults; `budget set --max-loss-per-idea-pct 2
    --reason ...` bumps version; `budget set` with no field flags →
    `MISSING_ARGUMENT`.
15. `audit verify` OK path; tampered line in `audit.jsonl` → failure.
16. JSON mode for at least propose/approve/list/report/replay baseline asserting the
    `CliResponse` envelope per CLAUDE.md patterns
    (`result.errors[0].code == CliErrorCode.POLICY_VIOLATION.value`).
14. `propose-baseline` success from a local snapshot fixture, no-signal
    no-op behavior, duplicate decision handling, malformed snapshot input, and
    JSON output with decision ids, record hashes, states, and approval-preview
    warnings/violations.

## Acceptance criteria

- [x] `gpt-trader ideas --help` lists all subcommands with accurate help text.
- [x] `gpt-trader ideas replay baseline --help` documents the local fixture
      replay surface and its broker-free boundary.
- [x] Full loop works on a clean checkout with no env vars:
      propose → list → show → approve → mark-submitted → mark-filled,
      and the audit log verifies afterward.
- [x] No import from `features/brokerages/` or `features/live_trade/` in
      `ideas.py` (enforce by review; this surface must stay execution-free).
- [x] All policy violations reach the user; no first-error-only truncation.
- [x] `propose-baseline` turns local candle fixtures into proposed records
      without broker/API access, and reports approval-preview warnings per
      proposal.
- [x] The focused `tests/unit/gpt_trader/cli/commands/test_ideas_*.py` suite
      covers the implemented command group.
- [ ] Re-run the repo-required quality bundle before merging any future change
      that modifies this CLI or its tests.
