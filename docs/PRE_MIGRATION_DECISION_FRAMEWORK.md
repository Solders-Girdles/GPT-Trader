# Pre-Migration Decision Framework

---
status: draft
last-updated: 2026-05-03
scope: AI-assisted trade research and execution migration
---

## Purpose

Use this framework before migrating the existing bot or adding broker execution
paths. It separates three decisions that should not be blended:

1. What AI is allowed to do.
2. Which products the system optimizes for.
3. Where execution happens.

No strategy migration, broker adapter work, or live order-routing change should
start until the migration trigger checklist at the end of this document is
complete.

## Local Baseline

This framework is grounded in current repository docs, not external broker
promises:

- [Architecture](ARCHITECTURE.md) describes GPT-Trader as Coinbase Advanced
  Trade oriented, with spot and CFM futures support and INTX paths gated by
  account access.
- [Coinbase Integration](COINBASE.md) documents Coinbase as the API-native
  broker lane in this repo.
- [Readiness Checklist](READINESS.md) covers paper/live trading readiness. It
  does not replace the AI thesis, approval, and audit gates below.

Robinhood, options execution, and event-contract venues should remain manual or
out of scope until a fresh official API and account-capability review proves
otherwise.

## Recommendation

The v1 migration target should be `human_approved_execution`.

AI may produce complete, broker-ready trade tickets, but a human approves,
changes, rejects, or manually executes them. This preserves a clear operator
decision point while creating the artifacts needed for future automation:
thesis records, risk checks, approval history, and audit logs.

`bounded_autonomy` should be treated as a later milestone. It needs
machine-enforced strategy envelopes, numeric risk budgets, kill-switch coverage,
and proven auditability before any order can be submitted without human
approval.

## Decision 1: Autonomy Boundary

| Mode | AI May Produce | AI May Submit Orders | v1 Status |
| --- | --- | --- | --- |
| `research_only` | Research notes, theses, watchlists | No | Allowed, but too weak as the main target |
| `human_approved_execution` | Broker-ready tickets and risk records | No, approval required first | Recommended default |
| `bounded_autonomy` | Tickets inside pre-approved envelopes | Yes, only inside hard limits | Future milestone |

Hard v1 rule: AI does not submit live orders. Any API execution lane must require
an explicit human approval event first.

## Decision 2: Strategy Eligibility

A strategy is eligible for v1 only if it survives human review latency.

Required properties:

- Holding period is measured in hours or days, not seconds or minutes.
- Entry zone, invalidation, target or exit rule, and max loss are explicit.
- The trade does not depend on immediate fill speed.
- The signal logic is explainable from recorded data, not a hidden "trust me"
  score.
- The expected position can be sized before entry and rejected if the fill moves
  outside the planned zone.
- The idea has a clear expiration time or review deadline.

Automatic rejection conditions:

- No invalidation level.
- No max-loss estimate.
- No reproducible data source.
- Requires continuous intraday babysitting to remain valid.
- Requires unsupported order types, unsupported products, or venue capabilities
  not verified for the target account.

## Decision 3: Product Scope

| Product Lane | v1 Role | Notes |
| --- | --- | --- |
| Regulated futures research | Primary | Best fit for explicit invalidation, max loss, and multi-hour/multi-day theses |
| Options research | Primary manual lane | Generate tickets and risk records; execute manually until API fit is proven |
| Crypto futures | Secondary API-test lane | Use Coinbase CFM or another verified API lane only after account access, product metadata, and risk gates are verified |
| Crypto spot | Research only by default | Do not migrate unrestricted spot automation into the new shape |
| Event contracts | Later evaluation | Needs separate venue, liquidity, fee, and compliance review |
| Mixed derivatives workbench | Later architecture target | Promote only after individual product lanes have compatible schemas and risk gates |

The first migrated system should optimize for futures and options research, with
Coinbase futures as a narrow API canary only when the product and account gates
are confirmed.

## Decision 4: Broker Role

Brokers are venues, not the architecture.

| Venue | Role | Boundary |
| --- | --- | --- |
| Internal record | Canonical thesis, risk, approval, and audit state | Must be broker neutral |
| Coinbase | API-capable crypto and CFM lane | Only for products the repo and account can verify |
| Robinhood | Manual derivatives execution lane | Treat as manual unless official API coverage is proven |
| Future adapters | Optional execution paths | Must adapt to the internal record, not reshape it |

The internal system should model trade intent first. Broker-specific payloads are
derived artifacts created after approval.

## Required Trade Idea Record

Every AI-generated idea must include these fields before review:

| Field | Requirement |
| --- | --- |
| `thesis` | Plain-language reason the trade exists |
| `instrument` | Exact symbol, contract, option chain item, or product identifier |
| `product_type` | Spot, futures, options, event contract, or other explicit type |
| `direction` | Long, short, spread, hedge, or no-trade |
| `entry_zone` | Price range or conditional trigger |
| `invalidation` | Concrete level or condition that makes the thesis false |
| `target_exit` | Target, time stop, trailing rule, or exit condition |
| `max_loss` | Dollar and percent estimate, including assumptions |
| `sizing_recommendation` | Proposed size and how it was calculated |
| `time_horizon` | Expected holding period and review/expiry time |
| `data_used` | Data sources, timestamps, and derived indicators used |
| `confidence` | Bounded confidence label plus why it may be wrong |
| `failure_mode` | Most likely way the trade fails |
| `do_not_trade_if` | Explicit conditions that block execution |

Conceptual shape:

```yaml
trade_idea:
  decision_id: "trade-YYYYMMDD-001"
  autonomy_mode: "human_approved_execution"
  thesis: ""
  instrument: ""
  product_type: ""
  direction: ""
  entry_zone:
    lower: null
    upper: null
    trigger: ""
  invalidation: ""
  target_exit: ""
  max_loss:
    amount: null
    percent_of_account: null
    assumptions: []
  sizing_recommendation:
    quantity: null
    notional: null
    rationale: ""
  time_horizon:
    expected_hold: ""
    expires_at: ""
  data_used: []
  confidence:
    label: "low|medium|high"
    rationale: ""
  failure_mode: ""
  do_not_trade_if: []
  broker_ticket:
    venue: "coinbase|robinhood|manual|none"
    status: "not_created|drafted|approved|submitted|cancelled"
```

This is a contract for migration planning. It should become a JSON Schema or
typed model only after the decisions in this document are accepted.

## Approval Workflow

Minimum v1 workflow:

1. AI creates a `proposed` trade idea record.
2. A reviewer checks strategy eligibility, data freshness, risk, and
   do-not-trade conditions.
3. The reviewer either rejects the idea, requests changes, or approves it.
4. Approved ideas produce a broker-specific ticket.
5. Execution is manual or explicitly triggered by a human approval event.
6. The result is appended to the audit log.

Allowed states:

| State | Meaning |
| --- | --- |
| `proposed` | AI generated the record; no execution allowed |
| `needs_changes` | Human requested edits or missing evidence |
| `rejected` | Human rejected the idea; terminal |
| `approved` | Human approved the ticket; execution may proceed by policy |
| `submitted` | Order was manually entered or API-submitted after approval |
| `filled` | Venue confirmed fill |
| `cancelled` | Human or venue cancelled before fill |
| `expired` | Idea passed its review or execution deadline |

Approval must be append-only. Do not overwrite the original thesis when a human
changes the ticket; append a new event that explains the change.

## Audit Log Contract

The audit log should be append-only JSONL. Minimum event fields:

| Field | Requirement |
| --- | --- |
| `event_id` | Unique event identifier |
| `timestamp` | Time the event was recorded |
| `decision_id` | Trade idea identifier |
| `actor_type` | `ai`, `human`, `system`, or `venue` |
| `actor_id` | Stable identifier for the actor or process |
| `action` | Proposed, changed, approved, rejected, submitted, filled, cancelled, expired |
| `before_state` | Previous workflow state |
| `after_state` | New workflow state |
| `reason` | Why the event occurred |
| `record_hash` | Hash of the trade record version being acted on |
| `evidence` | Paths or references used for the decision |
| `venue` | Broker or manual venue, if applicable |
| `external_order_id` | Venue order identifier, if applicable |

No secrets, API keys, session tokens, or full account credentials belong in the
audit log.

## Risk Budget Gate

The migration must not start until numeric risk limits are chosen. The framework
requires decisions for:

- Max loss per idea.
- Max daily loss across all AI-assisted ideas.
- Max open notional by product type.
- Max concurrent approved-but-unexecuted tickets.
- Max review latency before a ticket expires.
- Whether AI sizing is advisory only or capped by enforceable limits.
- Whether options spreads, futures leverage, and naked short exposure are
  allowed.

Until these are filled in, AI may produce research only or draft tickets marked
`not_approved`.

## Migration Trigger Checklist

Do not migrate the existing bot into this shape until every item is complete:

- [ ] Autonomy mode selected.
- [ ] First asset and product universe selected.
- [ ] Broker execution policy selected.
- [ ] Numeric risk budget selected.
- [ ] Required output schema accepted.
- [ ] Approval workflow accepted.
- [ ] Audit log format accepted.
- [ ] Official venue/API capability review completed for any non-manual venue.
- [ ] Strategy eligibility gate accepted.
- [ ] Operator can reject, expire, and amend tickets before execution.
- [ ] No unrestricted spot-only bot continuation remains in scope.

## Initial Decision Record

| Decision | Recommended Default | Status |
| --- | --- | --- |
| Autonomy mode | `human_approved_execution` | Pending acceptance |
| Primary product universe | Regulated futures and options research | Pending acceptance |
| API canary lane | Coinbase CFM or crypto futures only after verification | Pending acceptance |
| Robinhood role | Manual derivatives ticket venue | Pending acceptance |
| Internal architecture | Broker-neutral trade idea, risk, approval, and audit records | Pending acceptance |
| Existing spot bot | Do not migrate as unrestricted autonomous spot execution | Pending acceptance |

Once this table is accepted and the trigger checklist is filled, migration work
can start with schemas and audit persistence before any execution adapter change.
