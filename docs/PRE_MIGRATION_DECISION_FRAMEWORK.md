# Pre-Migration Decision Framework

---
status: current
last-updated: 2026-06-24
scope: AI-assisted trade research and execution migration
---

## Accepted Direction (2026-06-11)

The project owner accepted the following direction. Where this section conflicts
with the exploratory tables below, this section wins.

- **Destination:** an autonomous trading entity — a bot that observes markets,
  does its own research, and manages funds inside machine-enforced limits
  (`bounded_autonomy`).
- **Path:** the current approval-gated execution phase
  (`human_approved_execution` compatibility label) is the required validation
  phase, not the end state. The AI side must first produce complete trade-idea
  records (thesis, entry, invalidation, max loss, sizing) that receive an
  explicit approval decision. Autonomy is granted per strategy envelope only
  after the approval-phase track record, risk budgets, kill switches, and audit
  logs exist.
- **Scope:** Coinbase only (spot + CFM futures). Options, Robinhood, and other
  venues are out of scope until the Coinbase lane works end to end.
- **INTX perpetuals:** frozen — no new work or tests; remove INTX-only surfaces
  opportunistically when they block other work.
- **Existing TA bot:** remains a gated implementation asset; do not expand its
  autonomous execution surface ahead of the ladder above.

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

The v1 migration target should be approval-gated execution, currently named
`human_approved_execution` in runtime and historical docs.

AI may produce complete, broker-ready trade tickets, but an explicit approval
decision changes, rejects, or releases them for manual or scoped API execution.
This preserves a clear decision point while creating the artifacts needed for
future automation: thesis records, risk checks, approval history, and audit
logs.

`bounded_autonomy` should be treated as a later milestone. It needs
machine-enforced strategy envelopes, numeric risk budgets, kill-switch coverage,
and proven auditability before any order can be submitted without an explicit
approval decision.

## Decision Packet Alignment

`decision-needed` is the pipeline route for unsettled policy, live-operation,
broker/API, venue, account, or AI-assisted trading choices. It is not a
terminal human-only stop. A decision packet may approve docs, code, tests,
mock-only controls, or a scoped runbook when the evidence supports that route.
In the current `human_approved_execution` phase, it may recommend or prepare a
trade-ticket approval path, but it does not replace the human approval event
required by runtime policy.

That packet does not by itself authorize real broker/API calls, live trading
commands, production preflight, canary operations, money movement, or order
submission. Those command lanes require the packet or runbook to explicitly
name the lane, constraints, verification boundary, and rollback/kill-switch
expectations.

## Decision 1: Autonomy Boundary

| Mode | AI May Produce | AI May Submit Orders | v1 Status |
| --- | --- | --- | --- |
| `research_only` | Research notes, theses, watchlists | No | Allowed, but too weak as the main target |
| `human_approved_execution` | Broker-ready tickets and risk records | No, explicit approval required first | Recommended default compatibility label |
| `bounded_autonomy` | Tickets inside pre-approved envelopes | Yes, only inside hard limits | Future milestone |

Hard v1 rule: AI does not submit live orders. Any API execution lane must
require an explicit approval event first.

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
| Coinbase CFM futures research | Current Coinbase-only lane | Best fit for explicit invalidation, max loss, and multi-hour/multi-day theses; API execution requires account, product, metadata, and risk-gate verification |
| Coinbase spot research | Current Coinbase-only lane | Keep spot work research/approval-gated; do not migrate unrestricted spot automation into the new shape |
| Options and manual derivatives | Future lane | Out of v1 scope until a later decision packet reopens the lane with product, venue, API/manual, risk, and account-capability evidence |
| Non-Coinbase crypto futures or alternate API venues | Future lane | Requires a fresh venue/API/account capability review and decision packet before becoming a migration or execution target |
| Event contracts | Future lane | Needs separate venue, liquidity, fee, compliance, and account-capability review |
| Mixed derivatives workbench | Later architecture target | Promote only after individual product lanes have accepted schemas, risk gates, and execution boundaries |

The first migrated system should optimize for Coinbase spot plus Coinbase CFM
futures research. Non-Coinbase venues, options or other manual derivatives, and
alternate API lanes require a later decision packet or fresh capability review
before they become active product scope.

## Decision 4: Broker Role

Brokers are venues, not the architecture.

| Venue | Role | Boundary |
| --- | --- | --- |
| Internal record | Canonical thesis, risk, approval, and audit state | Must be broker neutral |
| Coinbase | API-capable crypto and CFM lane | Only for products the repo and account can verify |
| Robinhood | Future/out-of-scope venue | Requires a later decision packet and official API/manual capability review before any migration work |
| Future adapters | Future optional execution paths | Must adapt to the internal record, not reshape it; require a fresh decision packet and venue/API/account capability review |

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
2. A human reviewer checks strategy eligibility, data freshness, risk, and
   do-not-trade conditions.
3. The human reviewer either rejects the idea, requests changes, or approves it.
4. Approved ideas produce a broker-specific ticket.
5. Execution is manual or explicitly triggered by the human approval event, and
   any API command lane must also be scoped by a decision packet or approved
   runbook.
6. The result is appended to the audit log.

Allowed states:

| State | Meaning |
| --- | --- |
| `proposed` | AI generated the record; no execution allowed |
| `needs_changes` | Human reviewer requested edits or missing evidence |
| `rejected` | Human reviewer rejected the idea; terminal |
| `approved` | Human approval decision accepted the ticket; execution may proceed by policy |
| `submitted` | Order was manually entered or API-submitted after approval |
| `filled` | Venue confirmed fill |
| `cancelled` | Reviewer, operator, system, or venue cancelled before fill |
| `expired` | Idea passed its review or execution deadline |

Approval must be append-only. Do not overwrite the original thesis when a
human reviewer changes the ticket; append a new event that explains the change.

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

Do not migrate the existing bot into this shape until every row has no pending
migration trigger. `Accepted` and `Implemented` rows are current evidence only;
they do not authorize real broker/API calls, live trading commands, production
preflight, money movement, or order submission.

| Trigger | Current Status | Evidence | Still Pending Before Migration |
| --- | --- | --- | --- |
| Autonomy mode selected | Accepted; current phase implemented | 2026-06-11 direction selected `human_approved_execution` as the validation phase and `bounded_autonomy` as the destination; `src/gpt_trader/features/trade_ideas/policy.py` enforces human approval in the current mode. | Any move to `bounded_autonomy` still needs a fresh decision packet plus strategy-envelope, kill-switch, and audit evidence. |
| First asset and product universe selected | Accepted for current research scope | Coinbase spot plus CFM futures research are the accepted Coinbase-only lane. | Account, product, and API capability must be verified before any non-manual execution lane. |
| Broker execution policy selected | Accepted for current phase; implemented in trade-idea rails | Broker-neutral records remain canonical; `src/gpt_trader/features/trade_ideas/workflow.py` and `src/gpt_trader/features/trade_ideas/service.py` only allow submission records after approval. | Any actual broker command lane still needs an explicit decision packet or approved runbook naming constraints and rollback/kill-switch expectations. |
| Numeric risk budget selected | Accepted and implemented as the Stage 0 seed | `src/gpt_trader/features/trade_ideas/budget.py` seeds versioned risk budgets from the accepted rubric; `src/gpt_trader/features/trade_ideas/policy.py` checks approvals against the current budget. | Future strategy-envelope budgets and agent-renegotiated budget changes must stay audited and reviewable. |
| Required output schema accepted | Implemented as typed broker-neutral record | `src/gpt_trader/features/trade_ideas/models.py` implements the trade-idea record fields, record hashing, and broker-neutral ticket envelope. | Schema publication or adapter-specific payload derivation remains separate migration work. |
| Approval workflow accepted | Implemented in trade-idea rails | `src/gpt_trader/features/trade_ideas/workflow.py` defines the state machine; `src/gpt_trader/features/trade_ideas/service.py` applies policy before approval and submission records. | Execution adapters must continue to consume approved records without bypassing the workflow. |
| Audit log format accepted | Implemented in trade-idea rails | `src/gpt_trader/features/trade_ideas/audit.py` implements append-only JSONL events pinned to record hashes and verifies per-decision sequencing. | Venue-specific external order identifiers can be recorded only after an approved submission lane exists. |
| Official venue/API capability review completed for any non-manual venue | Pending | None in this framework authorizes account, venue, product, or API capability. | Complete a fresh venue/API/account capability review before non-manual Coinbase or other venue execution. |
| Strategy eligibility gate accepted | Implemented in trade-idea rails | `src/gpt_trader/features/trade_ideas/eligibility.py` encodes automatic rejection conditions; `src/gpt_trader/features/trade_ideas/policy.py` blocks approval when eligibility or budget checks fail. | New product-specific eligibility rules need explicit evidence before expansion. |
| Operator can reject, expire, and amend tickets before execution | Implemented in trade-idea rails | `src/gpt_trader/features/trade_ideas/service.py` supports request-changes/resubmit, reject, cancel, expire, and expire-due lifecycle paths before submission. | Any new operator surface must remain a thin adapter over the audited service path. |
| No unrestricted spot-only bot continuation remains in scope | Accepted boundary | 2026-06-11 direction rejects unrestricted autonomous spot continuation. | Future spot work must stay research/approval-gated unless a new decision packet changes the scope. |

## Initial Decision Record

| Decision | Accepted Value | Status |
| --- | --- | --- |
| Autonomy mode | Approval-gated execution (`human_approved_execution` compatibility label) now; `bounded_autonomy` is the accepted destination | Accepted 2026-06-11 |
| Primary product universe | Coinbase spot + CFM futures research | Accepted 2026-06-11 |
| API canary lane | Coinbase CFM only after account/product verification | Accepted 2026-06-11 |
| Robinhood role | Out of scope | Accepted 2026-06-11 |
| Internal architecture | Broker-neutral trade idea, risk, approval, and audit records | Accepted 2026-06-11 |
| Existing spot bot | Do not migrate as unrestricted autonomous spot execution | Accepted 2026-06-11 |

The durable project log mirrors these accepted decisions in
[GPT-Trader Decision Log](GPT_TRADER_DECISION_LOG.md). The implemented Stage 0
trade-idea rails are evidence for the accepted record, approval, audit,
budget, eligibility, and operator-control triggers above; they are not evidence
that venue/account/API capability has been verified.

Once this table is accepted and the trigger checklist is filled, migration work
can start with schemas and audit persistence before any execution adapter change.
