# Operating Rubric

---
status: current
last-updated: 2026-06-11
---

This rubric defines what the bot must be able to do to "run successfully" as an
autonomous trading entity, and what evidence earns each capability more
autonomy. It complements the
[Pre-Migration Decision Framework](PRE_MIGRATION_DECISION_FRAMEWORK.md) (which
gates *whether* to build execution paths) by describing *what good operation
looks like* once they exist.

Grade work against this rubric before adding new feature surface: if a change
does not advance a listed capability or its evidence, it is probably scope
creep.

## Risk Philosophy (owner-accepted, 2026-06-11)

1. **Principal is fully at risk.** The owner funds the bot with a lump sum he
   is prepared to lose entirely. Loss of principal is an acceptable outcome;
   timidity that prevents the system from ever demonstrating skill is the
   bigger failure. Seeded budgets are therefore aggressive.
2. **Realized gains are not principal.** Tripling the portfolio and then
   riding it back to zero is explicitly the *worst* outcome — worse than
   losing the original stake outright. Drawdown-from-peak matters more than
   drawdown-from-principal. High-water-mark tracking, profit-taking, and
   rebalancing are first-class capabilities, not enhancements.
3. **Limits exist to bound system failure, not to second-guess the agent.**
   Budget levers cap the blast radius of bugs, bad data, venue outages, and
   compromised credentials. They are designed to be *renegotiated by the
   agents themselves* through the audited budget workflow — handed over
   progressively as track record accumulates, never silently removed.
4. **Every grant of autonomy is earned, recorded, and reversible.** Autonomy
   expansions ride the same append-only audit trail as trades.

## Capability Rubric

Each capability lists the evidence ("graduation evidence") that it works.
A stage is complete when every capability in it has its evidence.

### Stage 0 — Rails (in progress)

| Capability | Evidence |
|------------|----------|
| Broker-neutral trade-idea record with thesis, invalidation, max loss, sizing, expiry | Unit-tested round-trip + record hashing (`features/trade_ideas/models.py`) |
| Workflow where execution is impossible without approval | State-machine tests prove `submitted` is reachable only from `approved` |
| Append-only audit trail pinning every action to a record version | Audit log rejects out-of-order or conflicting appends |
| Eligibility gate encoding automatic rejection conditions | Ineligible ideas cannot be approved |
| Versioned risk budget that agents can later renegotiate | Budget changes append to their own audited log; current budget is always derivable |
| Policy module that encodes the autonomy mode | Approval by a non-human actor is refused in `human_approved_execution` mode |

### Stage 1 — Human-approved loop

| Capability | Evidence |
|------------|----------|
| A proposer that generates complete, eligible trade ideas from market data | ≥ 90% of proposed ideas pass the eligibility gate without human edits |
| Reviewer tooling (CLI first; TUI screen optional) | Owner can list, inspect, approve, and reject ideas end to end |
| Approved tickets executed in paper mode | Paper fills recorded back onto the idea's audit trail |
| Outcome attribution | Every closed idea records whether thesis, invalidation, or expiry resolved it, and realized PnL vs. max-loss estimate |
| Expiry sweep | Stale ideas expire automatically; nothing waits on a human to notice |
| Track-record report | One command summarizes proposal quality, approval rate, and outcome accuracy over time |

### Stage 2 — Bounded autonomy

| Capability | Evidence |
|------------|----------|
| Auto-approval inside the budget envelope | Ideas within budget auto-approve; outside it they queue for human review; both paths audited |
| High-water-mark protection | Equity peak is tracked; a configured share of peak gains is defended by de-risking or halting (the "triple then zero" guard) |
| Profit-taking and rebalancing policies | Gains are systematically skimmed/rebalanced per recorded policy, visible in the audit trail |
| Kill switch | A single owner action halts proposing, approving, and submitting; verified by drill, not assumption |
| Budget renegotiation by agents | An agent proposes a budget change with rationale; human approves; new version takes effect — exercised at least once |
| Daily-loss circuit breaker enforced at runtime | Breach halts submission the same day, demonstrated in paper or canary |

### Stage 3 — Self-directed entity

| Capability | Evidence |
|------------|----------|
| Capital allocation across multiple concurrent strategies | Allocation decisions are recorded with the same rigor as trade ideas |
| Self-adjusted budgets within a meta-envelope | Agent-initiated budget changes auto-approve inside owner-set meta-limits |
| Scheduled self-review | The system periodically evaluates its own track record and adjusts behavior, leaving an auditable review note |
| Owner reporting cadence | Owner receives a periodic plain-language report: positions, PnL vs. high-water mark, budget changes, notable decisions |

## Non-negotiables (every stage)

- Append-only audit trail for every state change, budget change, and autonomy grant.
- A reachable kill switch once anything can submit orders.
- No secrets or credentials in records, logs, or reports.
- Blast-radius caps always exist — they may be wide, but never absent.
- Tooling serves three audiences equally: the owner, agents developing the
  project, and agents operating it. Every operator action must be available
  as a library call with identity stamping; CLIs/TUIs/MCP servers are thin
  adapters over that.
