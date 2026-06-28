# Operating Rubric — v2 (draft)

---
status: draft
supersedes: OPERATING_RUBRIC.md (on acceptance)
---

> **Draft for owner review.** This restructures the v1 rubric without discarding
> its content. Every number marked **(tune)** is a proposed starting value, not a
> decision — they encode risk appetite and are yours to set. On acceptance this
> replaces `OPERATING_RUBRIC.md` and inherits its index entry. Nothing here
> authorizes live execution; the autonomy boundary stays `human_approved_execution`.

## Why v2 has this shape

The v1 rubric did three jobs at once — stated a philosophy, listed capabilities,
and tried to gate scope creep — and the three tangled. Its "evidence" column
meant three incompatible things (a file exists / a metric / demonstrated once),
so "Stage 0 complete" could be claimed because code existed, even though the
capability wasn't wired into a working loop. It also modelled features, not the
operating loop, so "the loop isn't connected to real data or real intelligence"
could not surface as a failed criterion.

v2 separates the three jobs and fixes the evidence problem:

- **Part 1 — Charter:** the north star. Philosophy + invariants. Rarely changes.
- **Part 2 — The operating loop:** the *object* the gates measure. Cadence and
  health, so "is the loop actually closed and fed" is checkable.
- **Part 3 — Graduation gates:** one consistent evidence type, with numeric
  promotion gates **and** demotion triggers per stage transition.

What is *built* (which file exists, which command works) is no longer "evidence"
here — that belongs in [Project Status](STATUS.md). Evidence in this rubric is
always a **measured outcome over a window**, never a code pointer.

---

# Part 1 — Charter

## Risk Philosophy (owner-accepted, 2026-06-11 — unchanged)

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

## Non-negotiables (invariants — true at every stage)

- Append-only audit trail for every state change, budget change, and autonomy grant.
- A reachable kill switch once anything can submit orders.
- No secrets or credentials in records, logs, or reports.
- Blast-radius caps always exist — they may be wide, but never absent.
- **Autonomy ratchets down automatically and up only by re-earning.** A breach
  suspends the current grant via an audited event (down-ladder is automatic, not
  silent); restoring it requires re-meeting the gate plus a recorded human decision.

*Architecture invariant (not a capability):* every operator action is a library
call with identity stamping; CLI / TUI / MCP are thin adapters over it. Tooling
serves owner, developing agents, and operating agents equally.

---

# Part 2 — The operating loop

The thing the gates grade is a loop, not a feature set. One turn of the loop:

```
market snapshot → propose → eligibility/policy filter → review (human or auto)
   → approve → execute (paper/live) → record fill → attribute at closeout
   → refresh track record → (feeds the next turn)
```

**Cadence (proposed — tune):**

| Step | Cadence / SLA |
|------|---------------|
| Propose | On schedule each session, or every 4h (tune) — from real market data, not fixtures |
| Review | Proposed ideas reviewed or auto-expired within the review-latency budget |
| Execute | Approved → submitted within one execution window |
| Closeout | Resolved ideas attributed within one reporting period |
| Track-record refresh | Regenerated each reporting period |

**Loop health (checkable every run — a red here blocks promotion):**

- Proposals flowing when the market is open (volume > 0).
- Review-latency p95 within the budgeted SLA (no silent backlog).
- Attribution coverage = 100% of closed ideas.
- Audit integrity verification passes.
- No step starved or backlogged.

A stage's autonomy is only valid while loop health is green.

---

# Part 3 — Graduation gates

## Evidence — one type only

> **Evidence = a demonstrated, measured outcome over a defined observation
> window, recorded in the audit/closeout trail.**

Not a file. Not "exercised once" hand-wave. Not "code compiles." If a criterion
can't be expressed as a measured outcome over a window, it isn't a gate — it's
either an invariant (Part 1) or a build task (STATUS + the issue tracker).

**Observation window (proposed — tune):** rolling 60 days *and* ≥ 200 closed
ideas, whichever is larger. Gates are evaluated over this window.

## Metric vocabulary (defined once, referenced by gates)

| Metric | Definition |
|--------|------------|
| Eligibility pass rate | Proposed ideas passing eligibility without human edit ÷ proposed |
| Attribution coverage | Closed ideas with recorded resolution + realized PnL ÷ closed |
| Risk calibration | Losing ideas whose realized loss ≤ recorded max-loss estimate ÷ losing ideas |
| Expectancy (avg R) | Mean realized return in R-multiples (realized PnL ÷ estimated max loss) |
| Benchmark edge | Expectancy(system) − Expectancy(deterministic baseline) on identical windows |
| Max drawdown-from-peak | Largest peak-to-trough equity decline over the window, % of peak |
| Track-record depth | (count of closed ideas, calendar days observed) |

## The ladder

Each transition lists **promotion gates** (all required, all measured over the
window) and inherits the **demotion triggers** below. Numbers are **(tune)**.

### Stage 0 — Rails · *complete*

Foundation, not a performance stage. Holds as long as the Part 1 non-negotiables
hold. (Per [STATUS.md](STATUS.md), the rails are shipped.)

### Stage 1 — Human-approved loop

**Entry (capability prerequisites — the loop is closed, no performance bar yet):**

- A proposer emits eligible ideas from **real market data**, not fixtures (#1031).
- Approved tickets execute in paper and fills **reconcile onto the audit trail
  automatically** (#1035).
- Every closed idea gets resolution + realized-PnL attribution.
- The expiry sweep runs on schedule.

Operate here, human-approving every idea, until the Stage 2 gate is met.

**Stage 1 → Stage 2 promotion gates** (earn auto-approval within budget):

| Gate | Threshold (tune) |
|------|------------------|
| Track-record depth | ≥ 200 closed paper ideas over ≥ 60 days |
| Eligibility pass rate | ≥ 90% |
| Attribution coverage | = 100% |
| Risk calibration | ≥ 95% (the system's loss estimates are trustworthy) |
| Skill | Expectancy > 0 **and** benchmark edge > 0 (beats the deterministic baseline, not just "doesn't lose") |
| Max drawdown-from-peak | ≤ the budget's configured peak-drawdown limit across the whole window |
| Kill-switch drill | Verified halt of propose/approve/submit (not assumed) |
| Daily-loss breaker | Demonstrated to halt submission in paper |

### Stage 2 — Bounded autonomy

Ideas inside the budget envelope auto-approve; outside it they queue for human
review; both paths audited. High-water-mark protection active. Profit-taking /
rebalancing per recorded policy. Daily-loss breaker enforced at runtime.

**Stage 2 → Stage 3 promotion gates** (earn self-direction):

| Gate | Threshold (tune) |
|------|------------------|
| Sustained bounded autonomy | ≥ 90 days with zero budget breaches that needed human rescue |
| HWM protection proven | De-risked on ≥ 1 real peak-drawdown event (paper or canary), not just configured |
| Budget renegotiation exercised | Agent proposed a budget change with rationale; human approved; new version took effect ≥ once |
| Profit-taking exercised | Skim/rebalance executed per policy, visible in audit ≥ 3 times |
| Calibration + skill sustained | At or above Stage-2-entry thresholds across the window |

### Stage 3 — Self-directed entity

Capital allocation across concurrent strategies (recorded with trade-idea rigor),
self-adjusted budgets inside an owner-set meta-envelope, scheduled self-review
leaving auditable notes, and a periodic plain-language owner report (positions,
PnL vs high-water mark, budget changes, notable decisions).

## De-escalation — the down-ladder (every stage with autonomy)

The current autonomy grant is **automatically suspended via an audited event**
when any trips over the rolling window:

| Trigger | Action |
|---------|--------|
| Drawdown-from-peak exceeds the configured peak-drawdown limit | De-risk + suspend auto-approval → back to human approval |
| Risk calibration falls below floor (realized loss exceeded estimate on > 10% of recent losers — tune) | Suspend auto-approval |
| Audit integrity check fails, or kill-switch drill fails | Immediate drop to Stage 0 (manual only) |
| Data staleness / venue degradation beyond threshold | Pause proposing + auto-approval until cleared |

Restoration is **not** automatic: re-meet the relevant promotion gate over a
fresh window, with a recorded human decision. This is the concrete form of
"earned, recorded, reversible" — and the operational expression of the
"triple-then-zero is the worst outcome" philosophy.

---

## Appendix — v1 → v2 mapping (nothing dropped)

| v1 element | Lands in v2 as |
|------------|----------------|
| Risk Philosophy (4 points) | Part 1 Charter, verbatim |
| Non-negotiables | Part 1 invariants (+ down-ladder made explicit) |
| Stage 0 evidence (file pointers) | Moved to [STATUS.md](STATUS.md); Stage 0 is "complete" there, not "evidence" here |
| Stage 1 capabilities (proposer, reviewer tooling, paper fills, attribution, expiry, report) | Stage 1 **entry prerequisites** + the operating loop (Part 2) |
| Stage 1 "≥90% eligibility" | A Stage 1→2 promotion gate |
| Stage 2 capabilities (auto-approval, HWM, profit-taking, kill switch, budget renegotiation, daily-loss breaker) | Split: Stage 2 description + Stage 1→2 and 2→3 promotion gates + down-ladder triggers |
| Stage 3 capabilities | Stage 3 description |
| "grade work against this before adding surface" intent | Realized structurally: capabilities are no longer phrased as a build-backlog |
