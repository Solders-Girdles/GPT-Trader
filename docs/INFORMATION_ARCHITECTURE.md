# Information Architecture — Where Every Fact Lives

---
status: current
---

This is the map of **where each kind of project knowledge lives, and the one
rule that keeps it from sprawling**. If you are an agent or a human deciding
where to write something down, this doc is the authority. When in doubt, follow
the routing table below rather than creating a new doc.

## The one rule

> **State each fact exactly once. Everywhere else links to it.**
> Prefer **derived truth** (code, tests, generated inventories, GitHub issues)
> over **authored prose**. Author prose only for what cannot be derived:
> **decisions** and **direction**.

A doc that restates a fact owned by another source is not "extra documentation"
— it is a second copy that will drift, and drift is the thing that forces
expensive "where are we actually?" audits. If you catch yourself summarizing
the current state, the backlog, or a decision that already has a home, **link
instead of copy**.

## Three tiers, by how often a fact changes

| Tier | What it holds | Home | Kept honest by |
|------|---------------|------|----------------|
| **Durable** | Things that change rarely and must be authored: direction, decisions, architecture invariants, naming/policy, standards | `docs/` (and root `AGENTS.md`/`CLAUDE.md`) | Human review + the doc scanners |
| **Living** | Things that track reality and change often: current state, the backlog | `docs/STATUS.md` (pointers only) + **GitHub Issues** | The scanners + `agent-regenerate --verify` |
| **Ephemeral** | Per-task working artifacts: plans, regrounding audits, scout findings, scratch packets | `work/` (gitignored) and `review_artifacts/` | Discarded; never promoted into `docs/` |

The failure mode this repo kept hitting: ephemeral planning artifacts and
living-state summaries leaked into `docs/`, where they became durable-looking
but went stale. Ephemeral work goes in `work/`. Living state stays small and
points at the source.

## Where does X go?

| You want to record… | It goes in… | Never in… |
|---------------------|-------------|-----------|
| A decision — made **or** still open | `docs/decisions/<slug>.md` (status field carries the lifecycle) | STATUS, a roadmap "Needs Decision" table, a framework "Accepted Direction" section |
| What is built / current state | `docs/STATUS.md` (small, pointer-only) + the code/tests/`var/agents/**` it points to | ARCHITECTURE prose, rubric stage tables, README |
| A unit of work / something to build or fix | A **GitHub issue** (+ `triage:*` / `agent-ready` / `decision-needed` labels) | A roadmap "Ready Queue", a STATUS "spine" list |
| Where we are going / the gates to get there | `docs/DIRECTION.md` | Multiple parallel rubric/framework/readiness docs |
| How the code is structured | `docs/ARCHITECTURE.md` + generated maps in `var/agents/**` | Hand-maintained module inventories |
| API / CLI / env-var / metric inventories | Generated `var/agents/**` (run `agent-regenerate`) | Hand-written reference tables that must be kept in sync |
| How agents work / repo rules | `AGENTS.md` (canonical); `.claude/CLAUDE.md` points at it | Duplicated process prose across several docs |
| A per-task plan, audit, or scout finding | `work/` (gitignored); promote only the **decision or issue** it produces | Anywhere under `docs/` |
| Standards (naming, testing, security, TUI) | Their existing single home under `docs/` | A second "guidelines" doc |

## Anti-bloat rules

1. **State once, link elsewhere.** Restating another doc's fact is a bug. Replace
   the restatement with a link to the owner.
2. **Prefer derived over authored.** If a fact can be produced from code, tests,
   `var/agents/**`, or the issue tracker, do not also assert it in prose.
3. **Open decisions are not backlog tables.** A pending owner decision is a
   decision file with `status: proposed`, found by querying that field — not a
   "Needs Decision" list that has to be hand-pruned.
4. **Ephemeral never lands in `docs/`.** Planning and audit artifacts live in
   `work/`. What survives a task is the issue it filed or the decision it
   recorded, not the worksheet.
5. **Retire by deleting.** Git history is the archive. `docs/archive/` is banned
   (`docs_reachability_check.py` enforces this). To retire a doc, delete it and
   fix inbound links; `git log -- <path>` recovers it.
6. **No version-suffixed docs.** A `_V2` draft beside a `V1` is two homes for one
   fact. The new version either replaces the old in place, or is a
   `status: proposed` decision until adopted.

## Enforcement (the bloat guard)

These already exist and run against every markdown file under `docs/`. They are
what makes "state once, prefer derived" mechanically checkable instead of a good
intention:

| Check | Guards |
|-------|--------|
| `scripts/maintenance/docs_reachability_check.py` | Every doc is reachable from `docs/README.md` and carries a valid `status`. Orphans and stray homes fail. |
| `scripts/maintenance/docs_link_audit.py` | No dangling links or repo paths. Catches every reference broken by a retirement. |
| `scripts/maintenance/docs_currency_scan.py` | Commands, paths, env vars, and modules named in docs still exist in the code. Catches stale prose. |
| `scripts/maintenance/generate_decision_index.py` | The `docs/decisions/` index is generated from frontmatter, so it cannot drift. |
| `agent-regenerate --verify` | Generated `var/agents/**` inventories match source truth. |

## When you add or change something

- Adding a fact? Find its row in the routing table and use that home. If no row
  fits, you are probably about to create bloat — prefer an issue or a decision
  file, and only add a new durable doc as a last resort.
- Changing state? Update the one owner (usually an issue, the code, or
  `STATUS.md`), not the three places that mention it.
- Retiring a doc? Delete it, then run `docs_link_audit.py` and
  `docs_reachability_check.py` and fix what they flag.
