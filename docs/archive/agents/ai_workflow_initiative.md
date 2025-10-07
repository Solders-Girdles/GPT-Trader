# AI Agent Workflow Improvement Initiative

This document tracks the cross-agent effort to streamline the GPT-Trader environment, tooling, and governance so agents can operate with less friction and higher confidence.

## 1. Mission & Outcomes
- **Mission:** Cut the average agent setup and orientation time in half while improving naming consistency, documentation accuracy, and testing confidence.
- **Success Metrics:**
  - < 15 minutes for a new or returning agent to reach first command execution (`poetry run perps-bot run --profile dev --dev-fast`).
  - < 3 minutes to surface naming or configuration conflicts via automation.
  - < 5% doc drift incidents per sprint (tracked in weekly log).
  - 100% of refactors accompanied by updated docs and targeted tests.

## 2. Timeline (Indicative 3-Sprint Plan)
| Sprint | Duration | Primary Goals |
|--------|----------|---------------|
| Sprint 0 | 1 week | Discovery, documentation audit, establish baseline metrics, confirm naming conventions. |
| Sprint 1 | 2 weeks | Deliver automation foundations (`scripts/agents/preflight.py`, `poetry agents:*` scripts, naming inventory prototype). |
| Sprint 2 | 2 weeks | Governance + reporting: change-log automation, review rotation, metrics dashboard, initiative retro. |

## 3. Workstreams & Owners
| Workstream | Lead | Supporting Roles | Key Deliverables |
|------------|------|------------------|------------------|
| Documentation Alignment | Codex (acting) | Open (volunteers welcome) | Doc audit matrix, consolidated onboarding checklist, updated guides. |
| Automation Tooling | Codex (acting) | Open (volunteers welcome) | Preflight script, naming inventory generator, lint/test shortcuts. |
| Governance & Metrics | Codex (acting) | Open (volunteers welcome) | Weekly status notes, rename change-log, dashboard scaffolding, retro summary. |

> **Note:** Owner assignments outside of Codex should be filled in during Sprint 0 interviews.

## 4. Interim Ownership
Until human partners volunteer, Codex carries the lead role across workstreams and keeps supporting slots open for contributors to opt in during Sprint 0. Document updates will note when a human maintainer picks up a lane.

## 4. Execution Cadence
- **Kickoff:** 30-minute meeting (agenda: goals, metrics, roles, timeline) held at Sprint 0 start.
- **Standups:** Weekly synchronous check-in (15 minutes) + twice-weekly async updates in the shared channel or tracker.
- **Status Reports:** Codex drafts end-of-week notes summarizing progress, blockers, and decisions. Store new notes alongside the project docs; historical Wave 1 reports remain available via git history.
- **Decision Log:** Capture key process/tooling decisions in the docs tree; older entries live in repository history (they were previously kept under `docs/archive/wave1_status/`).

## 5. Sprint 0 Deliverables
1. Documentation audit matrix populated with status, drift notes, and proposed owners.
2. Interview notes capturing top pain points from human teammates and agent stakeholders.
3. Tooling gap backlog grouped by expected impact (High/Medium/Low).
4. Draft naming conventions outline aligned with architecture and style guides.

## 6. Dependencies & Tooling Targets
- Inventory of current automation scripts (scripts/, docs/agents, tests/).
- Access to metrics/test reports for baseline counts.
- Agreement on naming standards (will build on `black`/`ruff` conventions where possible).

## 7. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Discovery fatigue delays Sprint 1 start | Limit interviews to 30 minutes, share async questionnaires. |
| Tooling introduces regressions | Require targeted tests, dry-run mode for automation scripts, staged rollout by subsystem. |
| Doc updates lag behind automation | Lock doc updates into definition-of-done for each task and review weekly. |
| Metrics lack reliable data source | Instrument preflight script to log to `var/agents/metrics.json` (git-ignored) and summarize weekly. |

## 8. Tracking Artifacts
- `docs/archive/agents/templates.md` (audit matrix, interview notes, tooling backlog examples)
- Historical status notes (Sprint 0) — see git history for the original Wave 1 status directory.

## 9. Next Actions
- Confirm team roster and availability for Sprint 0 interviews.
- Stand up shared Kanban (or equivalent) with workstreams as swimlanes.
- Review and approve templates before use.
- Schedule kickoff meeting.
