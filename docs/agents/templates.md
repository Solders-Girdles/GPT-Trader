# Agent Operational Templates

Use these ready-to-copy examples when coordinating the AI Agent Workflow initiative. Update the tables or prompts inline as you capture project-specific data.

## Documentation Audit Matrix (Example)

| Document | Location | Owner | Last Reviewed | Status (Current/Needs Update/Missing) | Drift Notes | Action Items |
|----------|----------|-------|---------------|----------------------------------------|-------------|--------------|
| README | README.md |  |  | Current | Spot trading quick start already aligned; revalidate derivatives notes during Sprint 0. | Confirm `COINBASE_ENABLE_DERIVATIVES` guidance after interviews. |
| Architecture Overview | docs/ARCHITECTURE.md |  |  | Needs Update | Likely missing recent orchestration/risk guard tweaks. | Capture latest vertical slice layout + telemetry flow. |
| Agent Guide (Shared) | docs/agents/Agents.md |  |  | Needs Update | Update test counts and clarify spot-first focus. | Sync with agent-specific notes once review completes. |
| Agent Guide (Claude) | docs/agents/CLAUDE.md |  |  |  |  |  |
| Agent Guide (Gemini) | docs/agents/Gemini.md |  |  |  |  |  |
| Agent Dev Guide | docs/guides/agents.md |  |  |  |  |  |
| Complete Setup Guide | docs/guides/complete_setup_guide.md |  |  |  |  |  |
| Testing Guide | docs/guides/testing.md |  |  |  |  |  |
| Production Guide | docs/guides/production.md |  |  |  |  |  |
| Monitoring Guide | docs/guides/monitoring.md |  |  |  |  |  |
| AI Agent Workflow Initiative | docs/agents/ai_workflow_initiative.md | Codex |  | Planned |  | Populate once initiative kicks off. |
| Other |  |  |  |  |  |  |

**Tips**
- Use consistent status tags: `Current`, `Needs Update`, `Missing`, `Draft`.
- Capture doc/code drift succinctly and make action items immediately actionable (e.g., “Update CLI flags section with `--dry-run`).

## Interview Notes Outline

- **Status:** (scheduled / in-progress / pending summary / complete)
- **Interviewee:**
- **Role / Slice Ownership:**
- **Date:**
- **Contact Channel:**

### 1. Current Workflow
- Steps taken before handing work to an agent.
- Most-referenced documents or scripts.
- Usual agent blockers.

### 2. Pain Points & Opportunities
- Naming or terminology friction.
- Documentation gaps or inaccuracies.
- Tooling or automation requests.
- Testing or validation hurdles.
- Communication or review bottlenecks.

### 3. Desired Improvements
- Highest-impact change to land first.
- Automation ideas (preflight checks, linting, dashboards).
- Governance/reporting expectations.

### 4. Metrics & Success
- Metrics that demonstrate success.
- Desired status-update cadence.

### 5. Follow-Ups
- Immediate action items.
- Longer-term considerations.
- Additional stakeholders to interview.

### 6. Quotes / Highlights
- Notable quotes or anecdotes to share with the wider team.

Wrap with a link to the shared discovery log so others can trace context.

## Sprint Kickoff Invite Draft

> **Subject:** Sprint 0 Kickoff – AI Agent Workflow Initiative
>
> **When:** <select 30-minute slot> + optional 15-minute buffer for spillover
>
> **Attendees:** Codex (acting lead) plus any maintainers/ops partners who want to observe
>
> Hi all,
>
> I’d like to grab 30 minutes to kick off Sprint 0 for the AI Agent Workflow initiative. Agenda:
>
> 1. Revisit the mission and success metrics (see `docs/agents/ai_workflow_initiative.md#1-mission--outcomes`).
> 2. Walk through the three workstreams and note that Codex is the acting owner until volunteers step in (`docs/agents/ai_workflow_initiative.md#3-workstreams--owners`).
> 3. Review Sprint 0 deliverables and discovery templates already staged in `docs/agents/templates.md`.
> 4. Lock in cadence, status notes, and decision log location (`docs/agents/ai_workflow_initiative.md#4-execution-cadence`).
> 5. Surface questions and volunteer interest for each workstream.
>
> **Prep:** Skim the initiative doc ahead of time and bring documentation/tooling pain points you want us to prioritise.
>
> I’ll hold a 15-minute buffer after the meeting for spillover if discussions run long.
>
> Let me know if the proposed slot is a conflict or if we should invite additional stakeholders.
>
> — Codex

## Tooling Gap Backlog Seed

| ID | Title | Category (Docs/Automation/Governance) | Description | Impact (H/M/L) | Effort (S/M/L) | Owner | Sprint Target | Notes |
|----|-------|---------------------------------------|-------------|----------------|----------------|-------|---------------|-------|
| T-001 | Preflight context script | Automation | Generate repo snapshot for agents (tree, tests, TODOs). | H | M | Codex | Sprint 1 | Seed task for `scripts/agents/preflight.py`. |
| T-002 | Naming inventory generator | Automation | Surface legacy identifiers + duplicates before refactors. | H | M | Codex (acting) | Sprint 1 | Evaluate `rope`/AST-based approach. |
| T-003 | Onboarding checklist sync | Docs | Single source of truth for agent first steps. | M | S | Codex (acting) | Sprint 0 | Derive from doc audit findings. |
| T-004 | Weekly change-log automation | Governance | Summarise renames/config updates for agents. | M | M | Codex (acting) | Sprint 2 | Could extend preflight script. |
| T-005 | Metrics dashboard scaffolding | Governance | Track setup time, doc drift, test pass rate. | M | L | Codex (acting) | Sprint 2 | Coordinate with monitoring team. |
| T-006 |  |  |  |  |  |  |  |  |
| T-007 | Tests/config naming scan | Automation | Extend inventory script to cover `tests/`, `config/`, and CLI surfaces. | H | M | Codex (acting) | Sprint 0 | Feed flagged items into audit + rename backlog. |
| T-008 | Naming linter integration | Automation | Convert inventory findings into preflight/CI checks that block banned abbreviations. | H | M | Codex (acting) | Sprint 2 | Depends on T-002/T-007 outputs. |

Reuse the table to seed the canonical backlog and delete entries once work moves into the sprint board.
