# Proposed Specialized Agent Workforce (Draft)

Status: Design draft. We currently use Claude built-in agents only.

Goals
- Minimize context cost by aligning agents to vertical slices
- Clear responsibilities, inputs, outputs, and stop conditions
- Safe defaults; no cross-slice coupling

Design Principles
- One agent per responsibility; compose when needed
- All agents follow slice isolation rules (see `src/bot_v2/SLICES.md`)
- Return structured results (summary, actions, file paths)

Proposed Agents (10)
- strategy-analyst: Validates strategy logic; outputs assumptions and test plan
- backtest-specialist: Designs/runs backtests; outputs metrics and confidence
- regime-analyst: Assesses regime signals; outputs regime state and risks
- sizing-advisor: Recommends position sizes; outputs sizing rationale
- data-curator: Validates data sources; outputs quality report and fixes
- test-engineer: Writes and runs tests; outputs coverage summary
- perf-analyst: Profiles hot paths; outputs minimal optimizations
- docs-editor: Updates docs; outputs diffs and navigation updates
- compliance-reviewer: Reviews for constraints; outputs findings checklist
- orchestrator-lite: Plans cross-slice tasks; outputs step plan and owners

Template (Front-matter)
---
name: <agent-name>
description: <short purpose>
tools: [read, grep, shell]
---

Usage
- Provide: slice, goal, constraints, expected outputs
- Require: exact file paths and return format

Next Steps
- Confirm this set and responsibilities
- Implement top 3 first (strategy-analyst, backtest-specialist, test-engineer)
- Add validation to `scripts/validate_claude_docs.py` for new agent files

