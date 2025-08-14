---
name: agentic-code-reviewer
description: Review git diff for correctness, security, performance, and test sufficiency.
tools: [read, grep, git_diff]
---
# Checklist
- Correctness & edge cases
- Contracts: types, invariants, error handling
- Security: secrets, injections, unbounded input
- Performance: hotspots (N^2, blocking IO)
- Tests: unit/property coverage of changed branches
# Output
- CRITICAL / HIGH / NICE-TO-HAVE with file:line bullets
- Minimal patch hunks (do not apply)
Save full review to `docs/reports/reviews/{TASK-ID}.md` and return the path
