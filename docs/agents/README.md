# Agent Docs Index

---
status: current
last-updated: 2026-01-23
---

Use this folder for AI-focused navigation aids and generated inventories.

## Core References

- [Codebase map](CODEBASE_MAP.md)
- [Reasoning artifacts](reasoning_artifacts.md)
- [Glossary](glossary.md)
- [Environment variables](../../var/agents/configuration/environment_variables.md)
- [Metrics catalog](../../var/agents/observability/metrics_catalog.md)
- [Naming patterns config](../../config/agents/naming_patterns.yaml)
- [Naming scan tool](../../scripts/agents/naming_inventory.py)
- [Document verification matrix](Document_Verification_Matrix.md)
- [Source → test map](../../var/agents/testing/source_test_map.json)

## Tooling

- [Agent tools reference](../guides/agent-tools.md)
- [Tooling helpers](../../scripts/agents/README.md)

## Regeneration

Most inventories are generated via:

```bash
uv run agent-regenerate
uv run agent-regenerate --only testing
```
