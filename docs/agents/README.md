# Agent Docs Index

Use this folder for AI-focused navigation aids and generated inventories.

## Core References

- Codebase map: `docs/agents/CODEBASE_MAP.md`
- Reasoning artifacts: `docs/agents/reasoning_artifacts.md`
- Glossary: `docs/agents/glossary.md`
- Naming inventory: `docs/agents/naming_inventory.md`
- Naming patterns config: `config/agents/naming_patterns.yaml`
- Document verification matrix: `docs/agents/Document_Verification_Matrix.md`
- Source → test map: `var/agents/testing/source_test_map.json`

## Tooling

- Agent tools reference: `docs/guides/agent-tools.md`
- Tooling helpers: `scripts/agents/README.md`

## Regeneration

Most inventories are generated via:

```bash
uv run agent-regenerate
uv run agent-regenerate --only testing
```
