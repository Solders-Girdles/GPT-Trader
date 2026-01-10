# Agent Tooling Helpers

Quick entrypoints for AI-assisted workflows.

## Makefile Shortcuts

```bash
make agent-check        # Quality gate (text output)
make agent-impact       # Change impact (from git)
make agent-map          # Dependency summary
make agent-tests        # Test inventory to stdout
make agent-risk         # Risk config docs
make agent-naming       # Naming scan
make agent-regenerate   # Regenerate var/agents context
make agent-docs-links   # Docs link audit
```

## CLI Commands

```bash
uv run agent-check --format text
uv run agent-impact --from-git --format text
uv run agent-map --format text
uv run agent-tests --stdout
uv run agent-risk --with-docs
uv run agent-naming
uv run agent-regenerate
```

## Reference Docs

- `docs/guides/agent-tools.md`
- `docs/agents/README.md`
