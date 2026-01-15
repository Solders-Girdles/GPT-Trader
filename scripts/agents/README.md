# Agent Tooling Helpers

Quick entrypoints for AI-assisted workflows.

## Setup

Install optional extras for full agent coverage (observability, live-trade, analytics).

```bash
make agent-setup
```

## Makefile Shortcuts

```bash
make agent-setup        # Install all optional extras
make agent-check        # Quality gate (text output)
make agent-impact       # Change impact (from git; file-only, no integration)
make agent-impact-full  # Change impact (from git; includes integration)
make agent-map          # Dependency summary
make agent-tests        # Test inventory to stdout
make agent-risk         # Risk config docs
make agent-naming       # Naming scan
make agent-health       # Alias for agent-health-full
make agent-health-fast  # Quick health report (skips tests; runs preflight/config)
make agent-health-fast AGENT_HEALTH_FAST_QUALITY_CHECKS=none # CI: skip lint/format/types
make agent-health-full  # Full health report (explicit envs + JSON/text output)
make agent-chaos-smoke  # Short chaos stress test (exports JSON, enforces thresholds)
make agent-chaos-week   # 7-day chaos stress test (exports JSON, enforces thresholds)
make agent-regenerate   # Regenerate var/agents context
make agent-docs-links   # Docs link audit
```

Chaos smoke defaults cap fees at 4.5% (7-day baseline ~4.34%) and drawdown at 10% to flag churn regressions
without being overly brittle. Override thresholds via the Makefile variables if needed.

## CLI Commands

```bash
uv run agent-check --format text
uv run agent-impact --from-git --format text
uv run agent-impact --from-git --source-files
uv run agent-impact --from-git --include-importers
uv run agent-impact --from-git --exclude-integration
uv run agent-map --format text
uv run agent-tests --stdout
uv run agent-tests --source gpt_trader.cli
uv run agent-tests --source gpt_trader.cli --source-files
uv run agent-risk --with-docs
uv run agent-naming
uv run agent-health
uv run agent-regenerate
uv run agent-regenerate --only testing
```

Naming defaults are loaded from `config/agents/naming_patterns.yaml`.

## Reference Docs

- `docs/guides/agent-tools.md`
- `docs/agents/README.md`
