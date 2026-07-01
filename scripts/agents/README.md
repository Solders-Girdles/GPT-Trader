# Agent Tooling Helpers

Quick entrypoints for AI-assisted workflows.

## Setup

Install optional extras for full agent coverage (observability, live-trade, analytics).

```bash
make agent-setup
```

## Canonical CLI Commands

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
uv run agent-naming
uv run agent-naming --strict --quiet
uv run agent-health
uv run agent-regenerate
uv run agent-regenerate --only testing
uv run agent-artifacts validate
uv run agent-artifacts package
uv run agent-artifacts verify-package
uv run agent-pr-ready --format markdown
```

## Additive Makefile Shortcuts

```bash
make agent-setup        # Install all optional extras
make agent-impact       # Change impact with the default local-review flags
make agent-impact-full  # Change impact with importers and integration included
make agent-health-fast  # Quick health report with default env and output paths
make agent-health-fast AGENT_HEALTH_FAST_QUALITY_CHECKS=none # CI: skip lint/format/types
make agent-health-full  # Full health report with default env and output paths
make agent-chaos-smoke  # Short chaos stress test with default thresholds/output
make agent-chaos-week   # 7-day chaos stress test with default thresholds/output
make agent-docs-links   # Docs link and reachability audits
```

Chaos smoke defaults cap fees at 4.5% (7-day baseline ~4.34%) and drawdown at 10% to flag churn regressions
without being overly brittle. Override thresholds via the Makefile variables if needed.

`agent-naming` dispatches to `scripts/agents/naming_inventory.py`; defaults are
loaded from `config/agents/naming_patterns.yaml`. Strict naming enforcement is
wired through the local pre-commit `naming-check` hook, and the current GitHub CI
workflow does not run a direct naming scan step.

## Artifact Publication

`.github/workflows/agent-artifacts-refresh.yml` is the scheduled publication
workflow for generated agent context. It regenerates `var/agents/**`, validates
the generated tree against `var/agents/index.json`, packages the tree as
`agent-artifacts.tar.gz`, uploads the package and manifest, verifies the
downloaded package, and publishes changed files to
`automation/agent-artifacts-refresh`.

The workflow does not require GitHub Actions to create pull requests. If the
repository setting blocks Actions-created PRs, the workflow still succeeds and
leaves the validated package plus the updated automation branch as the
publication outputs.

## Reference Docs

- `AGENTS.md`
- `docs/agents/README.md`
