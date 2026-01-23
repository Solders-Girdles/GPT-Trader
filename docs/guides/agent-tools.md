# Agent Tools Reference

---
status: current
last-updated: 2026-01-23
---

This guide documents all agent tools available in GPT-Trader for AI-assisted development workflows.

## Quick Reference

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `uv run agent-check` | Quality gate | Before PR/commit |
| `uv run agent-health` | Health report | Before sprint/CI baselines |
| `uv run agent-impact` | Change analysis | After making changes |
| `uv run agent-map` | Dependency graph | Architecture exploration |
| `uv run agent-tests` | Test inventory | Finding relevant tests |
| `uv run agent-risk` | Risk config query | Risk parameter lookup |
| `uv run agent-naming` | Naming check | Verifying naming standards |
| `uv run agent-regenerate` | Regenerate context | After schema changes |
| `make agent-chaos-smoke` | Chaos smoke test | Quick robustness signal (drawdown/fees thresholds) |
| `make agent-chaos-week` | Chaos week test | 7-day robustness signal (drawdown/fees thresholds) |

---

## Agent Environment Setup

Install optional extras so agent tooling can exercise observability, live-trade, and analytics paths.

```bash
make agent-setup
# or
uv sync --all-extras
```

If you only need observability coverage (tracing tests), install that extra:

```bash
uv sync --extra observability
```

---

## Makefile Shortcuts

Use these for quick local runs:

```bash
make agent-check
make agent-health
make agent-health-fast
make agent-health-full
make agent-chaos-smoke
make agent-chaos-week
make agent-impact
make agent-impact-full
make agent-map
make agent-tests
make agent-risk
make agent-naming
make agent-regenerate
make agent-docs-links
make scaffold-slice name=<slice> flags="--with-tests --with-readme"
```

Note: `make agent-impact` defaults to `--include-importers --source-files --exclude-integration`.
Use `make agent-impact-full` to include integration tests.
Chaos smoke defaults cap fees at 4.5% (based on a 7-day baseline of ~4.34%) and drawdown at 10% to keep
headroom while still flagging regression-level churn. Override with Makefile vars if needed.

See `scripts/agents/README.md` for the full list of helpers.

## Feature Slice Scaffold

Bootstrap new vertical slices for onboarding and growth workflows.

**Script**: `scripts/maintenance/feature_slice_scaffold.py`

**Usage**:
```bash
uv run python scripts/maintenance/feature_slice_scaffold.py --name foo --with-tests --with-readme
uv run python scripts/maintenance/feature_slice_scaffold.py --name foo --dry-run
```

Outputs land in `src/gpt_trader/features/<slice>/` with optional tests under
`tests/unit/gpt_trader/features/<slice>/`. The scaffold tool refuses overwrites.

---

## Detailed Tool Documentation

### agent-check (Quality Gate)

Runs all quality checks with machine-readable output.

**Script**: `scripts/agents/quality_gate.py`

**Usage**:
```bash
uv run agent-check                           # All checks, JSON output
uv run agent-check --format text             # Human-readable output
uv run agent-check --check lint,types        # Specific checks only
uv run agent-check --files src/gpt_trader/cli/  # Check specific paths
uv run agent-check --full                    # Include slow tests
```

**Available Checks**: `lint`, `format`, `types`, `tests`, `security`

**Output Format** (JSON):
```json
{
  "success": true,
  "total_duration_seconds": 45.2,
  "total_findings": 0,
  "checks_run": 4,
  "results": [...]
}
```

---

### agent-health (Health Report)

Aggregates lint/format/types plus optional tests, preflight, and config validation.

**Script**: `scripts/agents/health_report.py`

**Usage**:
```bash
uv run agent-health
uv run agent-health --pytest-args -q tests/unit
uv run agent-health --ci-input var/results/pytest_report.json
uv run agent-health --ci-input var/results/pytest_report.json --ci-input var/results/junit.xml
uv run agent-health --format json --output var/agents/health/health_report.json --text-output var/agents/health/health_report.txt
uv run agent-health --skip-preflight --skip-config
```

Fast pre-check (explicit envs, no tests; runs preflight/config):

```bash
make agent-health-fast
```

CI note (skip lint/format/types when already covered by CI jobs):

```bash
make agent-health-fast AGENT_HEALTH_FAST_QUALITY_CHECKS=none
```

Full baseline (explicit envs, tests + preflight + config, JSON/text output):

```bash
make agent-health-full
# or
make agent-health
```

**Schema**: `var/agents/health/agent_health_schema.json`

**Report Fields**: `schema_version`, `tool`, `ci_inputs`, `test_summary`

---

### agent-impact (Change Impact Analysis)

Analyzes changed files and suggests relevant tests to run.

**Script**: `scripts/agents/change_impact.py`

**Usage**:
```bash
uv run agent-impact --from-git               # Analyze git changes
uv run agent-impact --from-git --base main   # Compare to main branch
uv run agent-impact --files src/gpt_trader/cli/commands/orders.py
uv run agent-impact --include-importers      # Show importing modules
uv run agent-impact --source-files           # Prefer file-only test suggestions
uv run agent-impact --exclude-integration    # Drop integration tests
uv run agent-impact --format text            # Human-readable output
```

**Output Fields**:
- `impact_level`: low/medium/high/critical
- `suggested_tests`: Specific test files to run
- `pytest_command`: Ready-to-run command
- `affected_components`: Components touched by changes

If `var/agents/testing/source_test_map.json` is present, suggestions are augmented
using import-based source/test mapping.
Use `--source-files` to keep the recommended command to test file paths only.
`--include-importers` now also expands test suggestions with importer modules.
`--exclude-integration` removes tests under `tests/integration`.

---

### agent-map (Dependency Graph)

Builds and queries module dependency relationships.

**Script**: `scripts/agents/dependency_graph.py`

**Usage**:
```bash
uv run agent-map                              # Full graph JSON
uv run agent-map --format text                # Summary view
uv run agent-map --format dot                 # GraphViz output
uv run agent-map --dependencies-of gpt_trader.cli
uv run agent-map --depends-on gpt_trader.errors
uv run agent-map --check-circular             # Find circular imports
uv run agent-map --component-summary          # High-level view
```

---

### agent-tests (Test Inventory)

Generates comprehensive test inventory with marker and path filtering.

**Script**: `scripts/agents/generate_test_inventory.py`

**Usage**:
```bash
uv run agent-tests                            # Generate full inventory
uv run agent-tests --by-marker risk           # Tests with risk marker
uv run agent-tests --by-path tests/unit/gpt_trader/cli
uv run agent-tests --source gpt_trader.cli
uv run agent-tests --source gpt_trader.cli --source-files
uv run agent-tests --stdout                   # Output to stdout
```

`--source` matches tests that import the requested `gpt_trader` module (file-level scan).
Use `--source-files` to output test file paths only.

**Output Location**: `var/agents/testing/`
**Source/Test Map**: `var/agents/testing/source_test_map.json`

---

### agent-risk (Risk Configuration Query)

Queries risk configuration values with documentation.

**Script**: `scripts/agents/query_risk_config.py`

**Usage**:
```bash
uv run agent-risk                             # Full config JSON
uv run agent-risk --with-docs                 # Include field docs
uv run agent-risk --field max_leverage        # Query single field
uv run agent-risk --generate-schema           # JSON Schema output
uv run agent-risk --profile canary            # Profile-specific config
```

---

### agent-naming (Naming Standards)

Scans for naming convention violations.

**Script**: `scripts/agents/naming_inventory.py`

**Usage**:
```bash
uv run agent-naming                           # Full scan
uv run agent-naming --strict                  # Fail on violations
uv run agent-naming --quiet                   # Suppress stdout
```

Defaults (patterns, scan paths, skip token) load from `config/agents/naming_patterns.yaml`.
Use `--patterns` and `--paths` to override per run.

**Banned Patterns**: configured in `config/agents/naming_patterns.yaml` (defaults: `cfg`, `svc`, `mgr`, `util`, `utils`, `amt`, `calc`, `upd`)

**Suppression**: Add `# naming: allow` to suppress specific lines.

---

### agent-regenerate (Regenerate Context Files)

Regenerates all static context files in `var/agents/`.

**Script**: `scripts/agents/regenerate_all.py`

**Usage**:
```bash
uv run agent-regenerate                       # Regenerate all
uv run agent-regenerate --verify              # Check freshness only
uv run agent-regenerate --list                # List generators
uv run agent-regenerate --only testing        # Run a subset by key
```

**Generators**:

| Generator | Output Directory |
|-----------|-----------------|
| `generate_config_schemas.py` | `var/agents/schemas/` |
| `export_model_schemas.py` | `var/agents/models/` |
| `generate_event_catalog.py` | `var/agents/logging/` |
| `generate_test_inventory.py` | `var/agents/testing/` |
| `generate_validator_registry.py` | `var/agents/validation/` |
| `generate_broker_api_docs.py` | `var/agents/broker/` |
| `generate_reasoning_artifacts.py` | `var/agents/reasoning/` |
| `generate_agent_health_schema.py` | `var/agents/health/` |

`--only` keys match the output subdirectory names: `schemas`, `models`, `logging`,
`testing`, `validation`, `broker`, `reasoning`, `health`.

---

## Static Context Files

Pre-generated context files are committed to `var/agents/` for AI consumption:

```
var/agents/
├── index.json              # Master index
├── schemas/                # Configuration schemas
├── models/                 # Domain model schemas
├── logging/                # Event catalog
├── testing/                # Test markers and index
├── validation/             # Validator registry
├── broker/                 # Broker API docs
├── reasoning/              # CLI flow + guard/execution + market data + backtesting + reporting + entrypoints + validation/chaos + config linkage
└── health/                 # Agent health schema/example
```

These files are:
- Committed to the repository (including test_inventory.json)
- Regenerated with `uv run agent-regenerate`
- Used by AI agents for codebase understanding

---

## Integration with Development Workflow

### Pre-commit
- `agent-naming --strict` runs automatically via pre-commit hook

### CI Pipeline
- Context file freshness verified on every PR

### Claude Commands
- `/quality` - Runs `agent-check --format text`
- `/naming` - Runs `agent-naming`

---

## Troubleshooting

### Tool Not Found
```bash
# Ensure package is installed
uv sync
```

### Outdated Context Files
```bash
uv run agent-regenerate
git add var/agents/
git commit -m "chore: regenerate agent context files"
```

### Large Test Inventory
`test_inventory.json` is committed (and excluded from the large-file hook). Regenerate as needed:
```bash
uv run agent-regenerate --only testing
# or
uv run agent-tests
```
