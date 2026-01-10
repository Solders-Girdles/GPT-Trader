# Agent Tools Reference

This guide documents all agent tools available in GPT-Trader for AI-assisted development workflows.

## Quick Reference

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `uv run agent-check` | Quality gate | Before PR/commit |
| `uv run agent-impact` | Change analysis | After making changes |
| `uv run agent-map` | Dependency graph | Architecture exploration |
| `uv run agent-tests` | Test inventory | Finding relevant tests |
| `uv run agent-risk` | Risk config query | Risk parameter lookup |
| `uv run agent-naming` | Naming check | Verifying naming standards |
| `uv run agent-regenerate` | Regenerate context | After schema changes |

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

### agent-impact (Change Impact Analysis)

Analyzes changed files and suggests relevant tests to run.

**Script**: `scripts/agents/change_impact.py`

**Usage**:
```bash
uv run agent-impact --from-git               # Analyze git changes
uv run agent-impact --from-git --base main   # Compare to main branch
uv run agent-impact --files src/gpt_trader/cli/commands/orders.py
uv run agent-impact --include-importers      # Show importing modules
uv run agent-impact --format text            # Human-readable output
```

**Output Fields**:
- `impact_level`: low/medium/high/critical
- `suggested_tests`: Specific test files to run
- `pytest_command`: Ready-to-run command
- `affected_components`: Components touched by changes

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
uv run agent-tests --stdout                   # Output to stdout
```

**Output Location**: `var/agents/testing/`

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

**Banned Patterns**: `cfg`, `svc`, `mgr`, `util`, `utils`, `qty`, `amt`, `calc`, `upd`

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
└── broker/                 # Broker API docs
```

These files are:
- Committed to the repository (except large test_inventory.json)
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
The `test_inventory.json` file is excluded from git due to size. Regenerate locally:
```bash
uv run agent-tests
```
