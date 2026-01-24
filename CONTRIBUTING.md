# Contributing to GPT-Trader

We use a set of automated tools to ensure code quality and consistency. Please follow these steps for your development environment to ensure your contributions pass our automated checks.

## One-Time Setup

### 1. Install `pre-commit`

We use `pre-commit` to run checks before you commit your code. This helps catch issues early. You can install it using `pipx` (recommended) or `pip`.

**With `pipx` (Recommended):**
```bash
pipx install pre-commit
```

**With `pip`:**
```bash
pip install pre-commit
```

### 2. Install the Git Hooks

After installing `pre-commit`, you need to install the hooks into your local git repository.

```bash
pre-commit install
```

That's it! Now, every time you run `git commit`, the pre-commit hooks will run automatically. If they find any issues (like formatting errors), they may fix the files and abort the commit. In that case, just `git add` the modified files and run `git commit` again.

## Testing Requirements

### Current Expectations
- Keep the full test suite green.
- Add focused tests for every new feature or regression fix.
- Document any skips or deselections tied to legacy code paths.

### Pre-PR Verification Checklist
1. Review `docs/README.md` and prefer code + `var/agents/**` generated inventories for anything that drifts.
2. Refresh dependencies: `uv sync`.
3. Snapshot test discovery: `uv run pytest --collect-only`.
4. Run the full unit suite: `uv run pytest tests/unit -q`.
5. Execute slice-specific suites relevant to your change set (examples below).
6. Validate docs links: `make agent-docs-links` (runs `scripts/maintenance/docs_link_audit.py`).

### Recommended Commands

```bash
# Core suites
uv run pytest tests/unit/gpt_trader -q
uv run pytest tests/unit/gpt_trader/features/brokerages/coinbase -q
uv run pytest tests/unit/gpt_trader/features/live_trade -q

# Coverage snapshot
uv run pytest --cov=gpt_trader --cov-report=term-missing
```

### Test Metrics
- **Coverage Goal**: Maintain ~73% overall, >90% on new code paths
- **Integration Paths**: Coordinate with maintainers before toggling derivatives gates

## Running the Bot Locally

To run the spot trading bot for development, use the `gpt-trader` command (derivatives stay gated behind INTX + `COINBASE_ENABLE_INTX_PERPS=1`):

```bash
uv run gpt-trader run --profile dev --dev-fast
```

Launch the TUI via the CLI entry point to keep uv-locked deps and env/logging setup consistent:

```bash
uv run gpt-trader tui               # Mode selector
uv run gpt-trader tui --mode demo   # Skip selector for demos
```

## Development Workflow

Before branching, make sure to:
- Review `docs/README.md` for the current doc index (and verify key claims in code).
- Sync with the latest `main`.
- Run `uv sync` to pick up dependency changes.

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Write tests** for your changes
   - Unit tests required for all new functions
   - Integration tests for API interactions
   - Must maintain 100% pass rate on active tests
4. **Run the test suite** to ensure nothing is broken
   - `uv run pytest --collect-only` to verify test discovery
   - `uv run pytest tests/unit/gpt_trader -q` must pass
   - No new test failures allowed
5. **Follow repository organization standards**
   - Place files in correct directories (see Repository Organization below)
   - Update documentation using consolidated structure
   - Add new documentation to appropriate `/docs` subdirectories
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to your fork** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

## Quality Standards

### Code Quality
- Clean, readable code with meaningful variable names
- Adhere to [Naming Standards](docs/naming.md)
- Comprehensive docstrings for public functions
- Type hints where beneficial
- Maximum line length: 100 characters

### Test Quality
- Descriptive test names that explain what's being tested
- One assertion per test when possible
- Use fixtures for shared setup
- Use pytest `monkeypatch` for mocks; patch-style helpers are blocked in `tests/`
- Keep test modules <= 240 lines unless allowlisted by `scripts/ci/check_test_hygiene.py`
- Avoid `time.sleep`; prefer deterministic `fake_clock`
- Match marker conventions to folder (integration/contract/real_api)

### Resilience Testing

The `tests/fixtures/failure_injection.py` module provides deterministic failure simulation for testing retry logic, degradation behavior, and error handling without network calls or sleeps.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `FailureScript` | Scripted sequence of failures/successes |
| `InjectingBroker` | Wraps a broker to inject failures per-method |
| `no_op_sleep` | Instant sleep for deterministic timing |
| `counting_sleep` | Records sleep durations for backoff verification |

**Example Usage:**

```python
from tests.fixtures.failure_injection import FailureScript, InjectingBroker, counting_sleep

# Fail twice, then succeed
script = FailureScript.fail_then_succeed(failures=2)
injecting = InjectingBroker(mock_broker, place_order=script)

# Verify exponential backoff
sleep_fn, get_sleeps = counting_sleep()
executor = BrokerExecutor(broker=injecting, sleep_fn=sleep_fn)
executor.execute(order)
assert get_sleeps() == [0.5, 1.0]  # Exponential delays
```

**Running Resilience Tests:**

```bash
# Broker executor resilience tests
uv run pytest tests/unit/gpt_trader/features/live_trade/execution/test_broker_executor_resilience_*.py -v

# Degradation recovery tests
uv run pytest tests/unit/gpt_trader/features/live_trade/test_degradation.py::TestPauseExpiryRecovery -v

# Order submission idempotency test
uv run pytest tests/unit/gpt_trader/features/live_trade/execution/test_order_submission_flows.py::TestTransientFailureWithClientOrderIdReuse -v
```

All resilience tests run deterministically under `pytest -n auto`.

## Repository Organization

### Directory Structure Standards

The repository follows a standardized organization optimized for both human developers and AI agents:

#### Source Code & Configuration
- `/src/gpt_trader/` - Active trading system (vertical slice architecture)
- `/tests/` - Test files organized by component
- `/config/` - Configuration files, trading profiles, and templates
- `/scripts/` - Operational scripts organized by domain:
  - `agents/` - Automation helpers and inventory tooling
  - `analysis/` - Dependency and repository analysis utilities
  - `ci/` - Validation and guardrail checks
  - `maintenance/` - Cleanup and maintenance tasks
  - `monitoring/` - Monitoring and dashboards
  - Root `scripts/*.py` - Core runbooks and one-off utilities

#### Documentation Standards
- `/docs/guides/` - How-to guides and tutorials
- `/docs/reference/` - Technical reference documentation
- `/docs/operations/` - Operations and maintenance procedures

#### Archive Management
- Use git history for retired docs or scripts
- Record removals in `docs/DEPRECATIONS.md`

### File Placement Guidelines

#### New Documentation
- **Tutorials/Guides**: `/docs/guides/`
- **API Reference**: `/docs/reference/`
- **Operations**: `/docs/operations/`
- **Never**: Root directory or legacy locations

#### New Scripts
- **Core Operations**: `/scripts/` (root helpers)
- **Analysis/Benchmarks**: `/scripts/analysis/`
- **CI/Validation**: `/scripts/ci/`
- **Monitoring**: `/scripts/monitoring/`
- **Maintenance**: `/scripts/maintenance/`
- **Automation/Agents**: `/scripts/agents/`

#### Deprecated Content
- Remove from repo and rely on git history
- Document removals in `docs/DEPRECATIONS.md`
- Update all internal references

### Naming Conventions
- **Documentation**: `category_topic.md` (lowercase, underscores)
- **Scripts**: `action_target.py` (clear purpose indication)
- **Directories**: `lowercase_names/` (descriptive, single purpose)

### Link Maintenance
- All documentation links must be functional
- Use relative paths within repository
- Update references when moving files
- Test links before submitting PRs

### Documentation Quality
- Keep README.md updated with current state
- Document breaking changes
- Include examples for complex features
- Update `AGENTS.md` (canonical) or `docs/agents/*` when agent-facing context changes

## Quality Gate Commands

Before submitting a PR, run these commands to catch issues early:

### Type Checking (mypy)

```bash
# Full type check
uv run mypy src/gpt_trader

# Check specific module (faster iteration)
uv run mypy src/gpt_trader/features/live_trade/execution/
```

**Common mypy issues:**
- Resolve new errors or document existing ones in the PR summary
- New code should pass strict type checking

### Unit Tests

```bash
# Fast parallel run (recommended)
uv run pytest tests/unit -n auto -q

# Verbose output for debugging
uv run pytest tests/unit -v --tb=short

# Run specific test file
uv run pytest tests/unit/gpt_trader/features/live_trade/execution/test_order_submission_flows.py -v

# Run with coverage
uv run pytest tests/unit --cov=src/gpt_trader --cov-report=term-missing
```

### TUI CSS Regeneration

The TUI uses concatenated CSS modules. After editing any `.tcss` file in `src/gpt_trader/tui/styles/`:

```bash
# Regenerate main.tcss from source modules
python scripts/build_tui_css.py

# Verify the TUI renders correctly
uv run gpt-trader tui --mode demo
```

**Important**: Never edit `styles/main.tcss` directly - it's auto-generated.

### TUI Snapshot Tests

Snapshot tests verify TUI rendering consistency:

```bash
# Run snapshot tests
uv run pytest tests/unit/gpt_trader/tui/test_snapshots_*.py -q

# Update snapshots after intentional UI changes
uv run pytest tests/unit/gpt_trader/tui/test_snapshots_*.py --snapshot-update
```

### Naming Standards Check

```bash
# Check for naming convention violations
uv run agent-naming

# Run the /naming skill in Claude Code
/naming
```

## Common CI Failures and Fixes

| Failure | Cause | Fix |
|---------|-------|-----|
| `black --check` | Formatting | Run `uv run black .` |
| `ruff check` | Linting violations | Run `uv run ruff check --fix .` |
| `mypy` errors | Type issues | Fix type annotations (pre-existing shim errors can be ignored) |
| Snapshot mismatch | TUI changed | Review changes, run `--snapshot-update` if intentional |
| CSS out of sync | Edited `.tcss` module | Run `python scripts/build_tui_css.py` |
| Import error | Wrong module path | Use canonical paths (see `docs/DEPRECATIONS.md`) |
| Test using deprecated path | Patch targets shim | Update to patch canonical module directly |

### Full Quality Gate

Run everything before PR:

```bash
# Option 1: Use the /quality skill
/quality

# Option 2: Manual commands
uv run ruff check .
uv run black --check .
uv run mypy src/gpt_trader
uv run pytest tests/unit -n auto -q
pre-commit run --all-files
```

## CI Lanes

The CI workflow (`.github/workflows/ci.yml`) runs checks in parallel lanes for faster feedback and isolated failures.

### Lane Overview

| Lane | Job Name | Command | Purpose |
|------|----------|---------|---------|
| **Lint** | `lint` | `uv run ruff check . && uv run black --check .` | Code style |
| **Type Check** | `typecheck` | `uv run mypy src` | Static typing |
| **TUI CSS** | `tui-css` | `python scripts/ci/check_tui_css_up_to_date.py` | Generated CSS in sync |
| **Unit (Core)** | `unit-tests` | `uv run pytest tests/unit -n auto -q --ignore-glob=tests/unit/gpt_trader/tui/test_snapshots_*.py` | Fast unit tests |
| **TUI Snapshots** | `tui-snapshots` | `uv run pytest tests/unit/gpt_trader/tui/test_snapshots_*.py -v` | Visual regression |
| **Property** | `property-tests` | `uv run pytest tests/property -v` | Property-based tests |
| **Contract** | `contract-tests` | `uv run pytest tests/contract -v` | API contracts |

### Running Lanes Locally

```bash
# Lint lane
uv run ruff check . && uv run black --check .

# Type check lane
uv run mypy src

# TUI CSS lane
python scripts/ci/check_tui_css_up_to_date.py

# Unit tests (core, excluding TUI snapshots)
uv run pytest tests/unit -n auto -q --ignore-glob=tests/unit/gpt_trader/tui/test_snapshots_*.py

# TUI snapshot tests
uv run pytest tests/unit/gpt_trader/tui/test_snapshots_*.py -v

# Property tests
uv run pytest tests/property -v

# Contract tests
uv run pytest tests/contract -v
```

## Pre-commit Hook Configuration

Our pre-commit hooks enforce:
- **Black**: Code formatting (line length 100)
- **Ruff**: Fast Python linting
- **MyPy**: Type checking (optional types)
