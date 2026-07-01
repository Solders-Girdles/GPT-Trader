# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## Jules Setup

### Initial Setup Script

For Google Jules, paste this into the "Initial Setup" configuration window:

```bash
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
uv sync --all-extras --dev

test -f .env || cp config/environments/.env.template .env

uv run python -c "import re; from pathlib import Path; p=Path('.env'); p.write_text(re.sub(r'^MOCK_BROKER=.*$','MOCK_BROKER=1',p.read_text(),flags=re.M))"

uv run pytest tests/unit -n auto -q
```

### Environment Variables (Optional)

The setup script configures `.env` with safe defaults. If you need to override via Jules repo settings:

```
MOCK_BROKER=1
DRY_RUN=1
```

Note: `PYTHONWARNINGS` value must be `default` (not `1`) if set.

## Project Overview

GPT-Trader is a Coinbase-oriented trading system using vertical slice architecture with modern Dependency Injection via `ApplicationContainer`. Trading strategies currently use technical analysis and rule-based decisioning, not LLM inference.

Do not treat existing live profiles or broker adapters as proof that a product should be automated. Before migration or new execution work, use `docs/DIRECTION.md`: start from the current approval-gated execution phase (`human_approved_execution` compatibility label), keep broker-neutral records canonical, route unresolved live-control choices through `decision-needed` packets, and verify venue/API/account capability before adding or enabling execution paths.

**Trading Modes:**
- **Spot Trading** - Implemented Coinbase spot paths; require explicit profile and readiness gates
- **CFM Futures** - Implemented/gated US-regulated futures paths; require account, product, and risk verification
- **INTX Perpetuals** - Implemented/gated international derivatives paths; require eligible-region/account verification
- **AI-assisted execution** - Planning and approval-gated tickets only; live order submission requires recorded human approval plus any explicitly scoped decision packet or runbook constraints before any order submission

## Environment Setup

**Python Version:** 3.12 (required)
**Package Manager:** uv

Install dependencies:
```bash
uv sync --all-extras --dev
```

Create environment config:
```bash
cp config/environments/.env.template .env
```

For testing without real API access, set `MOCK_BROKER=1` in `.env`.

## Commands

### Testing

| Command | Purpose |
|---------|---------|
| `uv run pytest tests/unit -q` | Run unit tests |
| `uv run pytest tests/unit -n auto -q` | Run tests in parallel |
| `uv run pytest tests/unit/path/to/test.py -v` | Run single test file |
| `uv run pytest tests/unit --cov=src/gpt_trader` | Run with coverage |

### Local CI

Run `make ci-required` when you want the local PR-readiness command set. It runs
lint/format, docs audits, type checks, agent artifact freshness, test guardrails,
and core unit tests, stopping on the first local failure.
GitHub pull_request CI runs a related agent-freshness job, but stale artifacts
are reported as non-blocking on pull requests. GitHub pull_request CI also does
not run the canary readiness gate.

Run `uv run local-ci` (or `python -m gpt_trader.ci.local_ci`) with the default
strict/full profile when you also want local/live readiness evidence. Strict/full
runs the local PR-readiness validation set plus the canary readiness gate and
agent artifacts freshness; the CLI prints the selected profile and the
readiness/artifact statuses before executing commands.

For faster loops without readiness reports or regenerating `var/agents`, use the quick/dev profile via `--profile quick` or `--profile dev`. That profile disables the readiness gate and agent artifacts freshness steps (the output notes which checks were skipped and why). Run `make ci-required` for the local PR-readiness command set, and run strict `uv run local-ci` when you need the additional local/live readiness gate before operational readiness work.

## GitHub Workflow

`main` is protected: changes must land via pull request with required checks
passing. Open the PR; **merge is a separate, later decision** — merge only when
the change is explicitly routed/approved for merge and all review threads are
resolved. Green CI is not sufficient (`uv run agent-pr-ready` reconciles real
mergeability against green checks).

```bash
git switch -c <branch>
git push -u origin HEAD
gh pr create --fill
# Merge only once explicitly approved/routed and review threads are resolved:
# gh pr merge --squash --delete-branch
```

If you touch `var/agents/**` or any agent-artifact inputs (notably `scripts/agents/**` or `config/environments/.env.template`), run `uv run agent-regenerate` and commit the updated artifacts. Local `make ci-required` and non-PR CI fail when they are stale; GitHub pull request CI reports stale artifacts as non-blocking. To check quickly: `uv run agent-regenerate --verify`.

### Quality Checks

| Command | Purpose |
|---------|---------|
| `uv run ruff check .` | Lint code |
| `uv run ruff check . --fix` | Lint and auto-fix |
| `uv run black .` | Format code |
| `uv run mypy src/gpt_trader` | Type check |
| `uv run agent-check` | Full quality gate (JSON output) |
| `uv run agent-naming` | Check naming conventions |

### Agent Review Artifacts

Review and analysis artifacts that should be durable project records belong in
`review_artifacts/`. Commit only intended review CSV/XLSX deliverables there;
keep temporary review outputs under `review_artifacts/tmp/`, and do not commit
large datasets or secrets.

## Architecture

### Source Structure

| Path | Purpose |
|------|---------|
| `src/gpt_trader/app/` | Composition root - `ApplicationContainer`, config, bootstrap |
| `src/gpt_trader/features/` | Vertical slices (brokerages, live_trade, data, intelligence) |
| `src/gpt_trader/monitoring/` | Runtime guards, metrics |
| `src/gpt_trader/validation/` | Declarative validators |
| `src/gpt_trader/errors/` | Centralized error hierarchy |

### Key Feature Slices

| Slice | Purpose |
|-------|---------|
| `features/brokerages/coinbase/` | REST/WebSocket integration, client mixins |
| `features/live_trade/` | Trading engine, strategies, risk management |
| `features/live_trade/execution/` | Guards, validation, order submission |
| `features/intelligence/sizing/` | Kelly criterion position sizing |

### Dependency Injection

Use `ApplicationContainer` for all new services (see `docs/DI_POLICY.md`):

```python
# Preferred: Receive container or service as parameter
def my_function(container: ApplicationContainer) -> None:
    service = container.my_service

# Tests: Create own container instance
@pytest.fixture
def container(mock_config):
    return ApplicationContainer(mock_config)
```

**Avoid:** Module-level singletons, service locators in business logic.

## Naming Standards

**Banned abbreviations:** `cfg`, `svc`, `mgr`, `util`, `utils`, `amt`, `calc`, `upd`

Use `# naming: allow` to suppress warnings for external API fields.

See `docs/agents/glossary.md` for approved abbreviations (e.g., `qty`, `PnL`, `API`).

## Testing

### Test Organization

| Directory | Purpose | Default Run |
|-----------|---------|-------------|
| `tests/unit/` | Unit tests | Yes |
| `tests/integration/` | Integration tests | No (marker excluded) |
| `tests/property/` | Hypothesis property tests | Separate |
| `tests/contract/` | API contract tests | Separate |

### Markers

Tests excluded by default: `integration`, `real_api`, `uses_mock_broker`, `legacy_delete`, `legacy_modernize`

Run specific markers:
```bash
uv run pytest -m "spot" tests/unit         # Spot trading tests
uv run pytest -m "risk" tests/unit         # Risk management tests
uv run pytest -m "integration" tests/      # Integration tests (opt-in)
```

## Common Issues

| Issue | Fix |
|-------|-----|
| `black --check` fails | Run `uv run black .` |
| `ruff check` fails | Run `uv run ruff check --fix .` |

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/ARCHITECTURE.md` | System design, order execution pipeline |
| `docs/DI_POLICY.md` | Dependency injection patterns |
| `docs/naming.md` | Naming conventions |
| `CONTRIBUTING.md` | Full contribution workflow |

### Review Artifacts Convention (run goal-pipeline-20260626-001-gpt-trader-clean-discovery-scout)
Review deliverables (CSVs, spreadsheets from agent review lanes) are tracked only in `review_artifacts/`.
- `!review_artifacts/` exception in .gitignore (global *.csv still protects data/).
- See docs/agents/project_review_pipeline.md for details and verification.
- Keeps handoff data durable without broad un-ignores.
