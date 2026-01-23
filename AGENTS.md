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

uv run python scripts/ci/check_tui_css_up_to_date.py
uv run pytest tests/unit -n auto -q --ignore-glob=tests/unit/gpt_trader/tui/test_snapshots_*.py
```

### Environment Variables (Optional)

The setup script configures `.env` with safe defaults. If you need to override via Jules repo settings:

```
MOCK_BROKER=1
DRY_RUN=1
```

Note: `PYTHONWARNINGS` value must be `default` (not `1`) if set.

## Project Overview

GPT-Trader is a Coinbase trading system using vertical slice architecture with modern Dependency Injection via `ApplicationContainer`. Trading strategies use technical analysis, not LLM inference.

**Trading Modes:**
- **Spot Trading** - Active (BTC-USD, ETH-USD, top-10 USD pairs)
- **CFM Futures** - Available (US-regulated via Coinbase Financial Markets)
- **INTX Perpetuals** - Code ready, requires non-US INTX account

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

## GitHub Workflow

`main` is protected: changes must land via pull request, with required checks passing (0 approvals required). Use auto-merge to keep the loop fast.

```bash
git switch -c <branch>
git push -u origin HEAD
gh pr create --fill
gh pr merge --auto --squash --delete-branch
```

If you touch `var/agents/**`, run `uv run agent-regenerate` and commit the updated artifacts (CI fails when they’re stale).

### Quality Checks

| Command | Purpose |
|---------|---------|
| `uv run ruff check .` | Lint code |
| `uv run ruff check . --fix` | Lint and auto-fix |
| `uv run black .` | Format code |
| `uv run mypy src/gpt_trader` | Type check |
| `uv run agent-check` | Full quality gate (JSON output) |
| `uv run agent-naming` | Check naming conventions |

### TUI Development

After editing `.tcss` files, rebuild CSS:
```bash
python scripts/build_tui_css.py
```

## Architecture

### Source Structure

| Path | Purpose |
|------|---------|
| `src/gpt_trader/app/` | Composition root - `ApplicationContainer`, config, bootstrap |
| `src/gpt_trader/features/` | Vertical slices (brokerages, live_trade, data, intelligence) |
| `src/gpt_trader/tui/` | Terminal UI (Textual-based) |
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
| CSS out of sync | Run `python scripts/build_tui_css.py` |
| Snapshot mismatch | Review changes, run `--snapshot-update` if intentional |

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/ARCHITECTURE.md` | System design, order execution pipeline |
| `docs/DI_POLICY.md` | Dependency injection patterns |
| `docs/naming.md` | Naming conventions |
| `docs/TUI_STYLE_GUIDE.md` | TUI development guide |
| `CONTRIBUTING.md` | Full contribution workflow |
