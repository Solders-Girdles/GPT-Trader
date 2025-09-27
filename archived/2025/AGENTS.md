# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/bot_v2/`
  - `features/` (backtest, paper_trade, analyze, live_trade, optimize, monitor, data, ml_strategy, market_regime)
  - `orchestration/` (bot manager, execution, broker factory, workflows)
  - `security/` (secrets manager, auth)
  - `deployment/` (docker, k8s)
- Tests: `tests/` with `unit/`, `integration/`, and shared `fixtures/`.
- Utilities: `scripts/` (e.g., `run_perps_bot.py`).
- Config & docs: `config/`, `docs/`, `.env.template`.

## Build, Test, and Development Commands
- Install: `poetry install` — install deps incl. dev tools.
- Hooks: `pre-commit install` — enable lint/format checks on commit.
- Lint: `poetry run ruff check .` (auto-fix: `poetry run ruff --fix .`).
- Format: `poetry run black .` — apply code style.
- Types: `poetry run mypy src` — strict type checking.
- Tests: `poetry run pytest -q` — run fast test suite.
- Coverage: `poetry run pytest --cov=bot_v2 --cov-report=term-missing`.
- Run bot (dev): `poetry run perps-bot --profile dev --dev-fast`.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, max line length 100.
- Tools: Black (format), Ruff (lint), Mypy (types). Config in `pyproject.toml`.
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Type hints required (mypy strict; `self/cls` annotations ignored by policy).

## Testing Guidelines
- Framework: Pytest (`pytest.ini` config). Keep tests deterministic; no live broker/network.
- Structure: fast checks in `tests/unit/`; e2e/broker shims in `tests/integration/`.
- Fixtures: place shared data/helpers in `tests/fixtures/`.
- Naming: `tests/unit/test_<area>.py` (e.g., `test_week3_execution.py`).

## Commit & Pull Request Guidelines
- Commits: Prefer Conventional Commits (e.g., `feat(runner): add --duration-minutes flag`, `fix(auth): ensure JWT is default for derivatives`).
- PRs: include clear description, linked issues, before/after behavior, test coverage notes, and docs updates when needed.
- CI: all checks must pass (pre-commit, lint, types, tests). If `pyproject.toml` changes, commit updated `poetry.lock`.

## Security & Configuration Tips
- Copy `.env.template` to `.env`; never commit secrets.
- Prefer `bot_v2/security/secrets_manager.py` (optional Vault) for sensitive configs.
- Useful env flags: `COINBASE_SANDBOX=1`, `BROKER=coinbase`.
- Keep `.env` out of version control.

