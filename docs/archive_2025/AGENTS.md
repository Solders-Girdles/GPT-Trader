# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/bot_v2/` (feature slices under `features/`, orchestration, workflows, security, etc.).
- Tests: `tests/unit/bot_v2/` and `tests/integration/bot_v2/` (`test_*.py`).
- Config: `config/` (Poetry, pytest, pre-commit, and JSON configs).
- Scripts: `scripts/` (e.g., `run_all_tests.sh`, ML training/validation helpers).
- Docs: `docs/` and slice docs in `src/bot_v2/*.md`.
- Environment: copy `.env.template` to `.env` and fill secrets locally.

## Build, Test, and Development Commands
- Run full test suite: `bash scripts/run_all_tests.sh`
- Pytest (all tests): `python -m pytest tests -v --tb=short`
- Coverage locally: `python -m pytest tests --cov=src/bot_v2 --cov-report=term-missing`
- Lint/format: `black src/bot_v2` • `ruff check src/bot_v2 --fix` • `flake8 src/bot_v2 --max-line-length=100 --ignore=E203,W503`
- Types: `mypy src/bot_v2 --ignore-missing-imports`
Notes: CI also runs Bandit and Safety (see `src/bot_v2/deployment/ci/github-actions.yaml`).

## Coding Style & Naming Conventions
- Python 3.12, 4-space indent, max line length 100.
- Use type hints for public functions; prefer explicit `TypedDict`/`dataclass` when helpful.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Keep feature slices isolated; avoid cross-slice coupling.
- Tools: Black (format), Ruff/Flake8 (lint), MyPy (types). Tool settings live in `config/pyproject.toml` and `config/pytest.ini`.

## Testing Guidelines
- Framework: Pytest with marks for `unit`, `integration`, etc. (see `config/pytest.ini`).
- Place unit tests in `tests/unit/bot_v2/`, integration in `tests/integration/bot_v2/`.
- Name tests `test_*.py`; functions `test_*`.
- Target coverage on changed code; ensure critical paths in `src/bot_v2/features/*` and orchestration are covered.

## Commit & Pull Request Guidelines
- Commit style: use type prefixes and tags seen in history, e.g. `feat: add RSI strategy`, `[ORCH-002] fix: handle empty data`, `docs: update slice README`.
- Messages: imperative, concise subject (<72 chars), body for context/why.
- PRs: include summary, linked issues (e.g., `Closes #123`), test evidence (command output/screenshots), and note config/docs updates.
- Pre-flight: run formatters, linters, types, and tests locally; avoid committing `.env` or large data.

## Security & Configuration Tips
- Never hardcode secrets; use `.env` and reference keys via env vars. Detect-secrets baseline at `config/.secrets.baseline`.
- Prefer local/mocked data in tests; avoid external network calls.
- Follow isolation boundaries; security-sensitive code lives under `src/bot_v2/security/`.

## Agent-Specific Notes
- Agent metadata lives in `.claude/`; agent mapping is `.claude/agents/agent_mapping.yaml` (see `agents/README.md`).
- Keep slice docs concise for token efficiency; update `SLICES.md` and affected slice README on functional changes.

