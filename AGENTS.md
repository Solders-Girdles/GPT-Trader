# AGENTS.md — Start Here for AI Agents

This is the **first stop** for any AI coding agent (and a fine one for humans).
It routes; it does not restate policy. Each row below points at the one doc that
owns that fact — read that doc for detail, and change facts there, not here.

Two rules keep this repo from sprawling:

1. **State each fact once; link, don't copy.** The authority on where every kind
   of fact lives is [docs/INFORMATION_ARCHITECTURE.md](docs/INFORMATION_ARCHITECTURE.md).
2. **Opening a PR is not merging.** Merge is a separate, later, explicitly
   approved step (see [Merge discipline](#merge-discipline)).

## Where do I go?

| I need to… | Canonical home |
|------------|----------------|
| Decide **where a fact/doc should live** | [docs/INFORMATION_ARCHITECTURE.md](docs/INFORMATION_ARCHITECTURE.md) |
| Find **where code lives / where to change something** | [docs/agents/CODEBASE_MAP.md](docs/agents/CODEBASE_MAP.md) |
| Understand the **system design** (slices, order pipeline) | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Browse the **full doc index** | [docs/README.md](docs/README.md) |
| Know the **project direction, autonomy boundary, execution gates** | [docs/DIRECTION.md](docs/DIRECTION.md) |
| See **current shipped state** | [docs/STATUS.md](docs/STATUS.md) |
| Follow the **contribution workflow** (setup, PR checklist, test quality) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Understand **local CI, the verification bundle, and the CI-lane contract** | [docs/DEVELOPMENT_GUIDELINES.md](docs/DEVELOPMENT_GUIDELINES.md) |
| Apply **naming standards + approved abbreviations** | [docs/naming.md](docs/naming.md), [docs/agents/glossary.md](docs/agents/glossary.md) |
| Use **dependency injection** (`ApplicationContainer`) | [docs/DI_POLICY.md](docs/DI_POLICY.md) |
| Write or run **tests** | [docs/testing.md](docs/testing.md) |
| Touch the **TUI** | [docs/TUI_STYLE_GUIDE.md](docs/TUI_STYLE_GUIDE.md) |
| Run the **agent review/scout pipeline** or handle review artifacts | [docs/agents/project_review_pipeline.md](docs/agents/project_review_pipeline.md) |
| Find **generated inventories/maps** (env vars, metrics, flows) | `var/agents/**` + [docs/agents/README.md](docs/agents/README.md) |

## Environment (one time)

Python **3.12**, package manager **uv**. Full setup and troubleshooting live in
[CONTRIBUTING.md](CONTRIBUTING.md); the short version:

```bash
uv sync --all-extras --dev
cp config/environments/.env.template .env   # set MOCK_BROKER=1 to run without credentials
```

## Everyday commands

The commands you reach for on almost every task (full reference in
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/DEVELOPMENT_GUIDELINES.md](docs/DEVELOPMENT_GUIDELINES.md)):

```bash
uv run pytest tests/unit -n auto -q     # fast unit tests
uv run ruff check . --fix               # lint (auto-fix)
uv run black .                          # format
uv run mypy src/gpt_trader              # type check
uv run agent-naming                     # naming conventions
make ci-required                        # full local PR-readiness gate
uv run local-ci --profile quick         # faster loop (skips readiness + artifact freshness)
```

## Before you open a PR

- Run `make ci-required` (lint/format, docs audits, type check, agent-artifact
  freshness, TUI CSS, test guardrails, core unit tests). The blocking/advisory
  contract is owned by [docs/DEVELOPMENT_GUIDELINES.md](docs/DEVELOPMENT_GUIDELINES.md).
- If you touched agent-artifact inputs (`scripts/agents/**` or
  `config/environments/.env.template`), run `uv run agent-regenerate` and commit
  the updated `var/agents/**`; confirm with `uv run agent-regenerate --verify`.
- After editing any `.tcss` module, run `python scripts/build_tui_css.py`.
- Fill out [.github/pull_request_template.md](.github/pull_request_template.md);
  link the issue/finding with `Closes #<n>` when there is one.

## Merge discipline

`main` is protected and merge is **not** part of packaging. Open the PR, then stop
until the change is explicitly routed for merge. Before merging: re-read
current-head review/reaction signals, resolve every review thread, and confirm
generated artifacts are fresh. **Green CI is not sufficient** — run
`uv run agent-pr-ready`, which reconciles real mergeability against green checks.

```bash
git switch -c <branch>
git push -u origin HEAD
gh pr create --fill
# Merge only once explicitly approved and all threads are resolved:
# gh pr merge --squash --delete-branch
```

## Trading-safety boundary

Existing live profiles and broker adapters are implementation assets, **not**
approval to automate. Live order submission requires recorded human approval plus
any scoped decision packet; verify venue/API/account capability before adding or
enabling an execution path. The authority is [docs/DIRECTION.md](docs/DIRECTION.md);
findings route through [docs/agents/project_review_pipeline.md](docs/agents/project_review_pipeline.md).

## Hosted-agent setup (Google Jules)

Paste this into the Jules "Initial Setup" window. It configures `.env` with safe
mock defaults, then runs the core unit suite:

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

If you override env via Jules repo settings, use `MOCK_BROKER=1` and `DRY_RUN=1`
(and set `PYTHONWARNINGS=default`, not `1`, if you set it at all).
