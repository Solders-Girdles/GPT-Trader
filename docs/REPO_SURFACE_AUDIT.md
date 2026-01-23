# Repo Surface Audit

Scope
- Top-level: examples/, experiments/, deploy/, monitoring/, scripts/, data/
- Root dotfiles / root files: .github/, .python-version, .pre-commit-config.yaml, .coveragerc, .gitignore, AGENTS.md
- Excluded (local-only): logs/, runtime_data/, secrets/, .venv/, .uv-cache/
- Do not delete/move: src/, tests/, config/, var/agents/

## Top-level directories

| Path | Status | References/Proof | Recommendation | Notes |
| --- | --- | --- | --- | --- |
| examples/ | present | `rg -n "examples/" -S docs` -> docs/guides/backtesting.md, docs/LEGACY_DEBT_WORKLIST.md | keep | Documented runnable examples. |
| experiments/ | absent | `git ls-files experiments` -> no tracked files | none | Directory does not exist in repo. |
| deploy/ | present | Makefile: `COMPOSE_DIR=deploy/gpt_trader/docker`; docs/MONITORING_PLAYBOOK.md; .github/workflows/deploy.yml | keep | Primary deployment artifacts. |
| monitoring/ | absent | docs/LEGACY_DEBT_WORKLIST.md notes removal of legacy monitoring stack | none | Directory does not exist in repo. |
| scripts/ | present | .github/workflows/ci.yml runs ruff on scripts; Makefile targets; docs reference runbooks | keep (with archive candidates below) | Core tooling + CI guardrails. |
| data/ | absent | `git ls-files data` -> no tracked files | none | Directory does not exist in repo. |

## Root dotfiles / root files

| Path | Status | References/Proof | Recommendation | Notes |
| --- | --- | --- | --- | --- |
| .github/ | present | CI workflows and templates | keep | Required for CI/CD. |
| .python-version | present | Tooling default (pyenv/uv) | keep | Required by setup scripts. |
| .pre-commit-config.yaml | present | pre-commit config | keep | Dev workflow. |
| .coveragerc | present | coverage configuration | keep | Coverage tooling default. |
| .gitignore | present | git ignore rules | keep | Repo hygiene. |
| AGENTS.md | present | AI agent instructions | keep | Repo workflow. |

## Archive candidates (packet 1)

These are unreferenced by docs/CI/Makefile and appear to be manual utilities. To reduce surface without losing history, move to `archived/scripts/`.

| Path | References/Proof | Recommendation | Rationale |
| --- | --- | --- | --- |
| scripts/container_entrypoint.sh | `rg -n "container_entrypoint.sh" -S .` -> no matches | archive | Legacy entrypoint helper not used by deploy Dockerfile. |
| scripts/verify_tui.py | `rg -n "verify_tui.py" -S .` -> no matches | archive | Ad-hoc TUI verification utility. |
| scripts/validate_contrast.py | `rg -n "validate_contrast.py" -S .` -> no matches | archive | Standalone WCAG check, not wired into tooling. |
| scripts/inventory_xfail_skip.py | `rg -n "inventory_xfail_skip.py" -S .` -> no matches | archive | One-off inventory tool; not referenced by CI/docs. |

Planned action (packet 1)
- Move the four scripts above to `archived/scripts/` and add a small README noting they are archived utilities.
- Removal count: 4 moved paths (+ 1 new README) within the 20-path cap.

Post-move dangling reference scan
- `rg -n "container_entrypoint.sh|verify_tui.py|validate_contrast.py|inventory_xfail_skip.py" -S .` (expect no matches)

## Packet 2: scripts lint clean

Goal
- Clear `ruff check scripts` so the local scripts smoke stays green.

Commands used
- `uv run ruff check scripts`
- `uv run ruff check scripts/ci/check_tui_css_up_to_date.py --fix`
- `uv run ruff check scripts`
