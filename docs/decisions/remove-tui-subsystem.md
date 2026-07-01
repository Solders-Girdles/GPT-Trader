# Remove the TUI subsystem

---
status: accepted
date: 2026-06-30
deciders: rj
supersedes:
superseded-by:
---

## Context

The Textual TUI (`src/gpt_trader/tui/`) was the human-facing window into the
trading bot: dashboards, an ideas-review screen, and a no-credential demo mode.
It carried a large maintenance surface — ~58K lines across 127 Python and 47
TCSS files plus ~19K lines of tests, a bespoke CSS build step
(`scripts/build_tui_css.py`), snapshot-test and CSS-freshness CI jobs, a
pre-commit guardrail, mypy carve-outs, and the `textual` dependency tree.

Three facts made it a clean removal rather than load-bearing infrastructure:

- **It is a thin adapter, not autonomy machinery.** [DIRECTION.md](../DIRECTION.md)
  already treats CLI / MCP as interchangeable thin adapters over the core
  library. The project's priority is the system the operating agents drive, not
  the human dashboard over it.
- **The approval workflow does not depend on it.** The staged-autonomy
  human-approval loop has a full CLI surface (`gpt-trader ideas
  approve/reject/list/show/...`). The TUI ideas-review screen was a second skin
  over functionality the CLI already owns (see
  [TRADE_IDEA_INTERFACES_DESIGN_NOTES.md](../specs/TRADE_IDEA_INTERFACES_DESIGN_NOTES.md)).
- **Nothing else imported it.** Outside `tui/`, the only references were the
  `gpt-trader tui` command and the `--tui`/`--demo` branches of `gpt-trader run`.
  `StateRegistry`, `StateObserver`, and `TuiState` had no consumers outside the
  package. Demo mode existed only to exercise TUI widgets; the real
  no-credential dev path (`DeterministicBroker` via `gpt-trader run --profile dev
  --dev-fast`) is independent and survives.

## Options

- **Option A — Keep and maintain the TUI.** Preserves a polished operator
  dashboard, but pays ongoing cost (snapshot tests, CSS build, `textual`
  upgrades, mypy ratchet) for a surface that is not on the autonomy path and
  duplicates the `ideas` CLI.
- **Option B — Remove the TUI subsystem.** Drops the maintenance surface and a
  dependency tree; operators use the CLI (`gpt-trader run`, `gpt-trader ideas
  …`, `gpt-trader account …`). Loses the dashboard UX.

## Decision

Remove the TUI subsystem (Option B): the `src/gpt_trader/tui/` package, its
tests, the `gpt-trader tui` command, the `--tui`/`--demo` flags on `gpt-trader
run`, the CSS build tooling and CI jobs, the pre-commit guardrail, the mypy
carve-outs, and the `textual` / `pytest-textual-snapshot` dependencies. The CLI
is the sole human/agent interface; trade-idea review stays on `gpt-trader ideas`.

## Consequences

- Operators and agents drive the bot through the CLI; there is no terminal
  dashboard. `gpt-trader run --profile dev --dev-fast` remains the standalone
  no-credential dev path.
- ~58K lines of TUI code and ~19K lines of tests leave the tree; CI loses the
  TUI CSS and snapshot jobs; the dependency lock drops `textual` and its
  transitive packages.
- The trade-idea human-review surface is the `gpt-trader ideas` command group;
  the removed Textual review screen is recorded in
  [DEPRECATIONS.md](../DEPRECATIONS.md).
- If a future operator UI is wanted, it should be a fresh thin adapter over the
  core library / CLI, not a revival of this package.

## Safety boundary

This decision does not authorize real broker/API calls, live trading commands,
production preflight, canary operations, credential reads, money movement, or
order submission. It only removes a presentation-layer surface.
