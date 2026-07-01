# Remove the unwired account manager and strategy-dev lab

---
status: accepted
date: 2026-07-01
deciders: rj
supersedes:
superseded-by:
---

## Context

Two clusters of code shipped in the tree but were never wired into the
composition root, so no entrypoint could ever reach them. The owner reviewed
both in the 2026-07-01 rehabilitation-plan orchestration session and chose
deletion over wiring.

**Account manager cluster.** `CoinbaseAccountManager`
(`features/brokerages/coinbase/account_manager.py`) was never constructed
anywhere in `src/` — only its own unit tests instantiated it.
`ApplicationContainer` exposes no `account_manager` or `account_telemetry`
provider, so `TradingBot`'s `getattr(container, "account_manager", None)` /
`getattr(container, "account_telemetry", None)` reads were always `None`. That
made two consumers permanently dead:

- `gpt-trader treasury convert|move` (`cli/commands/treasury.py`) required
  `bot.account_manager` and therefore failed unconditionally with "Account
  manager does not support treasury operations". No working funds-movement
  surface ever existed behind it.
- `AccountTelemetryService` (`features/live_trade/telemetry/account.py`) was
  never constructed in `src/`; only tests built it. The `gpt-trader account
  snapshot` command reads `bot.account_telemetry`, which is always `None`, so
  it reports "not available" today — before and after this change.

**Strategy-dev lab cluster.** `features/strategy_dev/lab/` (experiment
tracking, parameter grids) and `features/strategy_dev/monitor/` (performance
monitor, metrics aggregator, alert manager) — about 2.6K source lines — had no
importers outside the package. The only reachable part of the slice is
`strategy_dev/config/`, which backs the `gpt-trader strategy` CLI command via
the config-diff surface.

## Options

- **Option A — Wire the code in.** Give the container real
  `account_manager`/`account_telemetry` providers and route the lab/monitor
  tooling into the optimize workflows. Pays integration and review cost now for
  surfaces nothing currently needs, ahead of any spec that demands them.
- **Option B — Delete the unwired code.** Remove the dead modules, their dead
  consumers, and their tests. Re-introduction later requires a fresh spec and a
  deliberate wiring decision instead of resurrecting speculative code.

## Decision

Delete the unwired code (Option B): `CoinbaseAccountManager`, the `gpt-trader
treasury` command group, `AccountTelemetryService`, the orphaned
`AccountManagerProtocol`, and the `strategy_dev/lab/` and
`strategy_dev/monitor/` subpackages, together with their tests.
`strategy_dev/config/` stays because the `gpt-trader strategy` command uses it.
If account telemetry, treasury operations, or experiment tooling are wanted
later, they start from a fresh spec — not from reviving these modules.

## Consequences

- `gpt-trader treasury` is gone from the CLI. It never worked, so there is no
  migration path; the removal is recorded in
  [DEPRECATIONS.md](../DEPRECATIONS.md).
- `gpt-trader account snapshot` remains registered and still reports that
  snapshot telemetry is unavailable — identical behavior to before. Making it
  work (or removing it and its runbook references) is a separate owner
  decision.
- `TradingBot` no longer exposes `account_manager`/`account_telemetry`
  attributes; no risk, guard, or order-submission logic changed.
- `features/strategy_dev/` shrinks to the config surface
  (`ConfigManager`, `StrategyRegistry`, `StrategyProfile`, config diff).
- Roughly 3.1K source lines and 1.7K test lines leave the tree.

## Safety boundary

This decision does not authorize real broker/API calls, live trading commands,
production preflight, canary operations, credential reads, money movement, or
order submission. It only deletes code that was verified unreachable from every
entrypoint.
