# Architecture Audit & Backlog

---
status: draft
last-updated: 2026-01-30
---

## Scope

Stabilization-focused audit of GPT-Trader’s architecture with an emphasis on cohesion,
clear ownership boundaries, and minimizing divergent implementations.

## Current Observations

- **Two backtesting stacks exist**: `src/gpt_trader/backtesting` is the canonical engine.
  `src/gpt_trader/features/research/backtesting` is now treated as legacy pending
  migration to the canonical engine (see decisions log).
- **Strategy configuration is split** across `PerpsStrategyConfig` (strategy) and
  `BotRiskConfig` (risk), with overlapping fields. Live sizing/leverage now
  treats `BotRiskConfig` as canonical, but research/backtests still rely on
  strategy-level parameters.
- **Optimization** uses the canonical backtesting engine but relies on live Coinbase API
  data for historical candles; it does not yet integrate with a dedicated offline dataset.
- **Strategy type naming** is now normalized to canonical types (`baseline`, `mean_reversion`,
  `ensemble`, `regime_switcher`) with legacy aliases preserved for CLI/YAML inputs.
- **Multiple parameter sources** exist for sizing (`strategy.config.position_fraction`,
  `BotConfig.perps_position_fraction`, `BotRiskConfig.position_fraction`).
- **Research backtesting adapter** accepts order metadata via `Decision.indicators`:
  `order_type`, `price`/`limit_price`, `stop_price`, `tif`/`time_in_force`, and
  `reduce_only` to route non-market orders through the canonical broker.

These are not necessarily “bugs,” but they erode clarity about the canonical data path
and configuration source of truth.

## Backlog (Stabilization)

### P0 — Cohesion & Correctness

- [x] **Backtesting consolidation plan**: declare one canonical backtest engine and
  deprecate the other; add adapters where needed for research workflows.
- [x] **Single source of truth for sizing + leverage**: BotRiskConfig is canonical
  for live sizing and leverage caps; `perps_position_fraction` is a legacy alias
  that syncs into BotRiskConfig, and strategy-level sizing/leverage is ignored
  for live trading (retained for research/backtests).
- [x] **Strategy type normalization**: establish canonical strategy type names and
  document mapping for legacy aliases (e.g., `perps_baseline → baseline`).
- [x] **Historical data pipeline**: offline/cache-only backtest data provider
  added (env: `BACKTEST_DATA_SOURCE=offline`, `BACKTEST_DATA_DIR`), with missing-data
  failures surfaced explicitly for deterministic runs.

### P1 — Architecture Clarity

- [ ] **Config layering specification**: explicit order of precedence across `.env`,
  CLI flags, YAML configs, and strategy artifacts.
- [ ] **Parameter space ownership**: generate optimization parameter space from the
  canonical strategy config dataclasses to reduce drift.
- [ ] **Artifact promotion workflow**: document a “research → publish → activate”
  workflow with required approvals and operational guardrails.

### P2 — Quality & Maintainability

- [ ] **Module ownership map**: clarify which slice owns which domain objects and
  forbid cross-slice imports where possible.
- [ ] **Naming cleanup**: remove legacy aliases, update docs/tests to canonical names.
- [ ] **Backtest/optimize telemetry**: unify metrics output schema for research and
  optimization to simplify artifact evidence ingestion.

## Suggested Next Decisions

- **Which backtesting stack is canonical** (and what timeline for deprecating the other)?
- **Which config layer is authoritative** for sizing/leverage/risk?
- **What data sources are acceptable** for optimization and research runs (live API vs
  cached datasets)?

## Notes

This audit focuses on cohesion and alignment, not feature expansion. Items are intended
to be sequenced as stabilization tasks before new strategy work resumes.
