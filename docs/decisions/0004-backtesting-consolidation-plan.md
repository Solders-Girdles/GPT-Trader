# 0004 - Backtesting Consolidation Plan

Date: 2026-01-30
Status: accepted

## Context
Two backtesting stacks exist:

- `src/gpt_trader/backtesting`: canonical engine used by optimization and
  core backtesting flows.
- `src/gpt_trader/features/research/backtesting`: a separate simulator used
  by research helpers and tests.

The split creates divergence in sizing, metrics, and execution semantics.

## Decision
`src/gpt_trader/backtesting` is the canonical backtesting engine. The research
backtesting module becomes legacy and will be migrated onto the canonical
engine via a thin adapter.

## Plan (Phased)
1. **Declare canonical engine** (this decision) and update documentation to
   point research users to `gpt_trader.backtesting`.
2. **Adapter layer**: implement a research-facing adapter that maps the
   research simulator inputs/outputs onto the canonical engine data provider,
   simulation broker, and metrics.
3. **Migrate callers/tests**: move research backtesting tests to the adapter
   or canonical engine; freeze feature work on the legacy simulator.
4. **Deprecate legacy module**: add a deprecation entry in `docs/DEPRECATIONS.md`,
   emit warnings (post-test migration), and remove `features/research/backtesting`
   once the adapter covers required use cases.

## Consequences
- Canonical backtest behavior is defined by `src/gpt_trader/backtesting`.
- Research flows will become a compatibility layer rather than a separate engine.
- Metrics and sizing logic will align with optimization and live-trading semantics.
