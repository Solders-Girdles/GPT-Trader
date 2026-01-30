# 0002 - Strategy Type Normalization

Date: 2026-01-30
Status: accepted

## Context
Strategy naming drifted across the codebase and tooling. Live config uses
canonical names like `baseline`, while optimization and research flows used
aliases such as `perps_baseline` and `spot`. This ambiguity made it unclear
which strategy family was authoritative and complicated artifact handling.

## Decision
Adopt canonical strategy type names and treat legacy names as aliases:

- Canonical types: `baseline`, `mean_reversion`, `ensemble`, `regime_switcher`
- Aliases: `perps_baseline → baseline`, `spot → baseline`
- Baseline variants are represented separately as `perps` (default) or `spot`

CLI and YAML inputs may still use aliases, but internal configuration and
artifacts resolve to canonical strategy types.

## Consequences
- Optimization config now records `strategy_type=baseline` plus a
  `strategy_variant` (`perps` or `spot`) when applicable.
- Strategy artifacts store canonical `strategy_type` values.
- Documentation can reference a single set of strategy names while preserving
  backwards compatibility for existing CLI/YAML inputs.
