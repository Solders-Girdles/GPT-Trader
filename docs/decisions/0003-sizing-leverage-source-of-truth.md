# 0003 - Sizing & Leverage Source of Truth

Date: 2026-01-30
Status: accepted

## Context
Sizing and leverage parameters appeared in multiple places:
`BotRiskConfig`, `PerpsStrategyConfig`, and legacy top-level overrides like
`perps_position_fraction`. This caused ambiguity about which values were
actually enforced in live trading and made audits harder.

## Decision
For live trading, `BotRiskConfig` is the canonical source of truth for:
`position_fraction`, `target_leverage`, and `max_leverage`.

Legacy inputs are treated as aliases only:
- `perps_position_fraction` syncs into `BotRiskConfig.position_fraction`.
- Strategy-level sizing/leverage parameters remain for research/backtests but
  are ignored for live sizing/risk enforcement.

## Consequences
- Live sizing in `TradingEngine` reads `BotRiskConfig.position_fraction`.
- The risk manager continues to derive limits from `BotRiskConfig`.
- Legacy overrides remain supported with warnings to guide migration.
