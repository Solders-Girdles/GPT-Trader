# 0001 - Project Intent: Live Trading First

Date: 2026-01-30
Status: accepted

## Context
The codebase accumulated mixed goals and inconsistent claims over time, which made onboarding and prioritization difficult. The project needs a clear, shared intent to guide stabilization and future work.

## Decision
Prioritize automated live trading on Coinbase as the primary product goal. Research/backtesting are supporting capabilities and must feed live trading through explicit, versioned artifacts. The TUI is the human operations layer; the CLI is designed to enable automation and AI-assisted workflows.

## Consequences
- Documentation and architecture must reflect an in-progress system (not production-ready).
- Non-core pathways can be trimmed or deferred in favor of a single canonical live trading path.
- Research outputs should be publishable artifacts that are auditable and reversible.
- Manual oversight remains required during stabilization, with the goal of reducing it over time.
