# Project Charter

---
status: current
last-updated: 2026-01-30
---

## Purpose
Build a Coinbase-focused automated live trading system that is safe, observable, and iteratively improving.

## Primary User
Single operator (project owner).

## Current Status
Foundation stabilization. The system is not yet production-ready; live trading is the north star, and research/backtesting exists to support that goal.

## Scope (In)
- Live trading engine and execution pipeline
- Coinbase integration (spot and CFM futures; INTX perps gated by account access)
- Risk management, guardrails, and preflight checks
- Observability, audit logs, and incident runbooks
- TUI for human operations; CLI for automation/agent workflows
- Research/backtesting outputs that can be validated and promoted to live trading

## Scope (Out of Scope for Now)
- Multi-exchange support
- Fully autonomous, unsupervised trading without oversight
- SaaS or multi-tenant UI
- Ultra-low-latency or HFT-style execution

## Success Measures
- Safe live trading with bounded risk, reliable order execution, and clear failure modes
- Repeatable, documented deploy/operate workflow for a single operator
- Strategy improvements that are traceable to validated research artifacts
- Reduced manual oversight over time as trust and safeguards increase

## Operating Principles
- Prefer clarity over cleverness; reduce paths and consolidate behavior
- Make safety constraints explicit and enforce them before live trading
- Keep research outputs versioned, auditable, and reversible
- Maintain a single canonical execution path for live orders
