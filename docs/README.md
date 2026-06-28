# Documentation Index

---
status: current
---

All docs under `docs/` must be reachable from this index (directly or via other linked docs).
When adding a new doc, link it below in the best-fit section.

### Doc metadata convention

Each doc carries a frontmatter block with a single required field:

```yaml
---
status: current   # one of: current, draft, deprecated, superseded
---
```

There is intentionally **no `last-updated` field**: it drifted out of sync with
reality and gave a false signal. **Git history is the source of truth for when a
doc changed** (`git log -1 -- docs/<file>.md`); `status` is the human signal git
cannot provide. The metadata block is enforced by
`scripts/maintenance/docs_reachability_check.py`.

## Start Here

0. **[Project Status](STATUS.md)** - Where we actually are: shipped reality vs the staged-autonomy target
1. **[Architecture](ARCHITECTURE.md)** - Understand vertical slices before touching code
2. **[Development Guidelines](DEVELOPMENT_GUIDELINES.md)** - Where to change things + guardrails
3. **[Readiness Checklist](READINESS.md)** - Gates to move from paper to live trading
4. **[Monitoring Playbook](MONITORING_PLAYBOOK.md)** - Metrics + runbooks (when operating)

## Quick Links

| Document | Purpose |
|----------|---------|
| [Project Status](STATUS.md) | Living "you are here": shipped state per stage vs the target |
| [Architecture](ARCHITECTURE.md) | System design and capabilities |
| [Architecture Boundaries](architecture/BOUNDARIES.md) | Layer ownership and dependency direction |
| [Ownership Map](architecture/OWNERSHIP.md) | Module ownership map and boundaries |
| [Entrypoints](architecture/ENTRYPOINTS.md) | CLI, TUI, preflight, and live bot wiring |
| [Core Cleanup Roadmap](CORE_CLEANUP_ROADMAP.md) | Cleanup lanes, decision backlog, and verification bundle |
| [Decision Log](GPT_TRADER_DECISION_LOG.md) | Durable product and engineering decisions |
| [Live Operations Guide](production.md) | Readiness-gated live operations, rollback, and emergency procedures |
| [Readiness Checklist](READINESS.md) | Gates to move from paper to live trading |
| [Pre-Migration Decision Framework](PRE_MIGRATION_DECISION_FRAMEWORK.md) | AI-assisted trading autonomy, product, venue, approval, and audit gates |
| [Trade-Idea Interface Design Notes](specs/TRADE_IDEA_INTERFACES_DESIGN_NOTES.md) | Implemented CLI and future TUI workstreams for human-approved trade ideas |
| [Operating Rubric](OPERATING_RUBRIC.md) | Staged capabilities and graduation evidence for the autonomous entity |
| [Operating Rubric v2 Draft](OPERATING_RUBRIC_V2.md) | Draft measured-outcome rubric for owner review |
| [Reliability Guide](RELIABILITY.md) | Guard stack, degradation responses, chaos testing |
| [TUI Guide](TUI_GUIDE.md) | Launching and operating the terminal UI |
| [TUI Style Guide](TUI_STYLE_GUIDE.md) | Visual standards for TUI components |
| [Coinbase Integration](COINBASE.md) | Coinbase configuration + pointers |
| [Naming Standards](naming.md) | Approved terminology and banned abbreviations |
| [Feature Flags](FEATURE_FLAGS.md) | Config precedence + canonical sources |
| [AI Agent Reference](../AGENTS.md) | Agent workflow + repo rules (canonical) |
| [Agent Docs Index](agents/README.md) | AI-focused maps and pointers |

## Core Documentation

### Getting Started
- [Testing Guide](testing.md) - Running and writing tests
- [Paper Trading](paper_trading.md) - Safe testing with mock broker

### Architecture & Design
- [System Architecture](ARCHITECTURE.md) - Component overview and vertical slices
- [Architecture Boundaries](architecture/BOUNDARIES.md) - Layer ownership and dependency direction
- [DI Policy](DI_POLICY.md) - Dependency injection patterns and container usage
- [Entrypoints](architecture/ENTRYPOINTS.md) - CLI, TUI, preflight, and live bot wiring
- [Core Seams](architecture/SEAMS.md) - Canonical boundaries for Strategy/Execution/Data/Config

### Trading Operations
- [Live Operations](production.md) - Readiness-gated live operations, monitoring, rollback, emergencies
- [Readiness Checklist](READINESS.md) - Paper/live gate criteria and evidence
- [Pre-Migration Decision Framework](PRE_MIGRATION_DECISION_FRAMEWORK.md) - Gates before AI-assisted execution migration
- [Reliability Guide](RELIABILITY.md) - Degradation matrix, config defaults, chaos harness
- [Monitoring Playbook](MONITORING_PLAYBOOK.md) - Metrics, alerting, and dashboards
- [Alert Runbooks](RUNBOOKS.md) - Incident response procedures by alert type
- [Observability Reference](OBSERVABILITY.md) - Metrics, traces, and structured logging

### Coinbase Integration
- [Coinbase Integration](COINBASE.md) - Configuration + internal entrypoints

### Development
- [Core Cleanup Roadmap](CORE_CLEANUP_ROADMAP.md) - Cleanup lanes, ready queue, decision backlog
- [Test Hygiene Policy](test_hygiene.md) - Line limits, allowlist policy, and splitting guidance
- [Development Guidelines](DEVELOPMENT_GUIDELINES.md) - Standards for contributing
- [Feature Slice Scaffolding](DEVELOPMENT_GUIDELINES.md#slice-scaffolding) - Bootstrap new slices
- [Naming Standards](naming.md) - Terminology, casing, and banned abbreviations
- [Naming Suppressions](naming_suppressions.md) - Strict mode behavior and how to suppress
- [Security](SECURITY.md) - Security practices and considerations
- [Deprecations](DEPRECATIONS.md) - Deprecated modules and shims (CI-checked)
- Research backtests: adapter order intent keys live in `Development Guidelines`

### TUI
- [TUI Guide](TUI_GUIDE.md) - Launching, configuration, and operator workflows
- [TUI Style Guide](TUI_STYLE_GUIDE.md) - Visual standards and component rules

## Configuration

- [Strategy Profile Diff](STRATEGY_PROFILE_DIFF.md) - Compare baseline strategy settings against runtime profile values

### Trading Profiles

Profiles are config snapshots, not execution approval. Live profiles
(`canary`, `prod`) only run after the gates in
[Live Operations](production.md) and the
[Pre-Migration Decision Framework](PRE_MIGRATION_DECISION_FRAMEWORK.md) are
satisfied with recorded human approval.

| Profile | Broker / data | Role |
|---------|---------------|------|
| **dev** | Mock broker | Development and testing |
| **observe** | Real data, blocked execution | Account and market observation |
| **canary** | Live broker, tightly capped | Legacy live-validation asset; runs require recorded approval |
| **prod** | Live broker | Legacy live-operation asset; runs require explicit approval and monitoring |

### Environment Setup
- [Environment Template](../config/environments/.env.template) - Minimal operator config (safe defaults)
- [Environment Variable Inventory](../var/agents/configuration/environment_variables.md) - Full, code-derived reference
- Default: Spot trading with JWT authentication
- CFM futures (US) require `TRADING_MODES=cfm` + `CFM_ENABLED=1`
- INTX perps require `COINBASE_ENABLE_INTX_PERPS=1`

## Additional Resources

- [Risk Integration](RISK_INTEGRATION_GUIDE.md) - Risk management configuration
- [PnL Calculations](PNL_CALCULATION_DIFFERENCES.md) - Profit/loss methodology
- [Changelog](CHANGELOG.md) - Version history

## Important Notes

### Spot vs Perpetuals

These flags are capability selectors, not approval. Enabling a flag exposes the
relevant adapter; live execution still requires the gates in
[Live Operations](production.md) and venue/account verification.

| Mode | Products | Authentication | Flag |
|------|----------|----------------|------|
| **Spot (default capability)** | BTC-USD, ETH-USD, etc. | JWT (CDP key) | `TRADING_MODES=spot` |
| **CFM futures (US)** | US futures contracts (expiry-coded symbols) | JWT (CDP key) | `TRADING_MODES=cfm` + `CFM_ENABLED=1` |
| **INTX perps** | BTC-PERP, ETH-PERP | JWT (CDP key) | `COINBASE_ENABLE_INTX_PERPS=1` |

**Note:** Sandbox does not support futures/perps. Default capability is spot only.

### Current Focus
- **Primary implementation:** Coinbase spot adapters (perps code paths compile but require INTX access and approval)
- **Architecture**: Vertical slice design under `src/gpt_trader/`
- **Testing**: `uv run pytest`

## Getting Help

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **AI Development**: [AGENTS.md](../AGENTS.md) and [Agent Docs Index](agents/README.md)
- **Issues**: GitHub Issues

---

*Documentation index updated January 2026. Prefer code + generated inventories over long-lived how-to docs.*
