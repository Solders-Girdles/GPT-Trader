# Documentation Index

---
status: current
last-updated: 2026-01-23
---

## Start Here

1. **[Architecture](ARCHITECTURE.md)** - Understand vertical slices before touching code
2. **[Monitoring Playbook](MONITORING_PLAYBOOK.md)** - Wire dashboards and alerts before live trading

## Quick Links

| Document | Purpose |
|----------|---------|
| [Architecture](ARCHITECTURE.md) | System design and capabilities |
| [Feature Flags](FEATURE_FLAGS.md) | Configuration precedence and canonical sources |
| [Production Guide](guides/production.md) | Deployment, rollout, and emergency procedures |
| [Readiness Checklist](READINESS.md) | Gates to move from paper to live trading |
| [Reliability Guide](RELIABILITY.md) | Guard stack, degradation responses, chaos testing |
| [TUI Guide](TUI_GUIDE.md) | Launching and operating the terminal UI |
| [TUI Style Guide](TUI_STYLE_GUIDE.md) | Visual standards for TUI components |
| [Coinbase Integration](COINBASE.md) | Coinbase configuration + pointers |
| [Naming Standards](naming.md) | Approved terminology and banned abbreviations |
| [AI Agent Guide](guides/agents.md) | For AI agents working with this codebase |
| [Agent Docs Index](agents/README.md) | AI-focused maps and inventories |
| [Agent Tools](guides/agent-tools.md) | Tooling and helper commands |

## Core Documentation

### Getting Started
- [Complete Setup Guide](guides/complete_setup_guide.md) - Full installation to first trade
- [Testing Guide](guides/testing.md) - Running and writing tests

### Architecture & Design
- [System Architecture](ARCHITECTURE.md) - Component overview and vertical slices
- [Feature Flags](FEATURE_FLAGS.md) - Configuration precedence and canonical sources
- [ADR Index](adr/README.md) - Architecture Decision Records

### Trading Operations
- [Production Deployment](guides/production.md) - Deployment, monitoring, rollback, emergencies
- [Canary Runbook](guides/canary_runbook.md) - Step-by-step canary validation phases
- [Readiness Checklist](READINESS.md) - Paper/live gate criteria and evidence
- [Reliability Guide](RELIABILITY.md) - Degradation matrix, config defaults, chaos harness
- [Monitoring Playbook](MONITORING_PLAYBOOK.md) - Metrics, alerting, and dashboards
- [Alert Runbooks](operations/RUNBOOKS.md) - Incident response procedures by alert type
- [Observability Reference](OBSERVABILITY.md) - Metrics, traces, and structured logging
- [Paper Trading](guides/paper_trading.md) - Safe testing with mock broker
- [Backtesting](guides/backtesting.md) - Historical strategy validation

### Coinbase Integration
- [Coinbase Integration](COINBASE.md) - Configuration + internal entrypoints

### Development
- [AI Agent Guide](guides/agents.md) - For Claude and other AI agents
- [Development Guidelines](DEVELOPMENT_GUIDELINES.md) - Standards for contributing
- [Feature Slice Scaffolding](DEVELOPMENT_GUIDELINES.md#slice-scaffolding) - Bootstrap new slices
- [DI Policy](DI_POLICY.md) - Dependency injection patterns and container usage
- [Naming Standards](naming.md) - Terminology, casing, and banned abbreviations
- [Security](SECURITY.md) - Security practices and considerations

### TUI
- [TUI Guide](TUI_GUIDE.md) - Launching, configuration, and operator workflows
- [TUI Style Guide](TUI_STYLE_GUIDE.md) - Visual standards and component rules
- [TUI Event Migration](guides/tui_event_migration.md) - Event-driven messaging patterns

## Configuration

### Trading Profiles
| Profile | Environment | Use Case |
|---------|-------------|----------|
| **dev** | Mock broker | Development and testing |
| **canary** | Production | Ultra-safe validation (tiny positions) |
| **prod** | Production | Full trading capabilities |

### Environment Setup
- [Environment Template](../config/environments/.env.template) - All configuration options
- Default: Spot trading with JWT authentication
- CFM futures (US) require `TRADING_MODES=cfm` + `CFM_ENABLED=1`
- INTX perps require `COINBASE_ENABLE_INTX_PERPS=1`

## Additional Resources

- [Risk Integration](RISK_INTEGRATION_GUIDE.md) - Risk management configuration
- [PnL Calculations](PNL_CALCULATION_DIFFERENCES.md) - Profit/loss methodology
- [Changelog](CHANGELOG.md) - Version history

## Important Notes

### Spot vs Perpetuals

| Mode | Products | Authentication | Flag |
|------|----------|----------------|------|
| **Spot (default)** | BTC-USD, ETH-USD, etc. | JWT (CDP key) | `TRADING_MODES=spot` |
| **CFM futures (US)** | US futures contracts (expiry-coded symbols) | JWT (CDP key) | `TRADING_MODES=cfm` + `CFM_ENABLED=1` |
| **INTX perps** | BTC-PERP, ETH-PERP | JWT (CDP key) | `COINBASE_ENABLE_INTX_PERPS=1` |

**Note:** Sandbox does not support futures/perps. Bot defaults to spot-only trading.

### Current Focus
- **Primary**: Coinbase spot trading (perps code future-ready)
- **Architecture**: Vertical slice design under `src/gpt_trader/`
- **Testing**: `uv run pytest`

## Getting Help

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **AI Development**: [guides/agents.md](guides/agents.md)
- **Issues**: GitHub Issues

---

*Documentation index updated January 2026. Historical docs live in git history.*
