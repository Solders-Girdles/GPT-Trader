# Documentation Index

---
status: current
last-updated: 2026-01-24
---

## Start Here

1. **[Architecture](ARCHITECTURE.md)** - Understand vertical slices before touching code
2. **[Development Guidelines](DEVELOPMENT_GUIDELINES.md)** - Where to change things + guardrails
3. **[Monitoring Playbook](MONITORING_PLAYBOOK.md)** - Metrics + runbooks (when operating)

## Quick Links

| Document | Purpose |
|----------|---------|
| [Architecture](ARCHITECTURE.md) | System design and capabilities |
| [Production Guide](production.md) | Deployment, rollout, and emergency procedures |
| [Readiness Checklist](READINESS.md) | Gates to move from paper to live trading |
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
- [DI Policy](DI_POLICY.md) - Dependency injection patterns and container usage

### Trading Operations
- [Production Deployment](production.md) - Deployment, monitoring, rollback, emergencies
- [Readiness Checklist](READINESS.md) - Paper/live gate criteria and evidence
- [Reliability Guide](RELIABILITY.md) - Degradation matrix, config defaults, chaos harness
- [Monitoring Playbook](MONITORING_PLAYBOOK.md) - Metrics, alerting, and dashboards
- [Alert Runbooks](RUNBOOKS.md) - Incident response procedures by alert type
- [Observability Reference](OBSERVABILITY.md) - Metrics, traces, and structured logging

### Coinbase Integration
- [Coinbase Integration](COINBASE.md) - Configuration + internal entrypoints

### Development
- [Development Guidelines](DEVELOPMENT_GUIDELINES.md) - Standards for contributing
- [Feature Slice Scaffolding](DEVELOPMENT_GUIDELINES.md#slice-scaffolding) - Bootstrap new slices
- [Naming Standards](naming.md) - Terminology, casing, and banned abbreviations
- [Security](SECURITY.md) - Security practices and considerations
- [Deprecations](DEPRECATIONS.md) - Deprecated modules and shims (CI-checked)

### TUI
- [TUI Guide](TUI_GUIDE.md) - Launching, configuration, and operator workflows
- [TUI Style Guide](TUI_STYLE_GUIDE.md) - Visual standards and component rules

## Configuration

### Trading Profiles
| Profile | Environment | Use Case |
|---------|-------------|----------|
| **dev** | Mock broker | Development and testing |
| **canary** | Production | Ultra-safe validation (tiny positions) |
| **prod** | Production | Full trading capabilities |

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
- **AI Development**: [AGENTS.md](../AGENTS.md) and [Agent Docs Index](agents/README.md)
- **Issues**: GitHub Issues

---

*Documentation index updated January 2026. Prefer code + generated inventories over long-lived how-to docs.*
