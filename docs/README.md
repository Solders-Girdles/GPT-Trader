# Documentation Index

---
status: current
last-updated: 2025-11-24
---

## Start Here

1. **[Architecture](ARCHITECTURE.md)** - Understand vertical slices before touching code
2. **[Monitoring Playbook](MONITORING_PLAYBOOK.md)** - Wire dashboards and alerts before live trading

## Quick Links

| Document | Purpose |
|----------|---------|
| [Architecture](ARCHITECTURE.md) | System design and capabilities |
| [Capability Map](CAPABILITIES.md) | Runtime flow, capability matrix, where to change things |
| [Feature Flags](FEATURE_FLAGS.md) | Configuration precedence and canonical sources |
| [Production Guide](guides/production.md) | Deployment, rollout, and emergency procedures |
| [Reliability Guide](RELIABILITY.md) | Guard stack, degradation responses, chaos testing |
| [Coinbase Reference](reference/coinbase_complete.md) | Complete API integration docs |
| [AI Agent Guide](guides/agents.md) | For AI agents working with this codebase |

## Core Documentation

### Getting Started
- [Complete Setup Guide](guides/complete_setup_guide.md) - Full installation to first trade
- [Testing Guide](guides/testing.md) - Running and writing tests

### Architecture & Design
- [System Architecture](ARCHITECTURE.md) - Component overview and vertical slices
- [Capability Map](CAPABILITIES.md) - Runtime flow, capability matrix, where to change things
- [Feature Flags](FEATURE_FLAGS.md) - Configuration precedence and canonical sources
- [Perpetuals Trading Logic](reference/trading_logic_perps.md) - INTX implementation (future-ready)
- [ADR Index](adr/README.md) - Architecture Decision Records
- [Tooling Reference](TOOLING.md) - Internal utilities and patterns

### Trading Operations
- [Production Deployment](guides/production.md) - Deployment, monitoring, rollback, emergencies
- [Reliability Guide](RELIABILITY.md) - Degradation matrix, config defaults, chaos harness
- [Monitoring Playbook](MONITORING_PLAYBOOK.md) - Metrics, alerting, and dashboards
- [Observability Reference](OBSERVABILITY.md) - Metrics, traces, and structured logging
- [Paper Trading](guides/paper_trading.md) - Safe testing with mock broker
- [Backtesting](guides/backtesting.md) - Historical strategy validation

### Coinbase Integration
- [Complete Reference](reference/coinbase_complete.md) - Authentication, endpoints, troubleshooting
- [Authentication Guide](reference/coinbase_auth_guide.md) - JWT, HMAC, OAuth2 details
- [WebSocket Reference](reference/coinbase_websocket_reference.md) - Real-time data channels

### Development
- [AI Agent Guide](guides/agents.md) - For Claude and other AI agents
- [Development Guidelines](DEVELOPMENT_GUIDELINES.md) - Standards for contributing
- [DI Policy](DI_POLICY.md) - Dependency injection patterns and container usage
- [Security](SECURITY.md) - Security practices and considerations

## Configuration

### Trading Profiles
| Profile | Environment | Use Case |
|---------|-------------|----------|
| **dev** | Mock broker | Development and testing |
| **canary** | Production | Ultra-safe validation (tiny positions) |
| **prod** | Production | Full trading capabilities |

### Environment Setup
- [Environment Template](../config/environments/.env.template) - All configuration options
- Default: Spot trading with HMAC authentication
- Perpetuals require INTX access + `COINBASE_ENABLE_DERIVATIVES=1`

## Additional Resources

- [Risk Integration](RISK_INTEGRATION_GUIDE.md) - Risk management configuration
- [Training Guide](TRAINING_GUIDE.md) - ML model training
- [PnL Calculations](PNL_CALCULATION_DIFFERENCES.md) - Profit/loss methodology
- [Changelog](CHANGELOG.md) - Version history

## Important Notes

### Spot vs Perpetuals

| Mode | Products | Authentication | Flag |
|------|----------|----------------|------|
| **Spot (default)** | BTC-USD, ETH-USD, etc. | HMAC (API key/secret) | — |
| **Perpetuals** | BTC-PERP, ETH-PERP | CDP (JWT) | `COINBASE_ENABLE_DERIVATIVES=1` |

**Note:** Sandbox does not support perpetuals. Bot defaults to spot-only trading.

### Current Focus
- **Primary**: Coinbase spot trading (perps code future-ready)
- **Architecture**: Vertical slice design under `src/gpt_trader/`
- **Testing**: `uv run pytest`

## Getting Help

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **AI Development**: [guides/agents.md](guides/agents.md)
- **Issues**: GitHub Issues

---

*Documentation consolidated November 2025. Historical docs available in git history.*
