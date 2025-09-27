# Documentation Index

---
status: current
last-updated: 2025-03-01
organization-updated: 2025-03-01
---

## 📍 Quick Links

- **[Complete Setup Guide](guides/complete_setup_guide.md)** - Full installation and configuration
- **[Coinbase Reference](reference/coinbase_complete.md)** - Complete integration documentation
- **[Coinbase API Links (Lite)](reference/coinbase_api_links.md)** - Fast links for agents
- **[Architecture](ARCHITECTURE.md)** - System design and capabilities
- **[AI Agent Guide](guides/agents.md)** - For AI agents and automation

## 📚 Core Documentation

### Getting Started
- [Complete Setup Guide](guides/complete_setup_guide.md) - Everything from installation to first trade
- [Testing Guide](guides/testing.md) - Running and writing tests

### Architecture & Design
- [System Architecture](ARCHITECTURE.md) - Component overview
- [Perpetuals Trading Logic](reference/trading_logic_perps.md) - Future-ready INTX implementation details
- [Vertical Slice Architecture](../src/bot_v2/SLICES.md) - Modular design

### Trading Operations
- [Production Deployment](guides/production.md) - Deployment guide (spot-first, INTX-gated perps)
- [Stage 3 Runner](../scripts/stage3_runner.py) - Legacy entry that forwards to `perps-bot`
- [Operations Runbook](ops/operations_runbook.md) - Operational procedures

### Development
- [AI Agent Guide](guides/agents.md) - For Claude and other AI agents
- [Repository Organization](guides/repository_organization.md) - Structure and maintenance standards
- [Contributing Guidelines](../CONTRIBUTING.md) - Development workflow
- [Performance Optimization](guides/performance_optimization.md) - Performance tuning

## 🔧 Configuration & Setup

### API Configuration  
- [Coinbase Complete Reference](reference/coinbase_complete.md) - All Coinbase integration documentation
- [Environment Template](../.env.template) - Environment variables template

### Trading Profiles
- **Development** - Mock broker, deterministic fills
- **Canary** - Ultra-safe production testing
- **Production** - Full trading capabilities

## 📊 Reports & Analysis

### Implementation Status
- [Production Readiness](guides/production.md#production-readiness-requirements)
- [System Capabilities](reference/system_capabilities.md) - Current state overview
- [System Reality](reference/system_reality.md) - Honest current state assessment
- [Compatibility & Troubleshooting](reference/compatibility_troubleshooting.md) - Technical requirements
- [Repository Standardization Audit](REPOSITORY_STANDARDIZATION_AUDIT.md) - Documentation consolidation progress

### Performance & Operations
- [Operations Runbook](ops/operations_runbook.md) - Daily operations and monitoring
- [Performance Optimization](guides/performance_optimization.md) - Tuning guidelines

## 🗄️ Archive

### Legacy Documentation
- [2024 Legacy Docs](ARCHIVE/2024_legacy/) - Pre-perpetuals era
- [2024 Implementation Archive](ARCHIVE/2024_implementation/) - Historical implementation docs

### Archived Reports
- [2025 Development Reports](../archived/2025_reports/) - Current year archives
- [Historical Archives](../archived/HISTORICAL/) - Long-term preserved data
- [Perps Runbooks (archived)](archive_2025/) - Legacy INTX/perps go-live guides

## 🚨 Important Notes

### Production vs Sandbox

| Environment | Products | API | Authentication |
|------------|----------|-----|----------------|
| **Production (default)** | Spot (BTC-USD, ETH-USD, …) | Advanced Trade v3 (HMAC) | API key/secret |
| **Production (perps)** | Perpetuals (INTX-gated) | Advanced Trade v3 | CDP (JWT) + `COINBASE_ENABLE_DERIVATIVES=1` |
| **Sandbox** | Not used for live trading (API diverges) | — | Use only with `PERPS_PAPER=1` |

**Critical:** Sandbox does **not** support perpetuals. The bot defaults to spot trading and only enables perps when INTX access plus derivatives flag are detected.

### Current Focus
- **Primary**: Coinbase spot trading (perps code paths kept future-ready)
- **Architecture**: Vertical slice design (production vs experimental slices clearly marked)
- **ML**: Strategy selection, regime detection, Kelly sizing (experimental slices)
- **Testing**: 480 collected / 422 selected tests (`poetry run pytest --collect-only`)

## 📞 Getting Help

- **Quick Questions**: Check [QUICK_START.md](QUICK_START.md)
- **Architecture**: Review [ARCHITECTURE.md](ARCHITECTURE.md)
- **AI Development**: See [guides/agents.md](guides/agents.md)
- **Issues**: GitHub Issues page

---

*Last validated: March 2025 (spot-first refresh)*
