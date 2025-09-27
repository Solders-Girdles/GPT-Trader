# Repository Organization Guide

---
status: current
created: 2025-01-01
purpose: Document standardized repository structure and maintenance guidelines
---

## Overview

This guide documents the standardized repository organization implemented in January 2025. The structure is optimized for both human developers and AI agents, with clear categorization and predictable file locations.

## Repository Structure

### Root Directory Policy

The root directory contains **only essential project files**:
- `.env` / `.env.template` - Environment configuration
- `README.md` - Project overview and quick start
- `pyproject.toml` / `poetry.lock` - Dependency management
- `pytest.ini` - Test configuration
- Configuration files: `.gitignore`, `.pre-commit-config.yaml`
- License and documentation: `LICENSE`, `CONTRIBUTING.md`

**Rule**: No source code, documentation, or scripts in root directory.

### Primary Directories

#### Source Code & Configuration
```
src/bot_v2/                     # Active trading system
├── cli.py                      # Main entry point
├── features/                   # 11 vertical slices (~500 tokens each)
│   ├── live_trade/            # Production trading engine
│   ├── ml_strategy/           # ML-driven strategy selection
│   ├── market_regime/         # Market condition detection
│   ├── position_sizing/       # Kelly Criterion & confidence sizing
│   └── brokerages/coinbase/   # Coinbase API integration
└── orchestration/             # System orchestration
    ├── perps_bot.py          # Main trading orchestrator (spot-first, perps future-ready)
    └── live_execution.py      # Order execution engine
```

```
tests/                         # Test suite
├── unit/bot_v2/              # Unit tests for active system
├── integration/bot_v2/       # Integration tests
└── conftest.py               # Test configuration

config/                       # Configuration management
├── profiles/                 # Trading profiles (dev, canary, prod)
├── brokers/                  # Broker-specific configurations  
└── system_config.json        # System-wide configuration
```

#### Organized Scripts Directory

Scripts are organized into **11 logical categories**:

```
scripts/
├── core/                     # Essential operations (7 files)
│   ├── capability_probe.py
│   ├── preflight_check.py
│   ├── stage3_runner.py
│   └── ws_probe*.py
├── testing/                  # Test runners (20 files)
│   ├── test_*.py
│   ├── exchange_sandbox*.py
│   └── paper_trade*.py
├── validation/               # Validation scripts (21 files)
│   ├── validate_*.py
│   └── verify_*.py
├── monitoring/               # Monitoring & dashboards (10 files)
│   ├── dashboard_*.py
│   ├── canary_*.py
│   └── manage_*.py
└── utils/                    # General utilities (27 files)
    ├── check_*.py
    ├── diagnose_*.py
    └── fix_*.py
```

#### Documentation Structure

```
docs/                         # Documentation structure (24 organized files of 170+ total)
├── guides/                   # How-to guides and tutorials  
│   ├── agents.md            # AI agent development guide
│   ├── production.md        # Production deployment
│   ├── testing.md           # Testing procedures
│   └── performance_optimization.md
├── reference/                # Technical reference
│   ├── coinbase.md          # Complete Coinbase reference
│   ├── trading_logic_perps.md # Future-ready INTX trading logic
│   ├── system_capabilities.md # Current system state (historical perps snapshot)
│   └── compatibility_troubleshooting.md
├── ops/                      # Operations documentation
│   └── operations_runbook.md # Daily operations procedures
└── ARCHIVE/                  # Historical documentation
    ├── 2024_legacy/         # Pre-perpetuals documentation
    └── 2024_implementation/ # Implementation archives

Note: This represents organized structure within /docs/, but 140+ additional 
documentation files remain scattered throughout the repository and require 
consolidation.
```

#### Archive Organization

```
archived/                     # Historical content preservation
├── 2025/                    # Current year artifacts
│   ├── dev_sessions/        # Development session reports
│   ├── cleanup/             # Cleanup operation records
│   └── reports/             # Consolidated reports
├── experiments/             # Research and exploration
│   └── domain_exploration_20250818/
├── infrastructure/          # System architecture history
│   └── legacy_systems/      # V1_SYSTEM, V2_HORIZONTAL
├── HISTORICAL/              # Long-term preserved data
│   ├── data/               # Historical datasets
│   ├── plans/              # Development plans
│   ├── reports/            # Legacy reports
│   └── scripts/            # Historical scripts
├── code_experiments/        # Coordination experiments
│   ├── coordination/       # Orchestration experiments
│   └── context/            # Context management
└── data_artifacts/          # Runtime artifacts
    ├── artifacts/          # Generated artifacts
    ├── cache/              # Cached data
    ├── memory/             # Memory stores
    ├── results/            # Execution results
    └── verification_reports/ # Validation reports
```

## File Placement Guidelines

### New Content Placement

#### Documentation
- **How-to Guides**: `/docs/guides/` (e.g., `setup_guide.md`)
- **API Reference**: `/docs/reference/` (e.g., `api_reference.md`)
- **Operations**: `/docs/ops/` (e.g., `maintenance_procedures.md`)
- **Never**: Root directory or scattered locations

#### Scripts
- **Core Operations**: `/scripts/core/` (essential system operations)
- **Testing**: `/scripts/testing/` (test runners, validation)
- **Monitoring**: `/scripts/monitoring/` (dashboards, health checks)
- **Utilities**: `/scripts/utils/` (helper scripts, diagnostics)

#### Source Code
- **New Features**: `/src/bot_v2/features/` (following vertical slice pattern)
- **Tests**: `/tests/unit/bot_v2/` or `/tests/integration/bot_v2/`
- **Configuration**: `/config/` with appropriate subdirectory

### Content Migration

#### When Moving Files
1. Use `git mv` to preserve history
2. Update all internal references
3. Create redirect stubs for high-traffic files
4. Test all documentation links
5. Verify functionality after moves

#### Deprecation Process
1. Move to appropriate `/archived/` subdirectory
2. Maintain organized structure within archives
3. Update documentation to reference new locations
4. Create redirect stub if necessary

## Naming Conventions

### Documentation Files
- Format: `category_topic.md`
- Use lowercase with underscores
- Be descriptive: `performance_optimization.md` not `perf.md`

### Script Files
- Format: `action_target.py`
- Clear purpose: `validate_perps_client.py` not `check.py`
- Group similar actions: `test_*`, `validate_*`, `monitor_*`

### Directories
- Use `lowercase_names/`
- Single purpose per directory
- Descriptive names: `monitoring/` not `mon/`

## Maintenance Guidelines

### Regular Maintenance Tasks

#### Weekly
- Check for files in root directory (should only be project essentials)
- Verify all documentation links are functional
- Review new files for proper categorization

#### Monthly
- Archive completed development sessions to `/archived/2025/`
- Review and organize `/archived/` structure
- Update documentation index in `docs/README.md`

#### Quarterly
- Full repository structure review
- Update this organization guide if patterns change
- Assess and optimize archive organization

### Quality Checks

#### Before Adding New Files
1. **Location Check**: Does it belong in the correct directory?
2. **Naming Check**: Does it follow naming conventions?
3. **Documentation Check**: Are any new references properly linked?
4. **Archive Check**: Should old content be archived first?

#### After Repository Changes
1. **Link Validation**: All internal links functional
2. **Navigation Test**: Can AI agents and developers find content easily?
3. **Structure Integrity**: Does organization remain logical?

## Benefits of This Organization

### For AI Agents
- **Predictable Locations**: Know exactly where to find content
- **Token Efficiency**: Load only relevant sections
- **Clear Navigation**: Logical hierarchy reduces search time
- **No Conflicting Information**: Single source of truth for all content

### For Human Developers
- **Fast Onboarding**: Clear structure immediately understandable
- **Easy Maintenance**: Know exactly where everything belongs
- **Reduced Cognitive Load**: No decision fatigue about file locations
- **Better Discoverability**: Standard naming makes everything findable

### For Operations
- **Operational Scripts**: Clearly categorized in `scripts/core/`
- **Monitoring Tools**: Organized in `scripts/monitoring/`
- **Validation Procedures**: All in `scripts/validation/`
- **Documentation**: Consolidated in logical hierarchy

## Implementation History

### January 2025 Standardization
- **Documentation**: Organized 24 files within /docs/ structure (140+ files remain scattered)
- **Scripts**: Organized 107 scripts into 11 logical categories ✅ COMPLETED
- **Archives**: Enhanced from scattered locations to 3-tier hierarchy ✅ COMPLETED  
- **References**: Fixed primary documentation links within /docs/ structure

**Status**: Scripts and archive organization completed successfully. Documentation consolidation is in early stages - significant work remains to consolidate the 170+ total documentation files throughout the repository.

This standardization represents a **sustainable foundation** for repository growth, ensuring the codebase remains organized and accessible as the project evolves.

## Getting Help

- **Quick Questions**: Check [docs/README.md](../README.md)
- **Development**: See [agents.md](agents.md)
- **Contributing**: Review [../../CONTRIBUTING.md](../../CONTRIBUTING.md)
