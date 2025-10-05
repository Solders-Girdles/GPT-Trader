# Current vs Target Architecture

**Generated**: 2025-10-05
**Purpose**: Phase 0 baseline inventory of active vs retired/candidate modules

---

## Executive Summary

**Codebase size**: 316 Python files in `src/bot_v2`
**Archive status**: Minimal archived code (1 monitoring module, historical docs)
**Overall health**: Clean structure with vertical slice architecture, some areas need attention

### Immediate Red Flags 🚨

1. **Coinbase Integration Incomplete**: 20+ endpoints still need routing implementation (Status: "Critical Fixes Applied ⚠️")
2. **Live Trade Complexity**: 33+ files in `live_trade/` feature - potential refactoring candidate
3. **Missing Code Markers**: No TODO/FIXME/DEPRECATED comments found (unusual for active codebase)

---

## Active Slices (Production/Development)

### ✅ Core Features - Fully Active

#### `features/adaptive_portfolio/`
**Status**: ACTIVE, WELL-DOCUMENTED
**Files**: 13 modules + strategy_handlers subdir
**Purpose**: Configuration-first portfolio management with tier-based adaptation
**Health**: Has comprehensive README, complete implementation

**Key components**:
- `adaptive_portfolio.py` - Main entry point
- `config_manager.py` - Hot-reloadable configuration
- `tier_manager.py` - Automatic tier detection with hysteresis
- `risk_manager.py` - Adaptive position sizing
- `strategy_selector.py` - Tier-based strategy allocation
- `strategy_handlers/` - Strategy implementations per tier

**Assessment**: Production-ready, good documentation, no immediate concerns

---

#### `features/paper_trade/`
**Status**: ACTIVE, FEATURE-COMPLETE
**Files**: 12 modules + dashboard subdir
**Purpose**: Paper trading simulation with live dashboard

**Key components**:
- `paper_trade.py` - Main coordinator
- `trading_loop.py` - Execution loop
- `execution.py` - Order execution simulation
- `dashboard/` - 6 modules for console/HTML reporting
  - `console_renderer.py`
  - `html_report_generator.py`
  - `metrics.py`
  - `display_controller.py`

**Assessment**: Complete, well-structured, active development

---

#### `features/live_trade/` ⚠️
**Status**: ACTIVE, HIGH COMPLEXITY
**Files**: 33+ modules + 3 subdirs (advanced_execution_models, risk, strategies)
**Purpose**: Live trading execution, risk management, position management

**Key components**:
- **Execution**: `advanced_execution.py`, `broker_adapter.py`, `order_validation_pipeline.py`
- **Risk**: `risk_calculations.py`, `risk_metrics.py`, `risk_runtime.py`
- **Liquidity**: `liquidity_service.py`, `liquidity_metrics_tracker.py`, `depth_analyzer.py`
- **Position Management**: `position_valuer.py`, `portfolio_valuation.py`, `equity_calculator.py`
- **Fees & Margin**: `fees_engine.py`, `margin_calculator.py`
- **Policy**: `order_policy.py`, `policy_validator.py`
- **Strategies**: Multiple strategy implementations in `strategies/` subdir

**Concerns**:
- Large number of files suggests potential over-engineering or lack of cohesion
- No clear module groupings beyond subdirs
- Candidate for Phase 1 structural review and potential modularization

**Assessment**: Functional but needs structural review in Phase 1

---

#### `features/brokerages/`
**Status**: ACTIVE, PARTIAL IMPLEMENTATION ⚠️

**Coinbase** (`brokerages/coinbase/`):
- **Files**: 10+ modules + client/ and rest/ subdirs
- **Purpose**: Coinbase Advanced Trade API v3 + Legacy Exchange API
- **Status**: "Critical Fixes Applied ⚠️" - Basic operations functional, 20+ endpoints pending routing
- **Has README**: Yes, comprehensive
- **Key gaps**:
  - Endpoint routing incomplete
  - Paper engine decoupling in progress
  - Comprehensive test coverage pending

**Core** (`brokerages/core/`):
- **Purpose**: Shared brokerage interfaces and abstractions
- **Status**: ACTIVE
- **Files**: `interfaces.py` defines `IBrokerage` contract

**Assessment**: Functional for basic use, needs completion before production scale-up

---

#### `features/optimize/`
**Status**: ACTIVE
**Files**: `backtester.py`, `optimize.py`, `strategies.py`, `types.py`
**Purpose**: Strategy backtesting and optimization
**Assessment**: Core functionality present, standard module

---

#### `features/analyze/`
**Status**: ACTIVE
**Files**: `analyze.py`, `indicators.py`, `patterns.py`, `strategies.py`, `types.py`
**Purpose**: Technical analysis, pattern detection
**Assessment**: Standard feature, no concerns

---

#### `features/data/`
**Status**: ACTIVE
**Files**: `data.py`, `cache.py`, `storage.py`, `quality.py`, `types.py`
**Purpose**: Data management, caching, quality control
**Assessment**: Clean, well-scoped

---

#### `features/position_sizing/`
**Status**: ACTIVE
**Files**: `position_sizing.py`, `kelly.py`, `confidence.py`, `regime.py`, `types.py`
**Purpose**: Position size calculation (Kelly Criterion, regime-based, confidence-weighted)
**Assessment**: Well-structured, focused responsibility

---

#### `features/strategies/`
**Status**: ACTIVE
**Files**: 8 strategy modules (breakout, ma_crossover, mean_reversion, momentum, scalp, volatility, etc.)
**Purpose**: Trading strategy implementations
**Assessment**: Good separation of concerns

---

#### `features/strategy_tools/`
**Status**: ACTIVE
**Files**: `enhancements.py`, `filters.py`, `guards.py`
**Purpose**: Cross-cutting strategy utilities
**Assessment**: Clean separation from strategy implementations

---

### ✅ Infrastructure - Fully Active

#### `orchestration/`
**Status**: ACTIVE, CORE INFRASTRUCTURE
**Files**: 26+ modules + execution/ subdir
**Purpose**: System bootstrap, service coordination, lifecycle management

**Key components**:
- `bootstrap.py` - System initialization
- `broker_factory.py` - Brokerage instantiation
- `runtime_coordinator.py` - Runtime orchestration
- `execution_coordinator.py` - Execution flow coordination
- `strategy_orchestrator.py` - Strategy management
- `service_registry.py` - Dependency injection
- `perps_bot.py`, `perps_bot_builder.py` - Main bot entry points
- `market_data_service.py`, `streaming_service.py` - Data coordination
- `guardrails.py`, `risk_gate_validator.py` - Safety checks

**Assessment**: Critical infrastructure, well-organized

---

#### `state/`
**Status**: ACTIVE, WELL-STRUCTURED
**Files**: 12+ modules across 5 subdirs
**Subdirs**: backup/, checkpoint/, recovery/, repositories/, utils/

**Key components**:
- `state_manager.py` - Central state coordination
- `backup_manager.py` - State backup orchestration
- `backup/services/` - Backup implementations
- `checkpoint/` - Checkpointing logic
- `recovery/handlers/` - Recovery handlers
- `repositories/` - Data persistence abstractions
- `batch_operations.py` - Batch state operations (recently extracted from state_manager)

**Assessment**: Clean separation, recent refactoring improvements evident

---

#### `monitoring/`
**Status**: ACTIVE
**Files**: 10+ modules across 3 subdirs
**Subdirs**: domain/, runtime_guards/, system/

**Key components**:
- `metrics_collector.py`, `metrics_server.py` - Prometheus integration
- `alerts_manager.py`, `alerts.py` - Alerting system
- `workflow_tracker.py` - Execution tracking
- `domain/perps/` - Domain-specific monitoring
- `runtime_guards/` - Runtime safety checks
- `system/` - System-level monitoring

**Assessment**: Comprehensive monitoring infrastructure

---

#### `cli/`
**Status**: ACTIVE
**Files**: 2 subdirs (commands/, handlers/)

**Commands available**:
- `run.py` - Main bot execution
- `orders.py` - Order management
- `account.py` - Account operations
- `move_funds.py` - Fund transfers
- `convert.py` - Currency conversion
- `signal_manager.py` - Signal management

**Entry points** (from pyproject.toml):
- `gpt-trader` → `bot_v2.cli:main`
- `perps-bot` → `bot_v2.cli:main`

**Assessment**: Full-featured CLI, active development

---

#### Supporting Modules (All ACTIVE)
- `config/` - Configuration management
- `security/` - Auth, secrets, validation
- `types/` - Shared type definitions
- `logging/` - Logging infrastructure
- `validation/` - Input validation
- `persistence/` - Data persistence
- `data_providers/` - External data sources
- `errors/` - Error definitions
- `utilities/` - General utilities

---

## Archived/Retired Code

### 📦 `archived/monitoring_2025_09_29/`
**Contents**: `alerting_system.py`
**Status**: RETIRED
**Retirement Date**: ~2025-09-29
**Reason**: Replaced by current `monitoring/` implementation
**Action**: Retain for historical reference, no migration needed

---

### 📦 `config/archive/`
**Contents**: README.md only (no actual config files)
**Status**: EMPTY ARCHIVE
**Notes**: Stage 1/2 scale-up manifests moved to `config/stage1_scaleup.yaml` and `config/stage2_scaleup.yaml`
**Action**: Keep README for historical reference

---

### 📦 `docs/archive/refactoring-2025-q1/`
**Contents**: 46+ markdown files documenting completed refactoring phases
**Status**: HISTORICAL DOCUMENTATION
**Coverage**:
- Phase 0-3 completion summaries
- CLI feature implementation (orders, move_funds, paper_trade dashboard)
- Session notes from 2025-10-02

**Action**: Retain as project history, valuable for understanding past decisions

---

### 📦 `docs/archive/legacy-deployment/`
**Contents**: Old deployment documentation
**Status**: RETIRED
**Action**: Verify current deployment docs are up-to-date in `deploy/` or `docs/operations/`

---

## Code Quality Observations

### Positive Signals ✅
1. **Vertical slice architecture**: Clear feature boundaries, minimal coupling
2. **README documentation**: Key features (adaptive_portfolio, coinbase) have comprehensive docs
3. **Recent refactoring**: Evidence of ongoing cleanup (e.g., batch_operations.py extracted from state_manager)
4. **Type safety**: Consistent use of `types.py` modules per feature
5. **Test infrastructure**: pytest, coverage, benchmarks configured

### Concerns ⚠️

1. **No inline code markers**: Zero TODO/FIXME/DEPRECATED comments found
   - Could indicate:
     - Very disciplined development (good)
     - Issue tracking moved entirely to external system (check GitHub issues)
     - Technical debt not being captured inline (risky)
   - **Action**: Review issue tracker, consider adding inline markers for discoverability

2. **Coinbase integration incomplete**: 20+ endpoints pending
   - Blocks full production readiness
   - **Action**: Prioritize endpoint routing completion in Phase 1

3. **Live trade complexity**: 33+ files without clear module hierarchy
   - Potential signs of feature creep or insufficient abstraction
   - **Action**: Phase 1 structural review - candidate for submodule grouping or extraction

4. **Legacy code references**: Several files mention "legacy" in comments
   - Example: `features/brokerages/core/interfaces.py` references `legacy_quantity`
   - **Action**: Phase 0 - catalog these references, assess if backward compat still needed

---

## Retirement Candidates (None Identified)

**Current assessment**: No active code identified as candidate for retirement.

All modules in `src/bot_v2/` appear to be in use or recently developed. The codebase is relatively young (most activity 2024-2025 based on git log), with recent refactoring already moving obsolete code to `archived/`.

**Monitoring for future phases**:
- If `live_trade/` refactoring results in extracted modules, older implementations may become candidates
- Any broker adapters for non-Coinbase brokers (if added and later deprecated)

---

## Recommended Actions

### Phase 0 (Current)
1. ✅ Complete this inventory document
2. ⏳ Run tooling audit (poetry check, ruff, mypy, pytest) - see separate log
3. ⏳ Document dependency tree and policy
4. ⏳ Config drift scan

### Phase 1 (Next)
1. **Coinbase integration**: Complete 20+ pending endpoint routing implementations
2. **Live trade review**: Structural analysis of `features/live_trade/` - identify opportunities for:
   - Submodule grouping (e.g., `features/live_trade/liquidity/`, `features/live_trade/fees/`)
   - Potential extraction to separate features
   - Interface documentation (README needed)
3. **Legacy reference audit**: Document all "legacy" code mentions, assess backward compatibility needs
4. **Inline markers**: Establish policy for TODO/FIXME usage and ensure team alignment

### Phase 2+
1. Monitor `live_trade/` refactoring outcomes for new archive candidates
2. Quarterly review of archived code - consider permanent deletion of items older than 12 months
3. Keep this document updated as slices evolve

---

## Appendix: Directory Metrics

```
src/bot_v2/
├── cli/                    (2 subdirs, ~16 command modules)
├── config/                 (4 modules)
├── data_providers/         (TBD - not inventoried in detail)
├── errors/                 (Error definitions)
├── features/               (10 feature slices)
│   ├── adaptive_portfolio/ (13 modules + subdir)
│   ├── analyze/           (5 modules)
│   ├── brokerages/        (2 subdirs: coinbase, core)
│   │   ├── coinbase/      (10+ modules + 2 subdirs) ⚠️
│   │   └── core/          (1 module: interfaces.py)
│   ├── data/              (5 modules)
│   ├── live_trade/        (33+ modules + 3 subdirs) ⚠️
│   ├── optimize/          (4 modules)
│   ├── paper_trade/       (12 modules + dashboard subdir)
│   ├── position_sizing/   (6 modules)
│   ├── strategies/        (8 strategy modules)
│   └── strategy_tools/    (3 modules)
├── logging/               (Logging infrastructure)
├── monitoring/            (10+ modules, 3 subdirs)
├── orchestration/         (26+ modules + execution subdir)
├── persistence/           (Persistence layer)
├── scripts/               (V2-native utility scripts)
├── security/              (4 modules: auth, secrets, validation)
├── state/                 (12+ modules, 5 subdirs)
├── types/                 (Shared types)
├── utilities/             (General utilities)
└── validation/            (Input validation)

Total Python files: 316
```

---

**Next Steps**: Run tooling audit (poetry, ruff, mypy, pytest) to establish quality baseline.
