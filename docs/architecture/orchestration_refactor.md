# Orchestration Refactoring Plan

**Generated**: 2025-10-05
**Purpose**: Detailed refactoring roadmap for orchestration layer
**Based on**: `orchestration_analysis.md` (automated dependency analysis)

---

## Executive Summary

**Current State**: 36 modules, 7,810 lines, 7 circular dependencies
**Goal**: Reduce complexity, break circular dependencies, extract domain modules
**Estimated Effort**: 3-4 weeks (Phase 1 of structural hardening)

### Key Issues Identified

1. **🔴 Circular Dependencies** (7 detected)
   - `perps_bot` is central to 5 circular dependencies
   - Prevents clean module boundaries and testing

2. **🟡 Large Modules** (5 modules >300 lines)
   - `configuration.py` (485 lines) - needs splitting
   - `streaming_service.py` (477 lines) - extract to feature
   - `live_execution.py` (402 lines) - refactor execution logic

3. **🟢 Extraction Opportunities** (6 modules)
   - Market data, streaming, telemetry - low coupling, ready to extract

---

## Phase 1: Break Circular Dependencies (Week 1)

### Priority 1: Decouple perps_bot

**Problem**: `perps_bot.py` has circular imports with 5 modules:
- `perps_bot` ↔️ `perps_bot_builder`
- `perps_bot` ↔️ `system_monitor`
- `perps_bot` ↔️ `lifecycle_service`
- `perps_bot` ↔️ `runtime_coordinator`
- `perps_bot` ↔️ `strategy_orchestrator`
- `perps_bot` ↔️ `execution_coordinator`

**Root Cause**: `PerpsBot` is both a service container and orchestrator (violates SRP)

**Solution**: Introduce dependency injection via interfaces

#### Step 1.1: Extract PerpsBot Interface
```python
# orchestration/core/bot_interface.py (NEW)
from typing import Protocol

class IBotRuntime(Protocol):
    """Bot runtime interface for injecting into services."""

    @property
    def config(self) -> BotConfig: ...

    @property
    def is_running(self) -> bool: ...

    def stop(self) -> None: ...
```

#### Step 1.2: Update Dependent Services
```python
# Before (circular):
from bot_v2.orchestration.perps_bot import PerpsBot

class SystemMonitor:
    def __init__(self, bot: PerpsBot):  # Circular!
        self.bot = bot

# After (interface):
from bot_v2.orchestration.core.bot_interface import IBotRuntime

class SystemMonitor:
    def __init__(self, bot: IBotRuntime):  # No circular dep
        self.bot = bot
```

#### Step 1.3: Break perps_bot ↔️ perps_bot_builder
- Move builder to `orchestration/builders/`
- Builder should only import interfaces, not concrete `PerpsBot`
- Use forward references for type hints

**Files to Modify**:
- `perps_bot.py` - Extract interface
- `perps_bot_builder.py` - Use interface
- `system_monitor.py` - Use interface
- `lifecycle_service.py` - Use interface
- `runtime_coordinator.py` - Use interface
- `strategy_orchestrator.py` - Use interface
- `execution_coordinator.py` - Use interface

**Success Criteria**: Zero circular dependencies involving `perps_bot`

---

### Priority 2: Break configuration ↔️ symbols

**Problem**: `configuration` and `symbols` import each other

**Analysis**:
- `configuration.py` imports symbol utilities
- `symbols.py` imports configuration for validation

**Solution**: Extract symbol utilities to shared module

```python
# bot_v2/shared/symbol_utils.py (NEW)
def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format (move from symbols.py)."""
    ...

def validate_symbol(symbol: str) -> bool:
    """Validate symbol format (move from symbols.py)."""
    ...
```

Update imports:
```python
# configuration.py
from bot_v2.shared.symbol_utils import normalize_symbol  # No circular dep

# symbols.py
from bot_v2.shared.symbol_utils import validate_symbol
# Remove import of configuration
```

**Files to Modify**:
- Create `src/bot_v2/shared/symbol_utils.py`
- Update `configuration.py`
- Update `symbols.py`

---

## Phase 2: Extract Domain Modules (Week 2)

### Extract 1: Market Data → features/market_data/

**Modules to Extract**:
- `market_data_service.py` (131 lines, 0 orch deps) ✅
- `market_monitor.py` (68 lines, 0 orch deps) ✅
- `symbols.py` (88 lines, 3 orch deps after fixing circular)

**New Structure**:
```
features/market_data/
├── __init__.py
├── service.py          # from market_data_service.py
├── monitor.py          # from market_monitor.py
├── symbols.py          # from symbols.py (after cleanup)
└── README.md
```

**Migration Steps**:
1. Create `features/market_data/` directory
2. Move files and update imports
3. Update orchestration modules to import from new location
4. Add feature README documenting interface contract

**Impact**: Reduces orchestration from 36 to 33 modules

---

### Extract 2: Streaming → features/streaming/

**Modules to Extract**:
- `streaming_service.py` (477 lines, 2 orch deps) ⚠️ Large module

**Dependencies to Resolve**:
- Imports `market_data_service` (will be in features/market_data)
- Imports `orchestration.types` (can stay)

**New Structure**:
```
features/streaming/
├── __init__.py
├── service.py          # Main streaming service
├── websocket.py        # WebSocket handling
├── fallback.py         # REST fallback logic
└── README.md
```

**Refactoring Opportunity**: Split 477-line module while extracting
- `service.py` - Core streaming coordination (200 lines)
- `websocket.py` - WS connection management (150 lines)
- `fallback.py` - REST fallback when WS fails (100 lines)

**Migration Steps**:
1. Split `streaming_service.py` into 3 modules
2. Move to `features/streaming/`
3. Update imports in orchestration
4. Add tests for each extracted module

**Impact**: Reduces orchestration by 477 lines, improves streaming module structure

---

### Extract 3: Account Telemetry → monitoring/telemetry/

**Modules to Extract**:
- `account_telemetry.py` (102 lines, 0 orch deps) ✅

**New Location**: `monitoring/telemetry/account.py`
(Already have `monitoring/` dir, add `telemetry/` subdir)

**Rationale**: Telemetry belongs with monitoring, not orchestration

**Migration Steps**:
1. Create `monitoring/telemetry/` directory
2. Move `account_telemetry.py` → `monitoring/telemetry/account.py`
3. Update imports in orchestration (3 modules import this)

**Impact**: Reduces orchestration by 1 module, improves monitoring structure

---

### Extract 4: Equity Calculator → features/live_trade/equity/

**Modules to Extract**:
- `equity_calculator.py` (117 lines, 0 orch deps) ✅

**New Location**: `features/live_trade/equity_calculator.py`

**Rationale**: Equity calculation is live trade domain logic

**Migration Steps**:
1. Move to `features/live_trade/`
2. Update imports (used by orchestration for portfolio valuation)
3. May need to create `EquityService` interface

**Impact**: Reduces orchestration by 1 module, consolidates live trade logic

---

## Phase 3: Split Large Modules (Week 3)

### Split 1: configuration.py (485 lines → 3 files)

**Current Classes** (4 total):
- `BotConfig` - Main configuration dataclass
- `Profile` - Profile enum
- `ConfigManager` - Configuration loading/validation
- (Others to be identified)

**Proposed Split**:
```
orchestration/config/
├── __init__.py
├── models.py           # BotConfig, Profile dataclasses
├── manager.py          # ConfigManager
└── validation.py       # Config validation logic
```

**Benefits**:
- Each file < 200 lines
- Clear separation: models vs management vs validation
- Easier to test and maintain

---

### Split 2: perps_bot_builder.py (380 lines → 2 files)

**Current**: Single `PerpsBotBuilder` class with many build methods

**Proposed Split**:
```
orchestration/builders/
├── __init__.py
├── bot_builder.py      # Main builder (200 lines)
└── service_builder.py  # Service initialization (180 lines)
```

**Split Strategy**:
- `bot_builder.py` - Core bot construction, config setup
- `service_builder.py` - Build individual services (market data, streaming, etc.)

**Alternative**: Keep as one file, extract helper functions

---

### Split 3: live_execution.py (402 lines → 2 files)

**Proposed Split**:
```
orchestration/execution/
├── live_engine.py      # Main LiveExecutionEngine (250 lines)
└── helpers.py          # Helper functions (150 lines)
```

**Analysis Needed**: Read file to understand structure before splitting

---

## Phase 4: Reorganize orchestration/ (Week 4)

### Proposed New Structure

```
orchestration/
├── __init__.py
├── core/                       # Core orchestration
│   ├── __init__.py
│   ├── bot.py                 # PerpsBot (from perps_bot.py)
│   ├── bot_interface.py       # IBotRuntime protocol
│   └── bootstrap.py           # System bootstrap
│
├── config/                     # Configuration
│   ├── __init__.py
│   ├── models.py              # BotConfig, Profile
│   ├── manager.py             # ConfigManager (from configuration.py)
│   ├── controller.py          # ConfigController
│   └── validation.py
│
├── services/                   # Service management
│   ├── __init__.py
│   ├── registry.py            # ServiceRegistry
│   ├── rebinding.py           # Service rebinding
│   └── lifecycle.py           # LifecycleService
│
├── builders/                   # Bot builders
│   ├── __init__.py
│   ├── bot_builder.py         # PerpsBotBuilder
│   └── service_builder.py
│
├── execution/                  # Execution layer (existing)
│   ├── __init__.py
│   ├── live_engine.py         # LiveExecutionEngine
│   ├── coordinator.py         # ExecutionCoordinator
│   ├── guards.py
│   ├── validation.py
│   ├── order_placement.py
│   ├── order_submission.py
│   ├── runtime_supervisor.py
│   └── state_collection.py
│
├── strategy/                   # Strategy orchestration
│   ├── __init__.py
│   ├── orchestrator.py        # StrategyOrchestrator
│   ├── executor.py            # StrategyExecutor
│   └── registry.py            # StrategyRegistry
│
├── runtime/                    # Runtime management
│   ├── __init__.py
│   ├── coordinator.py         # RuntimeCoordinator
│   └── session_guard.py       # TradingSessionGuard
│
└── legacy/                     # Deprecated (to be removed)
    ├── deterministic_broker.py
    └── spot_profile_service.py
```

### Migration Strategy

1. **Create new directories** (core/, config/, services/, etc.)
2. **Move files incrementally** (one directory at a time)
3. **Update imports** after each directory migration
4. **Run tests** to ensure nothing breaks
5. **Update documentation** to reflect new structure

---

## Testing Strategy

### During Refactoring

1. **Unit Tests**: Ensure all existing tests pass after each change
2. **Integration Tests**: Add scenario tests for refactored components
3. **Import Tests**: Add tests to detect circular imports

```python
# tests/unit/orchestration/test_no_circular_imports.py
def test_no_circular_imports():
    """Ensure no circular imports in orchestration."""
    import importlib
    import sys

    # Try importing all orchestration modules
    modules = [
        "bot_v2.orchestration.core.bot",
        "bot_v2.orchestration.config.models",
        # ... all modules
    ]

    for module_name in modules:
        # Clear any cached imports
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Import should succeed without circular dependency errors
        importlib.import_module(module_name)
```

### Post-Refactoring

1. **Dependency Graph Tests**: Verify no circular dependencies
2. **Coverage Check**: Ensure coverage doesn't drop
3. **Benchmark Tests**: Ensure performance hasn't degraded

---

## Risk Mitigation

### High Risk Areas

1. **perps_bot.py** - Central to entire system
   - **Mitigation**: Use feature flags to switch between old/new implementation
   - **Rollback Plan**: Keep old version in `orchestration/legacy/`

2. **Circular Dependencies** - Breaking may cause import errors
   - **Mitigation**: Fix one circular dep at a time, test thoroughly
   - **Rollback Plan**: Git branches for each fix

3. **Module Extraction** - May break existing imports
   - **Mitigation**: Use deprecation warnings for old import paths
   - **Rollback Plan**: Symlinks for backward compatibility

### Example Deprecation Warning

```python
# orchestration/market_data_service.py (deprecated location)
import warnings
from bot_v2.features.market_data import MarketDataService

warnings.warn(
    "Importing MarketDataService from orchestration is deprecated. "
    "Use: from bot_v2.features.market_data import MarketDataService",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["MarketDataService"]
```

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Zero circular dependencies
- [ ] All tests passing
- [ ] No regressions in functionality

### Phase 2 Complete When:
- [ ] Market data, streaming, telemetry, equity extracted to appropriate features
- [ ] Orchestration reduced from 36 to <30 modules
- [ ] All imports updated and working

### Phase 3 Complete When:
- [ ] No modules >300 lines
- [ ] Configuration, builder, live_execution split and organized
- [ ] Code coverage maintained or improved

### Phase 4 Complete When:
- [ ] New directory structure implemented
- [ ] All modules in logical groups (core/, config/, services/, etc.)
- [ ] Documentation updated
- [ ] Deprecated modules moved to `legacy/` with removal timeline

---

## Timeline & Ownership

| Phase | Duration | Effort | Dependencies |
|-------|----------|--------|--------------|
| Phase 1: Break Circular Deps | Week 1 | 5-8 days | None |
| Phase 2: Extract Domain Modules | Week 2 | 5-7 days | Phase 1 complete |
| Phase 3: Split Large Modules | Week 3 | 5-7 days | Phase 1 complete |
| Phase 4: Reorganize Structure | Week 4 | 5-7 days | Phases 2&3 complete |

**Ownership**: Trading Team (with code review from Architecture)

**Parallel Work**: Phase 2 and 3 can partially overlap if different developers

---

## Appendix: Detailed Module Analysis

### Circular Dependency Details

```
perps_bot.py (391 lines, 1 class)
├─[imports]─> perps_bot_builder.py
├─[imports]─> system_monitor.py
├─[imports]─> lifecycle_service.py
├─[imports]─> runtime_coordinator.py
├─[imports]─> strategy_orchestrator.py
└─[imports]─> execution_coordinator.py

All of the above import perps_bot.py back (circular!)
```

**Fix**: Introduce `IBotRuntime` protocol, have services depend on interface

### Extraction Candidates (Low Coupling)

| Module | Lines | Orch Imports | Destination |
|--------|-------|--------------|-------------|
| account_telemetry | 102 | 0 | monitoring/telemetry/ |
| equity_calculator | 117 | 0 | features/live_trade/ |
| market_data_service | 131 | 0 | features/market_data/ |
| market_monitor | 68 | 0 | features/market_data/ |
| symbols | 88 | 3 (after fix: 0) | features/market_data/ |
| streaming_service | 477 | 2 | features/streaming/ |

---

**Next Steps**:
1. Review and approve this plan
2. Create feature branches for each phase
3. Start with Phase 1 (break circular dependencies)
