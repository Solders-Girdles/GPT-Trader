# Orchestration Refactoring - Phase 3 Plan

**Status**: Planning
**Created**: 2025-10-05
**Prerequisites**: Phase 2 complete and stable in production

---

## Phase 2 Results (Baseline)

- **Modules**: 40 → 31 (-22%)
- **Lines**: 8,546 → 7,382 (-14%)
- **Circular Dependencies**: 7 → 0 ✅
- **Tests**: 5,235 passing ✅

**Key Achievements**:
- Eliminated all circular dependencies
- Extracted domain modules to feature packages
- Established clean package boundaries
- Zero production incidents

---

## Phase 3 Objectives

### Primary Goal
**Consolidate core orchestration** by completing domain extractions and organizing remaining coordination logic into clear subpackages.

### Success Metrics
- Target: **25-28 modules** (20-30% additional reduction)
- Maintain: Zero circular dependencies
- Ensure: All tests passing, no production regressions
- Improve: Clear separation of concerns in orchestration core

---

## Tier 3 Extractions (Domain Modules)

### 1. streaming_service (476 lines, 0 orchestration imports)
**Priority**: HIGH - Largest extractable module with zero dependencies

**Plan**:
- Extract to: `features/streaming/websocket_service.py`
- Update: `perps_bot.py`, `perps_bot_builder.py` imports
- Tests: Move to `tests/unit/bot_v2/features/streaming/`
- Risk: **LOW** - Zero orchestration coupling

**Impact**: 31 → 30 modules (-3%)

### 2. guardrails (241 lines, 2 orchestration imports)
**Priority**: MEDIUM - Trading safety logic

**Plan**:
- Extract to: `features/live_trade/guardrails/trading_gates.py`
- Dependencies: Uses `configuration.BotConfig`, `deterministic_broker`
- Strategy: Extract alongside session_guard for cohesive risk package
- Risk: **MEDIUM** - Tightly coupled to trading flow

**Impact**: 30 → 29 modules (-3%)

### 3. system_monitor (284 lines, 3 orchestration imports)
**Priority**: MEDIUM - Complete monitoring migration

**Plan**:
- Extract to: `monitoring/system/health_monitor.py`
- Dependencies: Uses `perps_bot`, `configuration`, `service_registry`
- Strategy: Break dependencies via protocol/interface injection
- Risk: **MEDIUM** - Runtime health monitoring is critical

**Impact**: 29 → 28 modules (-3%)

**Tier 3 Total Impact**: 31 → 28 modules (-10%)

---

## Core Orchestration Reorganization

### Current Structure Issues
```
orchestration/
├── configuration.py           # 473 lines - config + 4 classes
├── perps_bot.py              # 390 lines - main runtime
├── live_execution.py         # 402 lines - execution logic
├── builders/
│   └── perps_bot_builder.py  # 379 lines - construction
├── execution/                # 6 modules - execution pipeline
└── [22 other modules]        # Various coordination logic
```

### Proposed Phase 3 Structure
```
orchestration/
├── core/                     # Core runtime (NEW)
│   ├── runtime.py           # PerpsBot main runtime
│   ├── lifecycle.py         # Lifecycle management
│   ├── interfaces.py        # IBotRuntime protocol
│   └── coordinator.py       # RuntimeCoordinator
├── config/                   # Configuration (NEW)
│   ├── models.py            # BotConfig, Profile, etc.
│   ├── controller.py        # ConfigController
│   └── validation.py        # Config validation
├── builders/                 # Construction (existing)
│   └── bot_builder.py       # Simplified builder
├── execution/               # Execution pipeline (existing)
│   ├── coordinator.py
│   ├── guards.py
│   ├── order_placement.py
│   └── validation.py
├── strategy/                 # Strategy orchestration (NEW)
│   ├── orchestrator.py      # StrategyOrchestrator
│   ├── executor.py          # StrategyExecutor
│   └── registry.py          # StrategyRegistry
└── services/                 # Service management (NEW)
    ├── registry.py          # ServiceRegistry
    └── lifecycle_service.py # Lifecycle coordination
```

### Reorganization Steps

1. **Create core/ package**
   - Move `perps_bot.py` → `core/runtime.py`
   - Move `runtime_coordinator.py` → `core/coordinator.py`
   - Move `lifecycle_service.py` → `core/lifecycle.py`
   - Move `core/interfaces.py` (already exists)

2. **Create config/ package**
   - Split `configuration.py`:
     - Config models → `config/models.py`
     - Validation → `config/validation.py`
   - Move `config_controller.py` → `config/controller.py`

3. **Create strategy/ package**
   - Move `strategy_orchestrator.py` → `strategy/orchestrator.py`
   - Move `strategy_executor.py` → `strategy/executor.py`
   - Move `strategy_registry.py` → `strategy/registry.py`

4. **Create services/ package**
   - Move `service_registry.py` → `services/registry.py`
   - Move `lifecycle_service.py` → `services/lifecycle.py`

**Impact**: Better organization, no module count reduction (internal restructuring)

---

## Risk Assessment

### High Risk Areas

1. **PerpsBot Runtime** (`perps_bot.py`)
   - Central to all trading operations
   - Heavy coupling to many modules
   - Mitigation: Protocol-based refactoring, extensive testing

2. **Configuration Split** (`configuration.py`)
   - 473 lines, 4 classes, widely imported
   - Breaking into submodules requires careful coordination
   - Mitigation: Maintain backward-compatible re-exports

3. **Live Execution** (`live_execution.py`)
   - 402 lines, critical path for order execution
   - Complex state management and error handling
   - Mitigation: Extract incrementally, comprehensive test coverage

### Medium Risk Areas

4. **Builder Complexity** (`perps_bot_builder.py`)
   - 379 lines, 18 imports
   - Heavy orchestration of dependency injection
   - Mitigation: Simplify via factory patterns

5. **Execution Pipeline** (6 modules)
   - Already well-structured
   - Minor reorganization risk
   - Mitigation: Move as atomic group

### Low Risk Areas

6. **Deterministic Broker** (357 lines)
   - Self-contained, clear boundaries
   - Could extract to `features/testing/`
   - Low coupling to orchestration

---

## Validation Strategy

### Pre-Extraction Validation
- [ ] Analyze import graph for each target module
- [ ] Identify all reverse dependencies
- [ ] Create interface contracts where needed
- [ ] Plan test migration strategy

### During Extraction
- [ ] Run orchestration analyzer after each move
- [ ] Verify zero circular dependencies maintained
- [ ] Run full test suite (5,235+ tests)
- [ ] Run integration/streaming tests
- [ ] Check characterization tests pass

### Post-Extraction Validation
- [ ] Run paper trading session (24hr minimum)
- [ ] Monitor error rates in production
- [ ] Validate execution latency unchanged
- [ ] Confirm no regression in trade execution

### Production Rollout
- [ ] Deploy to staging environment
- [ ] Run production replay scenarios
- [ ] Monitor for 1 week minimum
- [ ] Gradual rollout with feature flags
- [ ] Rollback plan: git tags for each extraction

---

## Timeline Estimate

### Tier 3 Extractions (3-4 weeks)
- Week 1: `streaming_service` extraction + validation
- Week 2: `guardrails` extraction + validation
- Week 3: `system_monitor` extraction + validation
- Week 4: Integration testing + production bake

### Core Reorganization (4-6 weeks)
- Weeks 1-2: config/ and core/ package creation
- Weeks 3-4: strategy/ and services/ package creation
- Weeks 5-6: Integration testing + production rollout

**Total Phase 3 Timeline**: 7-10 weeks

---

## Prerequisites

### Before Starting Phase 3

1. **Phase 2 Production Stability** ✅
   - [ ] No incidents related to Phase 2 changes (2+ weeks)
   - [ ] Monitoring confirms zero regressions
   - [ ] Team comfortable with new structure

2. **Documentation Complete** ✅
   - [ ] README.md updated with new architecture
   - [ ] QUICK_START.md reflects feature packages
   - [ ] Architecture diagrams updated
   - [ ] Developer onboarding guide refreshed

3. **Testing Infrastructure** ✅
   - [ ] Streaming integration tests in CI
   - [ ] Characterization tests automated
   - [ ] Performance benchmarks established
   - [ ] Rollback procedures documented

4. **Team Alignment** ✅
   - [ ] Phase 3 plan reviewed and approved
   - [ ] Risk assessment discussed
   - [ ] Timeline agreed upon
   - [ ] Resource allocation confirmed

---

## Success Criteria

### Technical
- ✅ 25-28 orchestration modules (from 31)
- ✅ Zero circular dependencies maintained
- ✅ All tests passing (5,235+)
- ✅ No production regressions
- ✅ Execution latency unchanged
- ✅ Clear package organization

### Process
- ✅ Each extraction tagged for rollback
- ✅ Documentation updated in lockstep
- ✅ Code review for each major change
- ✅ Production monitoring for 1+ week between phases

### Outcome
- ✅ **30% total reduction** from Phase 0 baseline (40 → 28 modules)
- ✅ Clean separation: core/config/strategy/execution/services
- ✅ Feature packages complete: streaming, guardrails, monitoring
- ✅ Maintainable architecture for future growth

---

## Open Questions

1. **streaming_service extraction timing**
   - Extract immediately after Phase 2 stabilization?
   - Or bundle with guardrails for cohesive release?

2. **Configuration split strategy**
   - Maintain backward compatibility with re-exports?
   - Or force migration to new imports?

3. **PerpsBot refactoring approach**
   - Extract functionality to separate classes first?
   - Or reorganize in-place then move?

4. **Testing requirements**
   - What's the minimum production bake time?
   - Should we require load testing?

---

## Next Steps

1. **Monitor Phase 2 in production** (2+ weeks)
2. **Update top-level documentation**
3. **Get team sign-off on Phase 3 plan**
4. **Schedule Phase 3 kickoff** when prerequisites met
5. **Begin with lowest-risk extraction** (streaming_service)

---

**Last Updated**: 2025-10-05
**Owner**: Orchestration Refactoring Team
**Review Cycle**: Every 2 weeks during production monitoring
