# Orchestration Refactoring Progress Tracker

**Started**: 2025-10-05
**Branch**: `refactor/orchestration-phase1`
**Reference Plan**: `docs/architecture/orchestration_refactor.md`

---

## Phase 1: Break Circular Dependencies (Week 1)

**Goal**: Eliminate all 7 circular dependencies involving `perps_bot`

**Target Completion**: End of Week 1

### Priority 1: Decouple perps_bot ✅ IN PROGRESS

**Problem**: `perps_bot.py` has circular imports with 5 modules

**Circular Dependencies**:
- [ ] `perps_bot` ↔️ `perps_bot_builder`
- [x] `perps_bot` ↔️ `system_monitor` ✅
- [x] `perps_bot` ↔️ `lifecycle_service` ✅
- [x] `perps_bot` ↔️ `runtime_coordinator` ✅
- [x] `perps_bot` ↔️ `strategy_orchestrator` ✅
- [x] `perps_bot` ↔️ `execution_coordinator` ✅

**Solution**: Introduce dependency injection via `IBotRuntime` protocol

#### Step 1.1: Extract IBotRuntime Interface ✅ COMPLETE

**File**: `orchestration/core/bot_interface.py` (NEW)

**Tasks**:
- [x] Create `orchestration/core/` directory
- [x] Define `IBotRuntime` protocol with required methods
- [x] Add type stubs for all properties/methods
- [x] Document interface contract in docstrings
- [x] Add unit tests for protocol compliance

**Interface Methods** (from refactor plan):
```python
@property
def config(self) -> BotConfig: ...

@property
def is_running(self) -> bool: ...

def stop(self) -> None: ...
```

**Verification**:
- [ ] mypy passes with new interface
- [ ] Protocol tests pass
- [ ] No circular import errors in `core/`

#### Step 1.2: Update Dependent Services ✅ COMPLETE

**Files Modified**:
- [x] `system_monitor.py` - Use `IBotRuntime` instead of `PerpsBot` ✅
- [x] `lifecycle_service.py` - Use `IBotRuntime` instead of `PerpsBot` ✅
- [x] `runtime_coordinator.py` - Use `IBotRuntime` instead of `PerpsBot` ✅
- [x] `strategy_orchestrator.py` - Use `IBotRuntime` instead of `PerpsBot` ✅
- [x] `execution_coordinator.py` - Use `IBotRuntime` instead of `PerpsBot` ✅
- [x] `perps_bot_builder.py` - Use `IBotRuntime` instead of `PerpsBot` ✅

**Pattern**:
```python
# Before:
from bot_v2.orchestration.perps_bot import PerpsBot
class SomeService:
    def __init__(self, bot: PerpsBot): ...

# After:
from bot_v2.orchestration.core.bot_interface import IBotRuntime
class SomeService:
    def __init__(self, bot: IBotRuntime): ...
```

**Verification**:
- [x] No import of `perps_bot` module in services ✅
- [x] Type hints use `IBotRuntime` ✅
- [x] All existing tests still pass (632/632) ✅
- [x] ruff --fix applied for import order ✅
- [x] Circular dependencies reduced from 7 to 2 ✅

#### Step 1.3: Break perps_bot ↔️ perps_bot_builder

**File**: `perps_bot_builder.py`

**Tasks**:
- [ ] Move builder to `orchestration/builders/` directory
- [ ] Update builder to import `IBotRuntime` only
- [ ] Use forward references for `PerpsBot` type hints
- [ ] Update all builder methods to use interface

**Verification**:
- [ ] Builder imports only interface types
- [ ] `perps_bot.py` can import builder without circular dep
- [ ] Builder tests pass with new location

**Success Criteria for Priority 1**:
- [ ] Zero circular dependencies involving `perps_bot`
- [ ] All tests passing
- [ ] mypy clean (no type errors)
- [ ] Circular dependency analyzer shows 0 cycles

---

### Priority 2: Break configuration ↔️ symbols

**Problem**: `configuration.py` and `symbols.py` import each other

**Solution**: Extract symbol utilities to shared module

**Tasks**:
- [ ] Create `src/bot_v2/shared/symbol_utils.py`
- [ ] Move `normalize_symbol()` from `symbols.py`
- [ ] Move `validate_symbol()` from `symbols.py`
- [ ] Update `configuration.py` to import from `shared/symbol_utils`
- [ ] Update `symbols.py` to import from `shared/symbol_utils`
- [ ] Remove circular imports

**Files to Modify**:
- [ ] Create `src/bot_v2/shared/symbol_utils.py`
- [ ] Update `orchestration/configuration.py`
- [ ] Update `orchestration/symbols.py`

**Verification**:
- [ ] No circular import between `configuration` and `symbols`
- [ ] All symbol utilities still work
- [ ] Tests pass
- [ ] Circular dependency analyzer confirms fix

**Success Criteria for Priority 2**:
- [ ] Zero circular dependencies in entire orchestration layer
- [ ] Symbol utilities work from shared location
- [ ] All tests passing

---

## Verification Steps

### After Each Step

1. **Run circular dependency analyzer**:
   ```bash
   poetry run python scripts/analysis/orchestration_analyzer.py
   ```
   Expected: Circular dependency count decreases

2. **Run affected tests**:
   ```bash
   poetry run pytest tests/unit/bot_v2/orchestration/ -v
   poetry run pytest tests/integration/perps_bot_characterization/ -v
   ```
   Expected: All tests pass

3. **Run type checker**:
   ```bash
   poetry run mypy src/bot_v2/orchestration/
   ```
   Expected: No new type errors (existing errors OK for now)

4. **Run scenario tests**:
   ```bash
   poetry run pytest tests/integration/scenarios/ -m scenario -v
   ```
   Expected: All tests pass or skip (no failures)

### Final Phase 1 Verification

1. **Zero circular dependencies**:
   ```bash
   poetry run python scripts/analysis/orchestration_analyzer.py | grep "Circular Dependencies"
   ```
   Expected output: "✅ No circular dependencies detected"

2. **All tests passing**:
   ```bash
   poetry run pytest tests/ -m "not real_api" -v
   ```
   Expected: 5189+ passing (may increase with new tests)

3. **No regressions**:
   ```bash
   poetry run pytest tests/integration/scenarios/ -m scenario -v
   ```
   Expected: All scenario tests pass or skip

4. **Type safety maintained**:
   ```bash
   poetry run mypy src/bot_v2/orchestration/
   ```
   Expected: No increase in error count from baseline

---

## Phase 2: Extract Domain Modules (Week 2)

**Status**: Not started

**Planned Extractions**:
- [ ] Market Data → `features/market_data/`
- [ ] Streaming → `features/streaming/`
- [ ] Account Telemetry → `monitoring/telemetry/`
- [ ] Equity Calculator → `features/live_trade/equity/`

See `orchestration_refactor.md` for detailed plan.

---

## Phase 3: Split Large Modules (Week 3)

**Status**: Not started

**Planned Splits**:
- [ ] `configuration.py` (485 lines → 3 files)
- [ ] `perps_bot_builder.py` (380 lines → 2 files)
- [ ] `live_execution.py` (402 lines → 2 files)

See `orchestration_refactor.md` for detailed plan.

---

## Phase 4: Reorganize orchestration/ (Week 4)

**Status**: Not started

**Planned Structure**:
```
orchestration/
├── core/          # Core orchestration
├── config/        # Configuration
├── services/      # Service management
├── builders/      # Bot builders
├── execution/     # Execution layer
├── strategy/      # Strategy orchestration
└── runtime/       # Runtime management
```

See `orchestration_refactor.md` for detailed plan.

---

## Migration to Shared DTOs

**Ongoing**: Migrate dict usage to shared DTOs as we touch files

**Progress**:
- [ ] `perps_bot.py` - Migrate to TradingSignal, OrderRequest
- [ ] `strategy_orchestrator.py` - Use shared types
- [ ] `execution_coordinator.py` - Use OrderRequest, OrderResult
- [ ] Other files as encountered

**Reference**: `src/bot_v2/shared/types.py` for available DTOs

---

## Metrics

### Baseline (Start of Phase 1)

```
Modules: 36
Lines: 7,810
Circular Dependencies: 7
Modules >300 lines: 5
```

### Current (After Step 1.2)

```
Modules: 37
Lines: ~7,810
Circular Dependencies: 2 (down from 7) ✅
  - configuration ↔️ symbols
  - perps_bot ↔️ perps_bot_builder
Modules >300 lines: 5
Unit Tests Passing: 632/632 ✅
```

### Target (End of Phase 1)

```
Modules: 36 (no change yet)
Lines: 7,810 (no change yet)
Circular Dependencies: 0 ✅
Modules >300 lines: 5 (will address in Phase 3)
```

---

## Notes and Blockers

### Decisions Made

*None yet - track decisions here as refactoring progresses*

### Blockers

*None yet - track blockers here*

### Questions for Review

*Track questions that need team input*

---

## Rollback Plan

If refactoring causes issues:

1. **Revert to baseline**: `git checkout cleanup/legacy-files`
2. **Cherry-pick fixes**: Extract non-breaking changes
3. **Re-plan**: Adjust approach based on issues encountered

**Backup commits**: Tag each major step for easy rollback
```bash
git tag phase1-step1-interface-extracted
git tag phase1-step2-services-updated
git tag phase1-complete
```

---

**Last Updated**: 2025-10-05
**Status**: Phase 1 Priority 1 - Step 1.2 Complete (5/6 circular deps eliminated)
**Next Milestone**: Step 1.3 - Break perps_bot ↔️ perps_bot_builder circular dependency
