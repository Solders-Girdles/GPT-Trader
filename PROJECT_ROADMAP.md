# GPT-Trader Project Roadmap

**Status**: Active Development
**Date**: 2025-11-29
**Focus**: Architectural Migration & Trading Performance

## Overview

This roadmap guides the completion of GPT-Trader's architectural migration to modern patterns while prioritizing trading performance improvements.

**Guiding Principles:**
1. Complete the architectural migration, don't abort it
2. Focus on foundations (DI) and trading performance (Strategy) first
3. Leverage existing infrastructure (full YAML profile mapping)
4. Stateful strategies with proper state restoration for performance

**Priority Order:** Phase 1 (DI Inversion) → Phase 2 (Strategy Performance) → Phase 3 (Profile System) → Phase 4+ (Infrastructure)

---

## Current State Assessment

### Completed Work
- `RuntimeSettings` deleted - configuration unified
- `BotConfig` is now the single source of truth
- No remaining global state issues
- 3,281 tests use `bot_config_factory` fixture
- CLI uses ApplicationContainer as entry point

### Current DI Pattern State

| Pattern | Status | Usage |
|---------|--------|-------|
| `ServiceRegistry` | Legacy (43 lines) | Integration tests, storage, coordinators |
| `ApplicationContainer` | Modern (173 lines) | CLI entry point, creates ServiceRegistry internally |
| `CoordinatorContext` | Intermediate | Live trading engine |

**Current Issue:** bootstrap.py creates ServiceRegistry first, then stores ApplicationContainer in `registry.extras["container"]`. This inverts the intended dependency direction.

---

## Phase 1: DI Inversion (Complete ApplicationContainer Migration)

**Goal:** Make ApplicationContainer the canonical composition root. ServiceRegistry becomes an internal implementation detail.

### 1.1 Consolidate Bootstrap Functions
**Files:** `src/gpt_trader/orchestration/bootstrap.py`

| Current | Target |
|---------|--------|
| `prepare_bot()` creates ServiceRegistry, stores container in extras | Single `build_bot()` returns TradingBot |
| `build_bot()` returns `(TradingBot, ServiceRegistry)` tuple | Container accessible via `bot.container` |

### 1.2 Update Integration Test Fixtures
**Files:** `tests/integration/conftest.py`

```python
# Current
def dev_bot(...) -> tuple[TradingBot, ServiceRegistry]:

# Target
def dev_bot(...) -> TradingBot:
    # Access container via bot.container
    # Access registry via bot.container.create_service_registry() if needed
```

### 1.3 TradingBot Constructor Simplification
**Files:** `src/gpt_trader/orchestration/trading_bot/bot.py`

- Current: Accepts both `container` and `registry` (optional)
- Target: Require `container`, derive registry internally

### 1.4 Storage Layer Update
**Files:** `src/gpt_trader/orchestration/storage.py`

Update `StorageBootstrapper` to accept `ApplicationContainer` instead of `ServiceRegistry`.

---

## Phase 2: Strategy Performance

**Goal:** Professional-grade stateful strategy execution with O(1) tick processing.

### 2.1 Incremental Statistics Implementation
**Files:** `src/gpt_trader/features/live_trade/` (strategy implementations)

| Current | Target |
|---------|--------|
| MeanReversionStrategy recalculates mean/stddev from scratch on every tick (O(n)) | Welford's algorithm for online mean/variance calculation (O(1)) |

```python
# Welford's algorithm structure
@dataclass
class IncrementalStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def stddev(self) -> float:
        return self.variance ** 0.5
```

### 2.2 Stateful Strategy Interface
**Decision: STATEFUL** - enforce proper state restoration

- Keep and properly implement `rehydrate()` method for state restoration
- Add `serialize_state()` method for checkpoint persistence
- Implement state validation on restoration

### 2.3 Rolling Window Buffer
For strategies needing windowed calculations:
- Implement efficient circular buffer for price history
- Support configurable window sizes
- O(1) add/remove operations

---

## Phase 3: Profile System Enhancement

**Goal:** Full YAML-to-BotConfig mapping to leverage existing profile infrastructure.

### 3.1 Complete YAML Field Mapping
**Files:**
- `src/gpt_trader/orchestration/configuration/bot_config/bot_config.py`
- `config/profiles/*.yaml`

Map all canary.yaml fields into BotConfig:
- Daily trading windows
- Circuit breaker configurations
- Risk limit overrides
- Symbol-specific settings

### 3.2 Profile Validation
Add validation to ensure YAML profiles are complete and consistent with BotConfig schema.

### 3.3 Remove Unused ConfigState
**Files:** `src/gpt_trader/orchestration/configuration/bot_config/state.py`

Verify ConfigState is unused and remove if confirmed.

---

## Phase 4: Infrastructure Hardening

**Goal:** Production readiness for the features documented in ARCHITECTURE.md.

### 4.1 Durable Restart State
**Files:** `src/gpt_trader/persistence/`

Harden OrdersStore/EventStore for production:
- Atomic writes
- Corruption detection
- Recovery procedures

### 4.2 WebSocket Event Handling
**Files:** `src/gpt_trader/features/brokerages/coinbase/`

Complete user-event handling:
- Order fill notifications
- Position updates
- Account balance changes

### 4.3 Bootstrap Testing
**Files:** `tests/unit/gpt_trader/orchestration/`

Once Phase 1 completes, add comprehensive bootstrap tests:
- Environment variable handling
- Profile loading
- Container initialization

---

## Phase 5: Future Activation (When INTX Access Granted)

**Goal:** Ready for perpetuals trading when Coinbase grants access.

### 5.1 Perps Execution Paths
- Currently compiled and tested
- Activate with `COINBASE_ENABLE_DERIVATIVES=1`

### 5.2 Funding Rate Integration
- Add funding rate accrual to DeterministicBroker
- Implement funding payment tracking

---

## Deprecation List

| Component | Status | Action |
|-----------|--------|--------|
| `runtime_settings.py` | DELETED | Complete |
| Global `config` in bot_config.py | DELETED | Complete |
| `ServiceRegistry` as public API | ACTIVE | Deprecate - move to internal |
| `ConfigState` class | UNUSED | Delete after verification |
| Backward-compat property aliases in BotConfig | ACTIVE | Review for removal |

---

## Critical Files

### Phase 1 (DI Inversion)
1. `src/gpt_trader/orchestration/bootstrap.py` - Main refactoring target (~500 LOC)
2. `src/gpt_trader/app/container.py` - Enhance as canonical root
3. `src/gpt_trader/orchestration/trading_bot/bot.py` - Constructor simplification
4. `src/gpt_trader/orchestration/storage.py` - Update dependencies
5. `tests/integration/conftest.py` - Update fixtures (~50 locations)

### Phase 2 (Strategy Performance)
1. `src/gpt_trader/features/live_trade/engines/` - Strategy engine implementations
2. `src/gpt_trader/features/strategy_tools/` - Shared strategy utilities
3. `src/gpt_trader/orchestration/strategy_orchestrator/` - Strategy lifecycle
4. New: `src/gpt_trader/features/strategy_tools/incremental_stats.py`
5. New: `src/gpt_trader/features/strategy_tools/rolling_buffer.py`

### Phase 3 (Profile System)
1. `src/gpt_trader/orchestration/configuration/bot_config/bot_config.py`
2. `src/gpt_trader/orchestration/configuration/bot_config/state.py`
3. `config/profiles/canary.yaml` (reference for expected fields)
4. `config/profiles/spot.yaml`, `paper.yaml`, `dev_entry.yaml`

---

## Success Criteria

| Phase | Criteria |
|-------|----------|
| **Phase 1** | `build_bot()` returns `TradingBot`, no `ServiceRegistry` in public API |
| **Phase 2** | Strategy tick processing is O(1) via Welford's algorithm; `rehydrate()` properly restores state |
| **Phase 3** | All canary.yaml fields mapped to BotConfig and usable at runtime |
| **Phase 4** | All items in ARCHITECTURE.md "What's Actually Working" verified |
| **Phase 5** | Perps code paths exercised in tests, ready for activation |
