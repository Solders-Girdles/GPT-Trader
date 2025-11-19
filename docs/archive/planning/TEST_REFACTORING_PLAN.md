# Test File Refactoring Plan

## Executive Summary

This document outlines a systematic approach to refactoring three large test files (1,150+ lines each) into smaller, more maintainable units. The refactoring will improve:

- **Developer Experience**: Easier navigation and faster test discovery
- **CI/CD Performance**: Better parallelization through granular test files
- **Maintainability**: Reduced cognitive load when working on specific features
- **Test Execution**: Ability to run targeted test subsets

---

## Refactoring Targets

| File | Lines | Test Classes | Proposed Split |
|------|-------|--------------|----------------|
| `test_telemetry.py` | 1,286 | 5 classes | 4 files |
| `test_strategy_orchestrator.py` | 1,223 | 19 classes | 5 files |
| `test_execution_coordinator.py` | 1,152 | 7 classes | 3 files |

**Total**: 3,661 lines → 12 focused files (~305 lines avg)

---

## Phase 1: test_telemetry.py Refactoring

**Current Location**: `tests/unit/bot_v2/orchestration/coordinators/test_telemetry.py`

**New Structure**:
```
tests/unit/bot_v2/orchestration/coordinators/telemetry/
├── conftest.py                          # Shared fixtures (40 lines)
├── test_telemetry_initialization.py     # 280 lines
├── test_telemetry_streaming.py          # 450 lines
├── test_telemetry_lifecycle.py          # 360 lines
└── test_telemetry_async.py              # 270 lines
```

### File Breakdown

#### 1. `conftest.py` (Shared Fixtures)
**Lines**: ~40
**Content**:
```python
from __future__ import annotations

import pytest
from unittest.mock import Mock

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry

BOT_ID = "coinbase_trader"


@pytest.fixture
def make_context():
    """Factory fixture for creating CoordinatorContext instances."""
    def _make_context(
        *,
        broker: object | None = None,
        risk_manager: object | None = None,
        symbols: tuple[str, ...] = ("BTC-PERP",),
    ) -> CoordinatorContext:
        config = BotConfig(profile=Profile.PROD)
        registry = ServiceRegistry(
            config=config,
            broker=broker,
            risk_manager=risk_manager,
            event_store=Mock(),
            orders_store=Mock(),
        )
        runtime_state = PerpsBotRuntimeState(list(symbols))

        return CoordinatorContext(
            config=config,
            registry=registry,
            event_store=registry.event_store,
            orders_store=registry.orders_store,
            broker=broker,
            risk_manager=risk_manager,
            symbols=symbols,
            bot_id=BOT_ID,
            runtime_state=runtime_state,
        )
    return _make_context
```

#### 2. `test_telemetry_initialization.py` (~280 lines)
**Purpose**: Tests for coordinator initialization, broker integration, and health checks

**Test Classes**:
- Standalone initialization tests (lines 47-69)
- `TestDynamicImportAndInitialization` (lines 124-225)
- `TestMetricEmissionAndErrorHandling` (lines 227-335)

**Migration Steps**:
1. Copy module docstring and imports
2. Import `make_context` fixture from conftest
3. Move initialization test functions (lines 47-69)
4. Move `TestDynamicImportAndInitialization` class
5. Move `TestMetricEmissionAndErrorHandling` class
6. Run tests: `pytest tests/unit/bot_v2/orchestration/coordinators/telemetry/test_telemetry_initialization.py -v`

#### 3. `test_telemetry_streaming.py` (~450 lines)
**Purpose**: Tests for streaming restart lifecycle and configuration changes

**Test Classes**:
- `TestStreamingRestartLifecycle` (lines 337-639) - 302 lines

**Key Features**:
- Config change detection
- Restart logic with streaming enabled/disabled
- Error handling for stop/start failures
- Symbol and stream level changes
- Concurrent configuration changes

**Migration Steps**:
1. Copy imports and module docstring
2. Import `make_context` from conftest
3. Move `TestStreamingRestartLifecycle` class
4. Run tests: `pytest tests/unit/bot_v2/orchestration/coordinators/telemetry/test_telemetry_streaming.py -v`

#### 4. `test_telemetry_lifecycle.py` (~360 lines)
**Purpose**: Tests for complete streaming lifecycle management

**Test Classes**:
- Startup and shutdown tests (lines 71-122)
- `TestStreamingLifecycleManagement` (lines 641-1018)

**Key Features**:
- Start streaming with various configurations
- Stop streaming cleanup logic
- Task cancellation handling
- Multiple start/stop cycles
- Profile-based streaming enablement

**Migration Steps**:
1. Copy imports and module docstring
2. Import `make_context` from conftest
3. Move background task tests (lines 71-122)
4. Move `TestStreamingLifecycleManagement` class
5. Run tests: `pytest tests/unit/bot_v2/orchestration/coordinators/telemetry/test_telemetry_lifecycle.py -v`

#### 5. `test_telemetry_async.py` (~270 lines)
**Purpose**: Tests for async coroutine scheduling and edge cases

**Test Classes**:
- `TestAsyncCoroutineScheduling` (lines 1020-1287)

**Key Features**:
- Coroutine scheduling with running loop
- Fallback to asyncio.run when no loop
- Thread-safe scheduling via loop task handle
- Error handling in scheduling paths
- Compatibility methods

**Migration Steps**:
1. Copy imports and module docstring
2. Import `make_context` from conftest
3. Move `TestAsyncCoroutineScheduling` class
4. Run tests: `pytest tests/unit/bot_v2/orchestration/coordinators/telemetry/test_telemetry_async.py -v`

### Validation
```bash
# Run all telemetry tests
pytest tests/unit/bot_v2/orchestration/coordinators/telemetry/ -v

# Run specific test file
pytest tests/unit/bot_v2/orchestration/coordinators/telemetry/test_telemetry_streaming.py::TestStreamingRestartLifecycle::test_restart_streaming_if_needed_handles_streaming_config_changes -v

# Verify coverage is maintained
pytest tests/unit/bot_v2/orchestration/coordinators/telemetry/ --cov=bot_v2.orchestration.coordinators.telemetry --cov-report=term-missing
```

---

## Phase 2: test_strategy_orchestrator.py Refactoring

**Current Location**: `tests/unit/bot_v2/orchestration/test_strategy_orchestrator.py`

**New Structure**:
```
tests/unit/bot_v2/orchestration/strategy_orchestrator/
├── conftest.py                                    # Shared fixtures (90 lines)
├── test_orchestrator_initialization.py            # 180 lines
├── test_orchestrator_data_preparation.py          # 280 lines
├── test_orchestrator_decision_execution.py        # 260 lines
├── test_orchestrator_risk_gates.py               # 180 lines
└── test_orchestrator_advanced_scenarios.py        # 310 lines
```

### File Breakdown

#### 1. `conftest.py` (Shared Fixtures)
**Lines**: ~90
**Content**: All fixtures from lines 32-96
- `mock_bot` - Core bot fixture
- `mock_spot_profile_service` - Spot profile service fixture
- `orchestrator` - Orchestrator instance fixture
- `test_balance` - Test balance fixture
- `test_position` - Test position fixture

**Migration Steps**:
1. Create new directory: `tests/unit/bot_v2/orchestration/strategy_orchestrator/`
2. Create `conftest.py` with all fixtures
3. Add module docstring explaining fixture purpose

#### 2. `test_orchestrator_initialization.py` (~180 lines)
**Purpose**: Strategy initialization, selection, and configuration

**Test Classes**:
- `TestStrategyOrchestratorInitialization` (lines 98-116)
- `TestInitStrategy` (lines 118-173)
- `TestGetStrategy` (lines 175-206)

**Coverage**:
- Bot initialization
- Perps vs Spot strategy creation
- Per-symbol strategy initialization
- Position fraction overrides
- Leverage configuration
- Strategy retrieval logic

**Migration Steps**:
1. Copy module docstring and imports
2. Move three test classes
3. Run tests: `pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/test_orchestrator_initialization.py -v`

#### 3. `test_orchestrator_data_preparation.py` (~280 lines)
**Purpose**: Balance, position, equity, and mark data handling

**Test Classes**:
- `TestEnsureBalances` (lines 208-228)
- `TestExtractEquity` (lines 231-265)
- `TestEnsurePositions` (lines 288-308)
- `TestBuildPositionState` (lines 311-333)
- `TestGetMarks` (lines 336-355)
- `TestAdjustEquity` (lines 358-379)
- Additional edge case tests from `TestStrategyOrchestratorEdgeCases` (relevant subset)

**Coverage**:
- Balance fetching and validation
- Equity extraction (USD/USDC)
- Position state building
- Mark price window management
- Equity adjustment for positions

**Migration Steps**:
1. Copy imports
2. Move six test classes focused on data preparation
3. Add relevant tests from edge cases class
4. Run tests: `pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/test_orchestrator_data_preparation.py -v`

#### 4. `test_orchestrator_decision_execution.py` (~260 lines)
**Purpose**: Strategy evaluation, decision recording, and execution flow

**Test Classes**:
- `TestEvaluateStrategy` (lines 462-502)
- `TestRecordDecision` (lines 505-514)
- `TestFetchSpotCandles` (lines 517-540)
- `TestProcessSymbol` (lines 543-615)
- `TestDecisionRoutingAndGuardChains` (lines 909-1072)

**Coverage**:
- Strategy.decide() invocation
- Decision recording
- Candle fetching for spot strategies
- Symbol processing orchestration
- Decision routing through guard chains
- Spot vs non-spot decision handling

**Migration Steps**:
1. Copy imports
2. Move test classes related to decision flow
3. Run tests: `pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/test_orchestrator_decision_execution.py -v`

#### 5. `test_orchestrator_risk_gates.py` (~180 lines)
**Purpose**: Risk gate validation and kill switch logic

**Test Classes**:
- `TestKillSwitchEngaged` (lines 268-285)
- `TestRunRiskGates` (lines 382-459)
- `TestKillSwitchLogic` (lines 841-907)

**Coverage**:
- Kill switch detection
- Volatility circuit breaker checks
- Market data staleness checks
- Kill switch early returns
- Warning message logging

**Migration Steps**:
1. Copy imports
2. Move three test classes
3. Run tests: `pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/test_orchestrator_risk_gates.py -v`

#### 6. `test_orchestrator_advanced_scenarios.py` (~310 lines)
**Purpose**: Edge cases, error handling, and position validation

**Test Classes**:
- `TestStrategyOrchestratorEdgeCases` (lines 618-839) - minus data prep tests
- `TestPositionStateBuildingAndValidation` (lines 1074-1224)

**Coverage**:
- Invalid position fraction handling
- Execution error logging
- No marks warnings
- Zero equity handling
- Short position scenarios
- Position state attribute validation
- Context preparation failures

**Migration Steps**:
1. Copy imports
2. Move edge case and position validation classes
3. Run tests: `pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/test_orchestrator_advanced_scenarios.py -v`

### Validation
```bash
# Run all orchestrator tests
pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/ -v

# Verify no test was lost
pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/ --collect-only | grep "test_"

# Check coverage
pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/ --cov=bot_v2.orchestration.strategy_orchestrator --cov-report=term-missing
```

---

## Phase 3: test_execution_coordinator.py Refactoring

**Current Location**: `tests/unit/bot_v2/orchestration/test_execution_coordinator.py`

**New Structure**:
```
tests/unit/bot_v2/orchestration/execution/
├── conftest.py                              # Shared fixtures (80 lines)
├── test_execution_workflows.py              # 420 lines
├── test_execution_error_handling.py         # 340 lines
└── test_execution_advanced_features.py      # 390 lines
```

### File Breakdown

#### 1. `conftest.py` (Shared Fixtures)
**Lines**: ~80
**Content**: All fixtures from lines 40-117
- `base_context` - Base coordinator context
- `coordinator` - ExecutionCoordinator instance
- `test_product` - Mock Product
- `test_order` - Mock Order

**Migration Steps**:
1. Create directory: `tests/unit/bot_v2/orchestration/execution/`
2. Create `conftest.py` with all fixtures
3. Add module docstring

#### 2. `test_execution_workflows.py` (~420 lines)
**Purpose**: Core execution workflows, order placement, and reconciliation

**Test Content**:
- Standalone workflow tests (lines 120-306)
- `TestExecutionCoordinatorAsyncFlows` (lines 308-403)
- Selected tests from `TestExecutionCoordinatorCoverageEnhancements` (execution flow tests)

**Coverage**:
- Execution engine initialization
- Order lock management
- Order placement inner logic
- Order reconciler caching
- Background task startup
- Reconciliation loop operation
- Dry run mode handling
- Decision execution invocation

**Migration Steps**:
1. Copy imports and docstring
2. Move standalone tests (lines 120-306)
3. Move `TestExecutionCoordinatorAsyncFlows`
4. Move relevant workflow tests from coverage enhancements class
5. Run tests: `pytest tests/unit/bot_v2/orchestration/execution/test_execution_workflows.py -v`

#### 3. `test_execution_error_handling.py` (~340 lines)
**Purpose**: Error handling, validation, and resilience

**Test Classes**:
- `TestExecutionCoordinatorFailureHandling` (lines 405-537)
- Error-related tests from `TestExecutionCoordinatorCoverageEnhancements`

**Coverage**:
- Validation error handling
- Execution error handling
- Unexpected error recovery
- Missing product handling
- Invalid mark price handling
- Close without position
- Runtime state unavailability
- Lock initialization errors
- Missing execution engine

**Migration Steps**:
1. Copy imports
2. Move `TestExecutionCoordinatorFailureHandling`
3. Move error-related coverage tests
4. Run tests: `pytest tests/unit/bot_v2/orchestration/execution/test_execution_error_handling.py -v`

#### 4. `test_execution_advanced_features.py` (~390 lines)
**Purpose**: Advanced features, guards, health checks, and edge cases

**Test Classes**:
- `TestExecutionCoordinatorGuardTriggers` (lines 539-630)
- `TestExecutionCoordinatorAdvancedEngine` (lines 632-695)
- `TestExecutionCoordinatorOrderLock` (lines 697-748)
- `TestExecutionCoordinatorHealthCheck` (lines 750-783)
- Remaining `TestExecutionCoordinatorCoverageEnhancements` tests

**Coverage**:
- Reduce-only mode enforcement
- Position side detection
- Leverage override handling
- Advanced execution engine selection
- Impact estimator building
- Order lock edge cases
- Health check status reporting
- Runtime settings management
- Metrics collection

**Migration Steps**:
1. Copy imports
2. Move guard, advanced engine, lock, and health check classes
3. Move remaining coverage enhancement tests
4. Run tests: `pytest tests/unit/bot_v2/orchestration/execution/test_execution_advanced_features.py -v`

### Validation
```bash
# Run all execution tests
pytest tests/unit/bot_v2/orchestration/execution/ -v

# Parallel execution test
pytest tests/unit/bot_v2/orchestration/execution/ -n auto

# Coverage verification
pytest tests/unit/bot_v2/orchestration/execution/ --cov=bot_v2.orchestration.coordinators.execution --cov-report=term-missing
```

---

## Implementation Checklist

### Pre-Refactoring
- [ ] Review current test coverage baseline
- [ ] Document all test counts per file
- [ ] Create feature branch: `git checkout -b refactor/test-file-organization`
- [ ] Ensure all tests currently pass: `pytest tests/unit/bot_v2/orchestration/ -v`

### Phase 1: test_telemetry.py
- [ ] Create directory: `tests/unit/bot_v2/orchestration/coordinators/telemetry/`
- [ ] Create `conftest.py` with `make_context` fixture
- [ ] Create `test_telemetry_initialization.py`
- [ ] Create `test_telemetry_streaming.py`
- [ ] Create `test_telemetry_lifecycle.py`
- [ ] Create `test_telemetry_async.py`
- [ ] Verify all tests pass: `pytest tests/unit/bot_v2/orchestration/coordinators/telemetry/ -v`
- [ ] Delete original `test_telemetry.py`
- [ ] Commit: `git commit -m "refactor: split test_telemetry.py into 4 focused files"`

### Phase 2: test_strategy_orchestrator.py
- [ ] Create directory: `tests/unit/bot_v2/orchestration/strategy_orchestrator/`
- [ ] Create `conftest.py` with all fixtures
- [ ] Create `test_orchestrator_initialization.py`
- [ ] Create `test_orchestrator_data_preparation.py`
- [ ] Create `test_orchestrator_decision_execution.py`
- [ ] Create `test_orchestrator_risk_gates.py`
- [ ] Create `test_orchestrator_advanced_scenarios.py`
- [ ] Verify all tests pass: `pytest tests/unit/bot_v2/orchestration/strategy_orchestrator/ -v`
- [ ] Delete original `test_strategy_orchestrator.py`
- [ ] Commit: `git commit -m "refactor: split test_strategy_orchestrator.py into 5 focused files"`

### Phase 3: test_execution_coordinator.py
- [ ] Create directory: `tests/unit/bot_v2/orchestration/execution/`
- [ ] Create `conftest.py` with all fixtures
- [ ] Create `test_execution_workflows.py`
- [ ] Create `test_execution_error_handling.py`
- [ ] Create `test_execution_advanced_features.py`
- [ ] Verify all tests pass: `pytest tests/unit/bot_v2/orchestration/execution/ -v`
- [ ] Delete original `test_execution_coordinator.py`
- [ ] Commit: `git commit -m "refactor: split test_execution_coordinator.py into 3 focused files"`

### Post-Refactoring
- [ ] Run full test suite: `pytest tests/unit/bot_v2/orchestration/ -v`
- [ ] Verify test count matches original
- [ ] Check coverage is maintained or improved
- [ ] Test parallel execution: `pytest tests/unit/bot_v2/orchestration/ -n auto`
- [ ] Update CI/CD configuration if needed
- [ ] Create PR with detailed description
- [ ] Request team review

---

## Expected Benefits

### 1. Developer Productivity
**Before**:
```bash
# Finding specific test in 1,286 line file
# Scroll through entire file, Ctrl+F for keywords
```

**After**:
```bash
# Clear file structure
tests/unit/bot_v2/orchestration/coordinators/telemetry/
├── test_telemetry_streaming.py     # ← Obviously here
```

### 2. Faster Test Execution
**Before**:
```bash
pytest test_telemetry.py  # Runs all 53 tests sequentially
# Time: ~12 seconds
```

**After**:
```bash
pytest telemetry/ -n 4  # 4 files run in parallel
# Time: ~4 seconds (3x faster)
```

### 3. Better CI/CD Feedback
**Before**:
```
❌ test_telemetry.py FAILED (one test in 53)
# Re-run entire file, wait for all 53 tests
```

**After**:
```
✅ test_telemetry_initialization.py PASSED
✅ test_telemetry_lifecycle.py PASSED
❌ test_telemetry_streaming.py FAILED (one test in 12)
# Re-run only 12 tests, faster feedback
```

### 4. Improved Maintainability
**Before**:
```python
# test_strategy_orchestrator.py (1,223 lines)
# Merge conflict on line 847
# Affects: initialization, execution, edge cases
```

**After**:
```python
# test_orchestrator_risk_gates.py (180 lines)
# Merge conflict isolated to risk gate tests
# Easy to resolve, clear context
```

---

## Migration Script Template

For automation, here's a template script to help with the refactoring:

```python
#!/usr/bin/env python3
"""
Test file refactoring automation script.
Usage: python migrate_tests.py --file test_telemetry.py --phase 1
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple


def extract_lines(file_path: Path, start: int, end: int) -> str:
    """Extract specific line range from file."""
    with open(file_path) as f:
        lines = f.readlines()
    return ''.join(lines[start-1:end])


def create_test_file(
    output_path: Path,
    imports: str,
    content: str,
    docstring: str
) -> None:
    """Create new test file with proper structure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f'"""{docstring}"""\n\n')
        f.write(imports)
        f.write('\n\n')
        f.write(content)


def migrate_telemetry(source: Path, target_dir: Path) -> None:
    """Migrate test_telemetry.py to multiple files."""
    # Implementation based on line ranges from plan
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate test files')
    parser.add_argument('--file', required=True)
    parser.add_argument('--phase', type=int, required=True)
    args = parser.parse_args()

    # Execute migration
    print(f"Migrating {args.file} (Phase {args.phase})")
```

---

## Rollback Plan

If issues arise during refactoring:

1. **Immediate Rollback**:
   ```bash
   git reset --hard HEAD~1  # Undo last commit
   ```

2. **Partial Rollback**:
   ```bash
   # Keep new files but restore original
   git checkout main -- tests/unit/bot_v2/orchestration/test_telemetry.py
   ```

3. **Test Verification**:
   ```bash
   # Ensure no tests were lost
   pytest --collect-only tests/unit/bot_v2/orchestration/ > after.txt
   diff before.txt after.txt
   ```

---

## Success Criteria

Refactoring is successful when:

1. ✅ All original tests pass in new structure
2. ✅ Test count matches exactly (307 tests)
3. ✅ Coverage percentage maintained or improved
4. ✅ CI/CD pipeline passes
5. ✅ Parallel execution works (`pytest -n auto`)
6. ✅ No duplicate test names
7. ✅ All fixtures properly shared via conftest
8. ✅ Import statements are correct
9. ✅ Team approves PR
10. ✅ Documentation updated

---

## Timeline Estimate

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 0 | Setup & baseline documentation | 30 min |
| 1 | Refactor test_telemetry.py | 2 hours |
| 1 | Testing & verification | 30 min |
| 2 | Refactor test_strategy_orchestrator.py | 3 hours |
| 2 | Testing & verification | 30 min |
| 3 | Refactor test_execution_coordinator.py | 2 hours |
| 3 | Testing & verification | 30 min |
| 4 | Final validation & PR creation | 1 hour |
| **Total** | | **10 hours** |

Recommended approach: 1 phase per day over 3 days.

---

## Questions & Decisions

### Q1: Should we keep integration tests together?
**Decision**: Yes, integration tests that span multiple components should remain together. This refactoring focuses on unit tests.

### Q2: What about shared helper functions?
**Decision**: Move to `conftest.py` as fixtures. Convert standalone functions to fixture factories.

### Q3: Should we update import paths in other files?
**Decision**: No other files import from these test modules, so no updates needed.

### Q4: What about test discovery with pytest?
**Decision**: Pytest auto-discovers all `test_*.py` files, so no configuration changes needed.

### Q5: Should we maintain git history for each test?
**Decision**: No, the refactoring commit will be the new baseline. Git blame will show the refactoring commit.

---

## Contact & Support

For questions during refactoring:
- Review this document
- Check pytest documentation: https://docs.pytest.org/
- Ask in team chat: #engineering-testing

---

**Document Version**: 1.0
**Last Updated**: 2025-10-22
**Author**: Claude Code
**Status**: Ready for Implementation
