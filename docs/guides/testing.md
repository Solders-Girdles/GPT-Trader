# Testing Guide

## Overview

This codebase targets a **100% pass rate** on the actively maintained spot trading suites. Legacy code from v1 architecture is properly quarantined with skip markers.

## Suite Layout

- `tests/unit/bot_v2/` – orchestration, broker adapters, risk, CLI, and feature slices
- `tests/unit/coinbase/` – Coinbase-specific utilities (quantisation, spec validation)
- `tests/support/` – reserved for shared fixtures when co-located fixtures are insufficient (currently minimal)

Historical integration, performance, and experimental suites were archived in September 2024; pull them from git history if you need a reference.

## Current Test Metrics

### Active Test Suites (Maintained)
- **Collection**: Run `poetry run pytest --collect-only` to view current test counts and selection status
- **Pass Target**: **100%** ✅ for all selected active suites
- **Coverage Focus**: Spot trading orchestration, Coinbase integration, risk controls

### Legacy Context
- V1/v1.5 test suites have been removed from the active tree; fetch them from
  repository history if needed.
- Real-API and performance harnesses were archived alongside the legacy code in
  git history.

## Running Tests

### Quick Commands

```bash
# Discover suites & verify dependency setup
poetry run pytest --collect-only

# Run primary regression suite (must pass)
poetry run pytest -q

# Component-focused slices (optional, targeted)
poetry run pytest tests/unit/bot_v2/features/live_trade/ -v
poetry run pytest tests/unit/bot_v2/orchestration/ -v
poetry run pytest tests/unit/bot_v2/features/brokerages/ -v

# Run with coverage
poetry run pytest --cov=bot_v2 --cov-report=term-missing tests/unit/bot_v2
```

## Test Organization

### Active Test Suites

#### 1. **Live Trade Tests** (`tests/unit/bot_v2/features/live_trade/`)
- **Purpose**: Validate live execution and risk rails for spot trading (perps code paths remain covered for INTX accounts)
- **Coverage**: Order execution, risk management, PnL tracking
- **Status**: Passing (tracked via CI and manual runs)

#### 2. **Orchestration Tests** (`tests/unit/bot_v2/orchestration/`)
- **Purpose**: Validate bot coordination and paper trading
- **Coverage**: Bot lifecycle, mock trading, event handling
- **Status**: Fully passing

#### 3. **Brokerage Tests** (`tests/unit/bot_v2/features/brokerages/`)
- **Purpose**: Validate exchange integrations
- **Coverage**: Coinbase API, WebSocket streaming, order management
- **Status**: Core functionality passing

#### 4. **Foundation Tests** (`tests/unit/test_foundation.py`)
- **Purpose**: Validate core system assumptions
- **Coverage**: Basic data flow, configuration, initialization
- **Status**: 4/4 passing (100%)

### Legacy Test Suites

Legacy V1/V1.5 suites (backtest, portfolio, risk, strategy, performance, etc.)
have been removed from the active repository. Historical copies remain
available in version control history for reference.

## Writing Tests

### Test Structure

```python
# Standard test structure for bot_v2
import pytest
from decimal import Decimal
from bot_v2.features.live_trade.risk import LiveRiskManager

class TestRiskValidation:
    """Test risk management validation."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager with test config."""
        config = RiskConfig(
            max_leverage=5,
            daily_loss_limit=Decimal("100")
        )
        return LiveRiskManager(config)

    def test_leverage_validation(self, risk_manager):
        """Test leverage limits are enforced."""
        # Test implementation
        assert risk_manager.validate_leverage(10) is False
```

### Key Testing Utilities

#### Quantization Helpers
Located in `src/bot_v2/utilities/quantization.py`:
- `quantize_size()` - Round order sizes to exchange requirements
- `quantize_price()` - Round prices to tick size

#### Broker Test Doubles
- `DeterministicBroker` (src/bot_v2/orchestration/deterministic_broker.py): Stable IBrokerage stub for unit and orchestration tests.
- `ReduceOnlyStubBroker`: Specialized for reduce-only testing (if present).

Note: Real-API and sandbox integration harnesses were removed with the legacy suites. Recreate targeted coverage inside `tests/unit/bot_v2/` when new external behavior needs validation.

## Test Conventions

- Markers for `integration`, `real_api`, and `perf` remain defined in `pytest.ini` for forward compatibility, even though no tests currently use them.
- Async tests should rely on `anyio` helpers and injected clocks (avoid `time.sleep`).
- New tests belong beside the slice they exercise—co-locate fixtures in the nearest `conftest.py` to keep ownership clear.
- The `test-hygiene` pre-commit hook checks for extremely long test modules and accidental `sleep` calls; keep the hook green or update it when the heuristics change.

## Continuous Integration

### Pre-commit Hooks
```bash
# Install hooks (one-time)
pre-commit install

# Run manually
pre-commit run --all-files
```

### GitHub Actions
- Runs on every PR
- Tests active suites only
- Requires 100% pass rate for merge

## Common Issues and Solutions

### Issue: Import Errors
**Solutions**:
- Ensure you're using `bot_v2` imports, not legacy `src` paths
- Run `poetry install` after dependency updates (e.g., `pyotp` for security tests)

### Issue: Decimal Precision
**Solution**: Use quantization helpers for consistent rounding

### Issue: Mock Complexity
**Solution**: Use dependency injection instead of patching globals

## Test Coverage Goals

### Current Coverage
- **Critical Path**: >90% coverage on spot orchestration and Coinbase adapter
- **Integration Points**: Major APIs exercised via unit and integration suites
- **Edge Cases**: Risk guard failures, telemetry persistence, error handling

### Areas Needing Coverage
1. WebSocket streaming edge cases (especially user events)
2. Multi-symbol position management under stress
3. Advanced order types and optional derivatives flows
4. Funding rate calculations (future derivatives work)

## Best Practices

1. **Test Behavior, Not Implementation**: Focus on outcomes, not internal details
2. **Use Fixtures**: Share setup code via pytest fixtures
3. **Keep Tests Fast**: Mock external dependencies
4. **Clear Names**: Test names should describe what's being tested
5. **One Assertion Per Test**: Makes failures clear
6. **Document Skips**: Always provide reason for skipped tests

## Debugging Tests

```bash
# Run with verbose output
poetry run pytest -xvs tests/unit/bot_v2/features/live_trade/

# Run specific test
poetry run pytest tests/unit/bot_v2/features/live_trade/test_risk_validation.py::TestBasicRiskValidation::test_leverage_validation

# Drop into debugger on failure
poetry run pytest --pdb tests/unit/bot_v2/

# Show local variables on failure
poetry run pytest -l tests/unit/bot_v2/
```

## Performance Testing

```bash
# Run with benchmark
poetry run pytest tests/unit/bot_v2/ --benchmark-only

# Profile slow tests
poetry run pytest tests/unit/bot_v2/ --durations=10
```

## Conclusion

The test suite provides strong confidence in the spot trading stack (with future-ready perps paths) and maintains a 100% pass target on all selected suites. Legacy tests are clearly marked and don't impact active development. Focus remains on sustaining this high standard as new features are added.
