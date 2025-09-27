# Testing Guide

## Overview

This codebase targets a **100% pass rate** on the actively maintained spot trading suites. Legacy code from v1 architecture is properly quarantined with skip markers.

## Current Test Metrics

### Active Test Suites (Maintained)
- **Collection Snapshot**: 480 collected / 58 deselected / 422 selected (`poetry run pytest --collect-only`)
- **Pass Target**: **100%** ✅ for the selected suites
- **Coverage Focus**: Spot trading orchestration, Coinbase integration, risk controls

### Legacy Context
- Legacy v1 suites remain skipped/deselected by default
- Integration and real-API tests require explicit markers and credentials
- Perpetuals-only metrics have been superseded by the spot-first counts above

## Running Tests

### Quick Commands

```bash
# Discover suites & verify dependency setup
poetry run pytest --collect-only

# Run primary spot suites (must pass)
poetry run pytest tests/unit/bot_v2 tests/unit/test_foundation.py -q

# Run specific component tests
poetry run pytest tests/unit/bot_v2/features/live_trade/ -v  # Perps trading
poetry run pytest tests/unit/bot_v2/orchestration/ -v         # Bot orchestration
poetry run pytest tests/unit/bot_v2/features/brokerages/ -v   # Exchange integration

# Run with coverage
poetry run pytest --cov=bot_v2 --cov-report=term-missing tests/unit/bot_v2

# Run integration tests (explicit marker overrides default skip)
poetry run pytest -m integration tests/integration/bot_v2 -q

# Run real API connectivity tests (opt-in; requires credentials)
poetry run pytest -m real_api tests/integration/real_api -q

# Canonical CDP auth + endpoints test
poetry run pytest -q tests/integration/test_cdp_comprehensive.py -m integration
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

### Legacy Test Suites (Skipped)

These tests are from v1 architecture and are intentionally skipped:

- `test_engine_broker_shim.py` - Legacy v1 broker interface
- `test_week1_core_components.py` - Superseded by bot_v2
- `test_week2_filters_guards.py` - Superseded by bot_v2
- `test_week3_execution.py` - Replaced by execution_v3
- `test_paper_trading_offline.py` - Legacy paper engine v1
- `test_coinbase_paper_integration.py` - Old paper trade v1
- `test_paper_engine_decoupling.py` - Legacy scaffolding

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
Located in `src/bot_v2/features/utils/quantization.py`:
- `quantize_size()` - Round order sizes to exchange requirements
- `quantize_price()` - Round prices to tick size

#### Broker Test Doubles
- `DeterministicBroker` (tests/utils/deterministic_broker.py): Stable IBrokerage stub for unit tests (preferred).
- `MockBroker` (src/bot_v2/orchestration/mock_broker.py): Deprecated for automated tests; still used in DEV profile and certain legacy tests. Marked via `@pytest.mark.uses_mock_broker` and skipped by default.
- `ReduceOnlyStubBroker`: Specialized for reduce-only testing (if present).

Note: Real API integration tests live under `tests/integration/real_api/` and are opt-in via environment (`BROKER=coinbase`, `COINBASE_SANDBOX=1`, sandbox keys). Run with the explicit marker: `pytest -m real_api tests/integration/real_api -q`.

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
poetry run pytest tests/unit/bot_v2/features/live_trade/test_risk_comprehensive.py::TestBasicRiskValidation::test_position_size_validation

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
