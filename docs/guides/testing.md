# Testing Guide

## Overview

This codebase targets a **100% pass rate** on actively maintained spot trading suites. The test infrastructure supports unit, integration, and behavioral testing with comprehensive coverage tracking.

## Quick Start

### Running Tests

```bash
# Run default suite (unit by default via pytest.ini addopts)
make test
uv run pytest

# Run with coverage
uv run pytest --cov --cov-report=html --cov-report=term

# Unit-only (explicit)
make test-unit

# Integration (opt-in; overrides pytest.ini addopts)
make test-integration-fast
make test-integration

# Run specific test file
uv run pytest tests/unit/gpt_trader/config/test_bot_config_env_aliasing.py

# Run specific test class or function
uv run pytest tests/unit/gpt_trader/config/test_bot_config_env_aliasing.py::TestBotConfigEnvAliasing::test_risk_max_leverage_with_prefix

# Run tests matching a pattern
uv run pytest -k "test_error"

# Run in parallel (faster)
uv run pytest -n auto

# Run with verbose output
uv run pytest -xvs tests/unit/gpt_trader/features/live_trade/
```

### Coverage Reports

```bash
# Full HTML report
uv run pytest --cov --cov-report=html:var/results/coverage/html --cov-report=term

# Open report (macOS)
open var/results/coverage/html/index.html

# Coverage for specific module
uv run pytest tests/unit/gpt_trader/config/ --cov=src/gpt_trader/config --cov-report=term-missing

# Quick coverage check
uv run pytest --cov --cov-report=term -q
```

## Test Organization

### Directory Structure

```
tests/
├── unit/                  # Unit tests (fast, isolated; default suite)
│   └── gpt_trader/         # Mirrors src/gpt_trader (preferred location)
├── integration/           # Integration tests (opt-in)
├── property/              # Property-based tests (opt-in)
├── contract/              # Contract tests (opt-in)
├── fixtures/              # Test fixtures and scenario data
├── support/               # Shared fixtures/helpers (when co-located insufficient)
└── _triage/               # Legacy test triage manifest
```

### Active Test Suites

| Suite | Path | Purpose |
|-------|------|---------|
| **Live Trade** | `tests/unit/gpt_trader/features/live_trade/` | Order execution, risk management, PnL tracking, bot lifecycle |
| **Brokerage** | `tests/unit/gpt_trader/features/brokerages/` | Coinbase API, WebSocket, order management |
| **Foundation** | `tests/unit/gpt_trader/core/` | Core system assumptions |

### Current Metrics

- **Pass Target**: 100% for selected suites
- **Coverage Baseline**: ~73%

## Writing Tests

### Test File Structure

```python
"""Tests for module_name."""
import pytest
from decimal import Decimal
from gpt_trader.module import ThingToTest


class TestThingToTest:
    """Test the ThingToTest class."""

    @pytest.fixture
    def sample_instance(self):
        """Provide standard test instance."""
        return ThingToTest(value=Decimal("100.00"))

    def test_creation(self) -> None:
        """Test basic object creation."""
        thing = ThingToTest()
        assert thing is not None

    def test_method_behavior(self, sample_instance) -> None:
        """Test a specific method."""
        result = sample_instance.method()
        assert result == expected_value

    def test_error_handling(self) -> None:
        """Test that errors are raised correctly."""
        with pytest.raises(ValueError):
            ThingToTest(invalid_param="bad")
```

### Best Practices

1. **Test Behavior, Not Implementation** - Focus on outcomes
2. **Use Fixtures** - Share setup via pytest fixtures
3. **Keep Tests Fast** - Mock external dependencies
4. **Clear Names** - `test_order_rejected_when_insufficient_funds`
5. **One Assertion Per Test** (guideline) - Makes failures clear
6. **Document Skips** - Always provide reason for skipped tests

### Testing Patterns

#### Async Code
```python
@pytest.mark.asyncio
async def test_async_function(self):
    """Test async function."""
    result = await async_fetch_data()
    assert result is not None
```

#### Mocking External Dependencies
```python
def test_api_call(mocker):
    """Test API call without hitting real API."""
    mock_response = {"price": "50000.00"}
    mocker.patch("gpt_trader.api.get_price", return_value=mock_response)
    result = fetch_price("BTC-USD")
    assert result == "50000.00"
```

#### Exceptions
```python
def test_custom_exception_message(self):
    """Test custom exception includes correct message."""
    error = ConfigurationError("Missing API key", config_key="api_key")
    assert str(error) == "Missing API key"
    assert error.context["config_key"] == "api_key"
```

## Test Markers

```python
# Integration test (skipped by default)
@pytest.mark.integration
def test_full_trading_cycle(self): ...

# Performance benchmark (opt-in with -m perf)
@pytest.mark.perf
def test_performance_benchmark(self): ...

# Requires real API (opt-in)
@pytest.mark.real_api
def test_live_api_connection(self): ...
```

```bash
# Run only integration tests
make test-integration-fast

# Skip integration tests (default)
make test
```

## Legacy Test Triage

We track tests that are **legacy-pattern** (delete vs modernize) in `tests/_triage/legacy_tests.yaml`.
The root `conftest.py` applies `legacy_delete` / `legacy_modernize` markers automatically based on that
manifest (tracking only; tests are not auto-skipped).

```bash
# Heuristic report of candidates + current manifest
make test-triage

# Fail (non-zero) if actionable candidates exist or manifest is invalid
make test-triage-check

# Collect just the modernize backlog (ignore pytest.ini addopts)
uv run pytest -o addopts= -m legacy_modernize --collect-only -q
```

## Coverage Guidelines

### Coverage Goals

- **New Code**: 80-90% coverage
- **Critical Paths**: 100% (trading logic, risk management)
- **Short-term Goal**: 80% overall
- **Long-term Goal**: 90% overall

### What to Cover

**Always Test:**
- Public API methods and functions
- Error handling and edge cases
- Business logic and calculations
- Data validation and state transitions

**Don't Need to Test:**
- Third-party libraries
- Python built-ins
- Simple getters/setters (unless they have logic)

## Debugging Tests

```bash
# Show print statements
uv run pytest -s

# Verbose output
uv run pytest -sv

# Drop into debugger on failure
uv run pytest --pdb tests/unit/gpt_trader/

# Show local variables on failure
uv run pytest -l tests/unit/gpt_trader/

# Profile slow tests
uv run pytest tests/unit/gpt_trader/ --durations=10
```

## Test Utilities

### Key Testing Helpers

- **Quantization**: `src/gpt_trader/utilities/quantization.py` - `quantize_price_side_aware()`
- **Broker Doubles**: `DeterministicBroker` in `src/gpt_trader/features/brokerages/mock/deterministic.py`

### Fixture Data

Use fixture data under `tests/fixtures/` for deterministic inputs:

- `tests/fixtures/backtesting/bt_20240101_000000_BTC-USD.json` - sample backtest dataset
- `tests/fixtures/brokerages/mock_products.yaml` - product catalog fixture

## Change Impact Analysis

Use the impact analyzer to see which tests map to a change:

```bash
uv run agent-impact --files src/gpt_trader/features/live_trade/bot.py --include-importers
uv run agent-impact --files src/gpt_trader/features/live_trade/bot.py --source-files
uv run agent-impact --files src/gpt_trader/features/live_trade/bot.py --include-importers --source-files
uv run agent-impact --files src/gpt_trader/features/live_trade/bot.py --exclude-integration --source-files
```

CI safeguards:
- `scripts/ci/check_test_hygiene.py` enforces unit test layout + basic hygiene guardrails
- Push builds to `main` always run the full suite

## Common Issues

### Import Errors
- Ensure using `gpt_trader` imports, not legacy `src` paths
- Use `uv sync --with security` for optional auth dependencies

### Fixtures Not Found
Ensure `conftest.py` is in the right place:
```
tests/
├── conftest.py           # Project-level fixtures
└── unit/
    └── gpt_trader/
        └── conftest.py   # Module-level fixtures
```

### Coverage Not Tracking
- Check `.coveragerc` configuration
- Clear cache: `uv run pytest --cache-clear`
- Remove old files: `rm -rf .coverage coverage.xml htmlcov/`

## CI Integration

### Pre-commit Hooks
```bash
pre-commit install      # Install hooks (one-time)
pre-commit run --all-files  # Run manually
```

### GitHub Actions
- Runs on every PR
- Tests active suites only
- Requires 100% pass rate for merge
