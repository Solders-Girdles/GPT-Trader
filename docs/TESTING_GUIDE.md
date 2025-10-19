# Testing Guide

## Quick Start

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov --cov-report=html --cov-report=term

# Run specific test file
poetry run pytest tests/unit/bot_v2/config/test_schemas.py

# Run specific test class or function
poetry run pytest tests/unit/bot_v2/config/test_schemas.py::TestBotConfig::test_valid_config

# Run tests matching a pattern
poetry run pytest -k "test_error"

# Run in parallel (faster)
poetry run pytest -n auto
```

### Viewing Coverage Reports

```bash
# Generate HTML coverage report
poetry run pytest --cov --cov-report=html:var/results/coverage/html

# Open in browser (macOS)
open var/results/coverage/html/index.html

# Open in browser (Linux)
xdg-open var/results/coverage/html/index.html
```

## Writing Tests

### Test File Structure

```python
"""Tests for module_name."""

import pytest
from bot_v2.module import ThingToTest


class TestThingToTest:
    """Test the ThingToTest class."""

    def test_creation(self) -> None:
        """Test basic object creation."""
        thing = ThingToTest()
        assert thing is not None

    def test_method_behavior(self) -> None:
        """Test a specific method."""
        thing = ThingToTest()
        result = thing.method()
        assert result == expected_value

    def test_error_handling(self) -> None:
        """Test that errors are raised correctly."""
        with pytest.raises(ValueError):
            ThingToTest(invalid_param="bad")
```

### Test Organization

```
tests/
├── unit/                  # Unit tests (fast, isolated)
│   └── bot_v2/
│       ├── config/        # Mirrors src structure
│       ├── features/
│       └── ...
├── integration/           # Integration tests (slower, test interactions)
└── fixtures/              # Shared test fixtures
```

### Best Practices

#### 1. One Test, One Assertion (Guideline)
```python
# Good - clear failure message
def test_account_has_positive_balance(self) -> None:
    account = Account(balance=100)
    assert account.balance > 0

def test_account_returns_correct_balance(self) -> None:
    account = Account(balance=100)
    assert account.balance == 100

# Acceptable - related assertions
def test_account_creation(self) -> None:
    account = Account(balance=100)
    assert account.balance == 100
    assert account.currency == "USD"
    assert account.is_active is True
```

#### 2. Use Fixtures for Common Setup
```python
@pytest.fixture
def sample_account():
    """Provide a standard test account."""
    return Account(balance=Decimal("1000.00"), currency="USD")

def test_deposit(sample_account):
    """Test depositing to account."""
    sample_account.deposit(Decimal("100.00"))
    assert sample_account.balance == Decimal("1100.00")
```

#### 3. Test Edge Cases
```python
def test_division_by_zero(self):
    """Test handling of division by zero."""
    with pytest.raises(ZeroDivisionError):
        calculate_ratio(10, 0)

def test_empty_list(self):
    """Test handling of empty input."""
    result = process_trades([])
    assert result == []

def test_negative_values(self):
    """Test handling of negative inputs."""
    with pytest.raises(ValueError):
        Account(balance=-100)
```

#### 4. Use Descriptive Test Names
```python
# Good
def test_order_rejected_when_insufficient_funds(self) -> None:
    """Test that order is rejected when account has insufficient funds."""

# Bad
def test_order(self) -> None:
    """Test order."""
```

#### 5. Test Both Success and Failure Paths
```python
def test_valid_order_accepted(self):
    """Test that valid order is accepted."""
    order = create_order(symbol="BTC-USD", quantity=1.0)
    assert order.status == OrderStatus.ACCEPTED

def test_invalid_order_rejected(self):
    """Test that invalid order is rejected."""
    with pytest.raises(ValidationError):
        create_order(symbol="INVALID", quantity=-1.0)
```

## Testing Patterns

### Testing Dataclasses
```python
def test_dataclass_creation(self):
    """Test dataclass can be created."""
    position = TradingPosition(
        symbol="BTC-USD",
        quantity=Decimal("1.0"),
        entry_price=Decimal("50000.00"),
    )
    assert position.symbol == "BTC-USD"
    assert isinstance(position.quantity, Decimal)

def test_dataclass_defaults(self):
    """Test dataclass default values."""
    position = TradingPosition(
        symbol="BTC-USD",
        quantity=Decimal("1.0"),
        entry_price=Decimal("50000.00"),
    )
    assert position.entry_timestamp is None
    assert position.unrealized_pnl is None
```

### Testing Exceptions
```python
def test_custom_exception_message(self):
    """Test custom exception includes correct message."""
    error = ConfigurationError("Missing API key", config_key="api_key")
    assert str(error) == "Missing API key"
    assert error.context["config_key"] == "api_key"

def test_exception_inheritance(self):
    """Test exception hierarchy."""
    assert issubclass(ConfigurationError, TradingError)
    assert issubclass(TradingError, Exception)
```

### Testing Async Code
```python
@pytest.mark.asyncio
async def test_async_function(self):
    """Test async function."""
    result = await async_fetch_data()
    assert result is not None

@pytest.mark.asyncio
async def test_async_error_handling(self):
    """Test async error handling."""
    with pytest.raises(NetworkError):
        await async_fetch_data(invalid_url="bad")
```

### Mocking External Dependencies
```python
def test_api_call(mocker):
    """Test API call without hitting real API."""
    mock_response = {"price": "50000.00"}
    mocker.patch("bot_v2.api.get_price", return_value=mock_response)

    result = fetch_price("BTC-USD")
    assert result == "50000.00"

def test_api_error_handling(mocker):
    """Test API error handling."""
    mocker.patch("bot_v2.api.get_price", side_effect=NetworkError("API down"))

    with pytest.raises(NetworkError):
        fetch_price("BTC-USD")
```

## Coverage Guidelines

### What to Cover

✅ **Always Test:**
- Public API methods and functions
- Error handling and edge cases
- Business logic and calculations
- Data validation
- State transitions

❌ **Don't Need to Test:**
- Third-party libraries
- Python built-ins
- Simple getters/setters (unless they have logic)
- Auto-generated code

### Coverage Goals

- **New Code**: Aim for 80-90% coverage
- **Critical Paths**: Aim for 100% coverage (trading logic, risk management)
- **Utilities**: Aim for 70-80% coverage
- **Types/Models**: Aim for 90-100% coverage (simple, should be easy to test)

### Measuring Coverage

```bash
# Check coverage for specific module
poetry run pytest tests/unit/bot_v2/config/ \
    --cov=src/bot_v2/config \
    --cov-report=term-missing

# Find uncovered lines
poetry run pytest --cov --cov-report=term-missing | grep "TOTAL"
```

## Test Markers

### Using Markers

```python
# Mark test as integration test (skipped by default)
@pytest.mark.integration
def test_full_trading_cycle(self):
    """Test complete trading workflow."""
    ...

# Mark test as slow (opt-in with -m perf)
@pytest.mark.perf
def test_performance_benchmark(self):
    """Benchmark strategy backtesting performance."""
    ...

# Mark test as requiring real API (opt-in)
@pytest.mark.real_api
def test_live_api_connection(self):
    """Test connection to real Coinbase API."""
    ...
```

### Running Marked Tests

```bash
# Run only integration tests
poetry run pytest -m integration

# Run performance tests
poetry run pytest -m perf

# Skip integration tests (default behavior)
poetry run pytest -m "not integration"
```

## Debugging Tests

### Print Debugging
```bash
# Show print statements
poetry run pytest -s

# Show print statements and verbose output
poetry run pytest -sv
```

### Debugging with pdb
```python
def test_complex_logic(self):
    """Test complex logic."""
    result = complex_function()
    import pdb; pdb.set_trace()  # Debugger will stop here
    assert result == expected
```

### Running Single Test
```bash
# Run single test with verbose output
poetry run pytest tests/unit/bot_v2/config/test_schemas.py::TestBotConfig::test_valid_config -vv
```

## Common Issues

### Import Errors
```bash
# Make sure you're in the project root
cd /path/to/GPT-Trader

# Make sure dependencies are installed
poetry install

# Check PYTHONPATH is set correctly (should happen automatically)
echo $PYTHONPATH
```

### Fixtures Not Found
```python
# Make sure conftest.py is in the right place
tests/
├── conftest.py           # Project-level fixtures
└── unit/
    └── bot_v2/
        └── conftest.py   # Module-level fixtures
```

### Coverage Not Tracking
```bash
# Make sure .coveragerc is configured correctly
cat .coveragerc

# Check that source path is correct
[run]
source = src/bot_v2  # Should point to your source code
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)
- Coverage report: `var/results/coverage/html/index.html`
- Coverage summary: `testing/TESTING_COVERAGE.md`
