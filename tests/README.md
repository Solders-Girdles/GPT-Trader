# Test Suite Documentation

## 📂 Test Organization

```
tests/
├── README.md           # This file
├── conftest.py        # Pytest configuration and fixtures
├── conftest_full.py   # Extended test configuration
├── factories.py       # Test data factories
├── pytest.ini         # Pytest settings
│
├── unit/              # Unit tests for individual components
│   ├── backtest/     # Backtesting engine tests
│   ├── portfolio/    # Portfolio management tests
│   ├── risk/         # Risk management tests
│   ├── strategy/     # Strategy implementation tests
│   └── test_*.py     # Other unit tests
│
├── integration/       # Integration tests
│   ├── workflow/     # Workflow integration tests
│   ├── phases/       # Phase-specific integration tests
│   ├── weeks/        # Weekly milestone tests
│   └── pipelines/    # Pipeline integration tests
│
├── system/           # System-level tests
│   ├── test_data_preparation.py
│   ├── test_multiprocessing.py
│   ├── test_talib_integration.py
│   └── test_joblib_migration.py
│
├── acceptance/       # User acceptance tests
├── performance/      # Performance benchmarks
└── production/       # Production readiness tests
```

## 🚀 Running Tests

### Run All Tests
```bash
poetry run pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
poetry run pytest tests/unit/

# Integration tests only
poetry run pytest tests/integration/

# System tests only
poetry run pytest tests/system/
```

### Run with Coverage
```bash
poetry run pytest --cov=src/bot --cov-report=html
```

### Run in Parallel
```bash
poetry run pytest -n auto
```

### Run with Verbose Output
```bash
poetry run pytest -v
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
- Fast, isolated tests for individual functions and classes
- No external dependencies (mocked)
- Should run in < 1 second each
- Located in subdirectories matching source structure

### Integration Tests (`tests/integration/`)
- Test interactions between multiple components
- May use test databases or mock external services
- Grouped by functional area:
  - `workflow/` - End-to-end workflow tests
  - `phases/` - Phase milestone verification
  - `weeks/` - Weekly progress validation
  - `pipelines/` - Data pipeline tests

### System Tests (`tests/system/`)
- Test system-wide functionality
- Verify external integrations (TA-Lib, multiprocessing)
- Data preparation and migration tests
- May be slower and require specific environment setup

### Acceptance Tests (`tests/acceptance/`)
- User-facing functionality tests
- Verify requirements are met
- May include UI/CLI testing

### Performance Tests (`tests/performance/`)
- Benchmark critical operations
- Track performance regressions
- Memory and CPU profiling

### Production Tests (`tests/production/`)
- Smoke tests for production deployment
- Health checks and monitoring
- Critical path validation

## 📝 Writing Tests

### Test Naming Convention
```python
def test_<component>_<action>_<expected_result>():
    """Test that <component> <action> produces <expected_result>."""
    pass
```

### Using Fixtures
```python
def test_strategy_backtest(sample_data, demo_strategy):
    """Test strategy backtesting with sample data."""
    results = demo_strategy.backtest(sample_data)
    assert results.total_return > 0
```

### Test Organization Example
```python
# tests/unit/strategy/test_demo_ma.py
import pytest
from src.bot.strategy.demo_ma import DemoMAStrategy

class TestDemoMAStrategy:
    """Test suite for DemoMAStrategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = DemoMAStrategy(window=20)
        assert strategy.window == 20
    
    def test_signal_generation(self, sample_data):
        """Test signal generation logic."""
        strategy = DemoMAStrategy()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
```

## 🔧 Configuration

### pytest.ini
Main pytest configuration file with test discovery patterns and settings.

### conftest.py
Shared fixtures and test configuration available to all tests.

### factories.py
Test data factories for generating consistent test data.

## 🎯 Test Coverage Goals

- **Unit Tests**: > 80% coverage
- **Integration Tests**: Critical paths covered
- **System Tests**: All external integrations
- **Overall**: > 70% total coverage

## 🛠️ Continuous Integration

Tests are automatically run on:
- Every commit (via pre-commit hooks)
- Pull requests (GitHub Actions)
- Nightly builds (full test suite)

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](../docs/DEVELOPMENT_GUIDELINES.md#testing)
- [CI/CD Pipeline](.github/workflows/test.yml)