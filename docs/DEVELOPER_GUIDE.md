# 👨‍💻 GPT-Trader Developer Guide

Complete guide for developers contributing to or extending GPT-Trader.

## 📋 Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Testing](#testing)
4. [Adding New Features](#adding-new-features)
5. [Code Standards](#code-standards)
6. [Debugging](#debugging)
7. [Performance Optimization](#performance-optimization)
8. [Contributing](#contributing)

## 🔧 Development Setup

### Prerequisites

```bash
# Required
python --version  # 3.12+
poetry --version  # 1.5+

# Optional but recommended
git --version     # 2.30+
make --version    # For automation
```

### Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/gpt-trader.git
cd gpt-trader

# Install dependencies (including dev dependencies)
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Verify installation
poetry run python -m bot.cli --version
```

### Environment Configuration

```bash
# Create local environment file
cp .env.template .env.local

# For development, use demo mode
echo "DEMO_MODE=true" >> .env.local
echo "LOG_LEVEL=DEBUG" >> .env.local

# Or set up real API credentials for full testing
# Edit .env.local with your Alpaca API keys
```

## 📁 Project Structure

```
gpt-trader/
├── src/bot/                 # Main application code
│   ├── backtest/            # Backtesting engine
│   ├── cli/                 # Command-line interface
│   ├── config/              # Configuration management
│   ├── dataflow/            # Data sourcing and validation
│   ├── exceptions/          # Error handling
│   ├── exec/                # Order execution
│   ├── indicators/          # Technical indicators
│   ├── live/                # Live trading components
│   ├── logging/             # Logging utilities
│   ├── metrics/             # Performance metrics
│   ├── monitor/             # Monitoring and alerts
│   ├── optimization/        # Strategy optimization
│   ├── portfolio/           # Portfolio management
│   ├── risk/                # Risk management
│   ├── security/            # Security and secrets
│   ├── strategy/            # Trading strategies
│   └── utils/               # Utility functions
│
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── system/             # System tests
│   ├── acceptance/         # User acceptance tests
│   └── performance/        # Performance tests
│
├── docs/                    # Documentation
├── examples/                # Example scripts
├── scripts/                 # Utility scripts
└── data/                   # Data directory (git-ignored)
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/bot --cov-report=html

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest -m "not slow"  # Skip slow tests

# Run in parallel
poetry run pytest -n auto

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/integration/test_cli_integration.py

# Run specific test
poetry run pytest tests/integration/test_cli_integration.py::TestCLIIntegration::test_backtest_basic
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/strategy/test_my_strategy.py
import pytest
import pandas as pd
from bot.strategy.my_strategy import MyStrategy

class TestMyStrategy:
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MyStrategy(param1=10, param2=20)
        assert strategy.param1 == 10
        assert strategy.param2 == 20

    def test_signal_generation(self, sample_data):
        """Test signal generation."""
        strategy = MyStrategy()
        signals = strategy.generate_signals(sample_data)

        assert len(signals) == len(sample_data)
        assert signals.dtype == bool

    @pytest.mark.parametrize("param1,param2,expected", [
        (10, 20, True),
        (5, 30, False),
        (15, 15, True),
    ])
    def test_parameter_validation(self, param1, param2, expected):
        """Test parameter validation."""
        strategy = MyStrategy(param1=param1, param2=param2)
        assert strategy.is_valid() == expected
```

#### Integration Test Example

```python
# tests/integration/test_backtest_integration.py
import pytest
from bot.backtest import run_backtest
from bot.strategy import DemoMAStrategy

def test_full_backtest_workflow():
    """Test complete backtest workflow."""
    # Setup
    strategy = DemoMAStrategy(window=20)

    # Run backtest
    results = run_backtest(
        strategy=strategy,
        symbol="AAPL",
        start="2024-01-01",
        end="2024-03-31"
    )

    # Verify results
    assert results is not None
    assert "total_return" in results
    assert "sharpe_ratio" in results
    assert results["total_return"] != 0
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 101,
        "low": np.random.randn(100).cumsum() + 99,
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)
    return data

@pytest.fixture
def mock_api_client(mocker):
    """Mock API client for testing."""
    client = mocker.Mock()
    client.get_account.return_value = {"cash": 100000}
    return client
```

## 🚀 Adding New Features

### Adding a New Strategy

1. **Create strategy file**:
```python
# src/bot/strategy/my_strategy.py
from bot.strategy.base import Strategy
import pandas as pd

class MyStrategy(Strategy):
    """My custom trading strategy."""

    def __init__(self, param1: int = 10, param2: float = 0.5):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        # Your strategy logic here
        signals = pd.Series(False, index=data.index)

        # Example: Simple condition
        signals[data["close"] > data["close"].rolling(self.param1).mean()] = True

        return signals
```

2. **Register strategy**:
```python
# src/bot/strategy/__init__.py
from .my_strategy import MyStrategy

AVAILABLE_STRATEGIES = {
    "my_strategy": MyStrategy,
    # ... other strategies
}
```

3. **Add CLI support**:
```python
# src/bot/cli/backtest.py
# Add to strategy choices
strategy_group.add_argument(
    "--strategy",
    choices=["demo_ma", "trend_breakout", "my_strategy"],  # Add here
    default="trend_breakout",
)
```

4. **Write tests**:
```python
# tests/unit/strategy/test_my_strategy.py
# See test examples above
```

### Adding a New CLI Command

1. **Create command module**:
```python
# src/bot/cli/my_command.py
import argparse
from rich.console import Console

console = Console()

def add_subparser(subparsers):
    """Add my-command subparser."""
    parser = subparsers.add_parser(
        "my-command",
        help="Description of my command",
        description="Detailed description",
    )

    parser.add_argument("--option", help="Command option")
    parser.set_defaults(func=run_my_command)

def run_my_command(args: argparse.Namespace) -> None:
    """Execute my command."""
    console.print("[bold]Running my command![/bold]")
    # Command logic here
```

2. **Register command**:
```python
# src/bot/cli/__main__.py
from . import my_command

# In create_parser function
my_command.add_subparser(subparsers)
```

## 📏 Code Standards

### Style Guide

```python
# Follow PEP 8 and use type hints
def calculate_returns(
    prices: pd.Series,
    period: int = 1,
    method: str = "simple"
) -> pd.Series:
    """Calculate returns over specified period.

    Args:
        prices: Price series
        period: Number of periods for return calculation
        method: Calculation method ('simple' or 'log')

    Returns:
        Series of returns

    Raises:
        ValueError: If method is not supported
    """
    if method == "simple":
        return prices.pct_change(period)
    elif method == "log":
        return np.log(prices / prices.shift(period))
    else:
        raise ValueError(f"Unsupported method: {method}")
```

### Linting and Formatting

```bash
# Format code with black
poetry run black src/ tests/

# Lint with ruff
poetry run ruff check src/ tests/

# Type check with mypy
poetry run mypy src/

# Run all checks
poetry run pre-commit run --all-files
```

### Commit Messages

```bash
# Format: <type>(<scope>): <subject>

# Examples:
git commit -m "feat(strategy): add RSI-based strategy"
git commit -m "fix(backtest): correct sharpe ratio calculation"
git commit -m "docs(cli): update command examples"
git commit -m "test(integration): add paper trading tests"
git commit -m "perf(data): optimize data loading speed"
```

Types: feat, fix, docs, test, perf, refactor, style, chore

## 🐛 Debugging

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG poetry run python -m bot.cli backtest ...

# Use Python debugger
poetry run python -m pdb -m bot.cli backtest ...

# Use IPython for interactive debugging
poetry run ipython
>>> from bot.backtest import run_backtest
>>> %debug run_backtest(...)
```

### Common Issues

#### Import Errors
```python
# Problem: ModuleNotFoundError
# Solution: Ensure PYTHONPATH includes src/
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

#### Data Issues
```python
# Add data validation
from bot.dataflow.validate import validate_daily_bars

data = fetch_data(symbol)
data = validate_daily_bars(data, strict=False)  # Repair mode
```

#### Memory Issues
```python
# Profile memory usage
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass
```

## ⚡ Performance Optimization

### Profiling

```bash
# CPU profiling
poetry run python -m cProfile -o profile.stats -m bot.cli backtest ...
poetry run python -m pstats profile.stats

# Memory profiling
poetry run mprof run python -m bot.cli backtest ...
poetry run mprof plot

# Line profiling
poetry run kernprof -l -v your_script.py
```

### Optimization Tips

```python
# 1. Use vectorized operations
# Bad
signals = []
for i in range(len(data)):
    if data.iloc[i]["close"] > threshold:
        signals.append(True)
    else:
        signals.append(False)

# Good
signals = data["close"] > threshold

# 2. Cache expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param):
    # Complex calculation
    return result

# 3. Use numba for numerical operations
from numba import jit

@jit(nopython=True)
def fast_calculation(arr):
    # Numerical operations
    return result
```

## 🤝 Contributing

### Development Workflow

1. **Fork and clone**:
```bash
git clone https://github.com/YOUR_USERNAME/gpt-trader.git
cd gpt-trader
git remote add upstream https://github.com/ORIGINAL_OWNER/gpt-trader.git
```

2. **Create feature branch**:
```bash
git checkout -b feature/my-new-feature
```

3. **Make changes and test**:
```bash
# Make your changes
poetry run pytest tests/
poetry run black src/ tests/
poetry run ruff check src/
```

4. **Commit and push**:
```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/my-new-feature
```

5. **Create pull request**:
- Go to GitHub
- Create PR from your branch to main
- Fill in PR template
- Wait for review

### Code Review Checklist

- [ ] Tests pass (`poetry run pytest`)
- [ ] Code is formatted (`poetry run black`)
- [ ] Code passes linting (`poetry run ruff check`)
- [ ] Type hints added where appropriate
- [ ] Documentation updated
- [ ] No sensitive data exposed
- [ ] Performance impact considered
- [ ] Breaking changes documented

## 📚 Resources

### Internal Documentation
- [Architecture Overview](ARCHITECTURE_REVIEW.md)
- [API Documentation](API_REFERENCE.md)
- [Strategy Development](STRATEGY_GUIDE.md)

### External Resources
- [Python Best Practices](https://docs.python-guide.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Pytest Documentation](https://docs.pytest.org/)

### Getting Help
- Open an issue on GitHub
- Join our Discord community
- Check existing issues and PRs
- Read the FAQ

---

**Questions?** Open an issue or reach out on Discord!
