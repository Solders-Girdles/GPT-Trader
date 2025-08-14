# Contributing to GPT-Trader

Thank you for your interest in contributing to GPT-Trader! This guide will help you understand our development process, code standards, and how to submit contributions.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guide](#code-style-guide)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Security Guidelines](#security-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Report any unacceptable behavior to the maintainers

## Getting Started

1. **Fork the Repository**: Create your own fork of the GPT-Trader repository
2. **Clone Your Fork**: `git clone https://github.com/your-username/GPT-Trader.git`
3. **Add Upstream Remote**: `git remote add upstream https://github.com/original/GPT-Trader.git`
4. **Create a Branch**: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites
- Python 3.12+
- Poetry for dependency management
- Pre-commit hooks installed

### Installation
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Verify installation
poetry run gpt-trader --version
```

### Environment Configuration
```bash
# Copy environment template
cp .env.template .env

# Edit with your configuration
nano .env
```

## Code Style Guide

### Python Code Standards

We use automated tools to enforce consistent code style:
- **Black**: Code formatting (line length: 100)
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking

### Style Principles

#### 1. Formatting
```python
# Line length: 100 characters max
# Use Black formatter - it's non-negotiable

# Good: Black will format this automatically
def calculate_portfolio_return(
    prices: pd.DataFrame,
    weights: np.ndarray,
    start_date: datetime,
    end_date: datetime,
) -> float:
    """Calculate portfolio return for given period."""
    pass
```

#### 2. Imports
```python
# Order: standard library, third-party, local
# Ruff will sort these automatically

# Standard library
import os
from datetime import datetime
from typing import Optional, List, Dict

# Third-party
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Local imports
from bot.core import Strategy
from bot.utils import calculate_sharpe
```

#### 3. Type Annotations
```python
# Always use type hints for function signatures
def process_data(
    data: pd.DataFrame,
    symbol: str,
    lookback: int = 20,
) -> tuple[pd.Series, float]:
    """Process market data and return signals."""
    pass

# Use Optional for nullable types
def find_strategy(name: str) -> Optional[Strategy]:
    """Find strategy by name."""
    pass
```

#### 4. Docstrings
```python
def optimize_portfolio(
    returns: pd.DataFrame,
    constraints: Dict[str, float],
    method: str = "mean-variance",
) -> np.ndarray:
    """
    Optimize portfolio allocation using specified method.

    Args:
        returns: DataFrame of asset returns with assets as columns
        constraints: Dictionary of portfolio constraints
            - 'min_weight': Minimum weight per asset (default: 0.0)
            - 'max_weight': Maximum weight per asset (default: 1.0)
            - 'target_return': Target portfolio return (optional)
        method: Optimization method to use
            - 'mean-variance': Markowitz optimization
            - 'risk-parity': Equal risk contribution
            - 'equal-weight': Simple equal weighting

    Returns:
        Array of optimal portfolio weights

    Raises:
        ValueError: If constraints are invalid or infeasible
        OptimizationError: If optimization fails to converge

    Examples:
        >>> returns = pd.DataFrame(...)
        >>> weights = optimize_portfolio(
        ...     returns,
        ...     {'min_weight': 0.05, 'max_weight': 0.40}
        ... )
        >>> assert np.isclose(weights.sum(), 1.0)
    """
    pass
```

#### 5. Error Handling
```python
# Use specific exceptions with context
from bot.core.exceptions import DataException, ValidationException

def validate_price(price: float, symbol: str) -> float:
    """Validate price data."""
    if price <= 0:
        raise ValidationException(
            f"Invalid price for {symbol}: {price}",
            field="price",
            value=price,
        )
    return price

# Always use exception chaining
try:
    data = load_data(filepath)
except IOError as e:
    raise DataException(f"Failed to load data from {filepath}") from e
```

#### 6. Constants and Configuration
```python
# Use UPPER_CASE for module-level constants
DEFAULT_LOOKBACK_PERIOD = 20
MAX_POSITION_SIZE = 100000
RISK_FREE_RATE = 0.02

# Use Enums for fixed choices
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
```

#### 7. Class Design
```python
class Strategy(ABC):
    """Base strategy class."""

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize strategy with configuration."""
        self.config = config
        self._is_initialized = False

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.config.name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from data."""
        pass

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data (private method)."""
        pass
```

### Security Standards

#### Never Do This:
```python
# NEVER hardcode secrets
API_KEY = "sk-1234567890abcdef"  # WRONG!

# NEVER use pickle for serialization
import pickle  # WRONG!
pickle.dump(data, file)

# NEVER build SQL queries with string formatting
query = f"SELECT * FROM {table}"  # WRONG!

# NEVER accept unvalidated user input
symbol = request.args.get('symbol')  # WRONG!
execute_trade(symbol)
```

#### Always Do This:
```python
# Use environment variables for secrets
import os
API_KEY = os.environ.get("API_KEY")

# Use joblib for serialization
import joblib
joblib.dump(data, file)

# Use parameterized queries or whitelisting
ALLOWED_TABLES = ["trades", "positions"]
if table in ALLOWED_TABLES:
    query = f"SELECT * FROM {table}"

# Validate all user input
from bot.security.input_validation import validate_trading_symbol
symbol = validate_trading_symbol(request.args.get('symbol'))
```

## Commit Guidelines

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Test additions or fixes
- **chore**: Build process or auxiliary tool changes
- **security**: Security improvements

### Examples
```bash
# Feature
git commit -m "feat(strategy): add momentum trading strategy"

# Bug fix
git commit -m "fix(backtest): correct position sizing calculation"

# Documentation
git commit -m "docs(api): update REST API documentation"

# Security fix
git commit -m "security(auth): implement rate limiting for login attempts"
```

## Pull Request Process

### Before Submitting

1. **Update from upstream**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run quality checks**:
```bash
# Format code
poetry run black src/

# Run linting
poetry run ruff check src/ --fix

# Type checking
poetry run mypy src/

# Run tests
poetry run pytest

# Run pre-commit hooks
poetry run pre-commit run --all-files
```

3. **Update documentation** if needed

### PR Guidelines

1. **Title**: Use the same format as commit messages
2. **Description**: Clearly describe:
   - What changes were made
   - Why they were necessary
   - Any breaking changes
   - Related issues

3. **Testing**: Include:
   - Unit tests for new features
   - Integration tests if applicable
   - Performance benchmarks for optimizations

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No security vulnerabilities introduced
```

## Testing Guidelines

### Test Structure
```python
# tests/unit/test_strategy.py
import pytest
from bot.strategy import MomentumStrategy

class TestMomentumStrategy:
    """Test momentum strategy implementation."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return MomentumStrategy(lookback=20)

    def test_signal_generation(self, strategy, sample_data):
        """Test signal generation logic."""
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert signals.isin([-1, 0, 1]).all()

    @pytest.mark.parametrize("lookback", [10, 20, 50])
    def test_different_lookbacks(self, lookback, sample_data):
        """Test strategy with different lookback periods."""
        strategy = MomentumStrategy(lookback=lookback)
        signals = strategy.generate_signals(sample_data)
        assert signals is not None
```

### Coverage Requirements
- Minimum 80% coverage for new code
- Critical paths must have 100% coverage
- Use `pytest-cov` to measure coverage

## Documentation Standards

### Code Documentation
- All public functions must have docstrings
- Complex algorithms should have inline comments
- Use type hints consistently

### User Documentation
- Update README.md for significant features
- Add examples to `examples/` directory
- Update API documentation if endpoints change

### Architecture Documentation
- Update `docs/ARCHITECTURE.md` for structural changes
- Document design decisions in `docs/decisions/`
- Keep dependency documentation current

## Security Guidelines

### Security Checklist
- [ ] No hardcoded secrets
- [ ] Input validation implemented
- [ ] SQL injection prevention
- [ ] XSS prevention (if applicable)
- [ ] Rate limiting considered
- [ ] Error messages don't leak sensitive info
- [ ] Dependencies are up-to-date

### Reporting Security Issues
**DO NOT** create public issues for security vulnerabilities. Instead:
1. Email security@gpt-trader.com
2. Include detailed description
3. Wait for response before disclosure

## Getting Help

### Resources
- [Documentation](docs/)
- [API Reference](docs/api/)
- [Examples](examples/)
- [Issue Tracker](https://github.com/GPT-Trader/issues)

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and ideas
- Email: dev@gpt-trader.com

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to GPT-Trader! 🚀
