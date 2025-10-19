# Coverage Badges

This document provides per-package coverage badges and status for the GPT-Trader codebase.

## Overall Coverage

![Overall Coverage](https://img.shields.io/badge/coverage-72.87%25-brightgreen)
![CI Status](https://img.shields.io/badge/CI-passing-brightgreen)

*Last updated: 2025-10-19*

## Package-Level Coverage

### Core Components

| Package | Coverage | Status | Description |
|---------|----------|--------|-------------|
| `bot_v2.config` | ![90%+](https://img.shields.io/badge/coverage-90%2B%25-brightgreen) | ✅ Excellent | Configuration management and validation |
| `bot_v2.utilities` | ![85%+](https://img.shields.io/badge/coverage-85%2B%25-brightgreen) | ✅ Good | Shared utilities and helpers |
| `bot_v2.persistence` | ![90%+](https://img.shields.io/badge/coverage-90%2B%25-brightgreen) | ✅ Excellent | Data persistence layer |

### Features

| Package | Coverage | Status | Description |
|---------|----------|--------|-------------|
| `bot_v2.features.analyze` | ![85%+](https://img.shields.io/badge/coverage-85%2B%25-brightgreen) | ✅ Good | Strategy analysis and signals |
| `bot_v2.features.brokerages.coinbase` | ![80%+](https://img.shields.io/badge/coverage-80%2B%25-brightgreen) | ✅ Good | Coinbase integration and APIs |
| `bot_v2.features.live_trade` | ![75%+](https://img.shields.io/badge/coverage-75%2B%25-brightgreen) | ✅ Good | Live trading strategies |
| `bot_v2.features.position_sizing` | ![80%+](https://img.shields.io/badge/coverage-80%2B%25-brightgreen) | ✅ Good | Position sizing logic |

### Infrastructure

| Package | Coverage | Status | Description |
|---------|----------|--------|-------------|
| `bot_v2.orchestration` | ![70%+](https://img.shields.io/badge/coverage-70%2B%25-yellow) | ⚠️ Needs Work | System orchestration and coordination |
| `bot_v2.monitoring` | ![75%+](https://img.shields.io/badge/coverage-75%2B%25-brightgreen) | ✅ Good | System monitoring and alerting |
| `bot_v2.security` | ![60%+](https://img.shields.io/badge/coverage-60%2B%25-yellow) | ⚠️ Needs Work | Security and authentication |

## Coverage Goals

- **Target**: 80% overall coverage
- **Stretch Goal**: 90% overall coverage
- **Minimum Acceptable**: 70% for any package

## Recent Improvements

### Sprint 2025-10-19
- ✅ Locked in 72.87% baseline with CI regression checks
- ✅ Expanded strategy analysis test coverage (+15% in analyze module)
- ✅ Added comprehensive brokerage specs and sizing tests
- ✅ Implemented property-based testing for critical invariants
- ✅ Added auth negotiation smoke tests
- ✅ Strengthened persistence layer contract tests
- ✅ Added developer coverage reporting script

### Next Sprint Targets (2025-11)
- 🎯 Reach 80% overall coverage
- 🎯 Improve orchestration coverage to 80%+
- 🎯 Enhance security module test coverage
- 🎯 Add integration test coverage for end-to-end flows

## Running Coverage Reports

### Quick Terminal Report
```bash
poetry run pytest --cov=src --cov-report=term-missing
```

### Full HTML Report
```bash
poetry run pytest --cov=src --cov-report=html:htmlcov --cov-report=term-missing
# Open htmlcov/index.html in browser
```

### Developer Script
```bash
./scripts/run_coverage_report.py
```

### CI Coverage Check
```bash
poetry run pytest --cov=src --cov-report=json:coverage.json --cov-fail-under=72.87
```

## Coverage Configuration

Coverage settings are configured in `pytest.ini`:

```ini
# Coverage configuration (when using --cov)
# Run with: pytest --cov=src --cov-report=term-missing --cov-report=json:coverage.json --cov-report=html:htmlcov
# Current baseline: 72.87% (as of 2025-10-19)
# Short-term goal: 80%
# Long-term goal: 90%
```

## Contributing to Coverage

When adding new code:

1. **Write tests first** - Aim for 80%+ coverage on new features
2. **Test edge cases** - Include error conditions and boundary values
3. **Use property-based testing** - For complex business logic
4. **Run coverage locally** - Use `./scripts/run_coverage_report.py`
5. **Update this document** - When coverage changes significantly

## Coverage Exclusions

The following are intentionally excluded from coverage:

- Test files themselves (`tests/`)
- Documentation and examples (`docs/`, `scripts/`)
- Third-party code and generated files
- Abstract base classes with no implementation
- Defensive error handling for truly unreachable code paths

## Troubleshooting

### Coverage Not Updating
- Clear pytest cache: `poetry run pytest --cache-clear`
- Remove old coverage files: `rm -rf .coverage coverage.xml htmlcov/`

### Inconsistent Results
- Ensure all tests run: `poetry run pytest --cov=src --cov-report=term-missing -x`
- Check for import issues or skipped tests

### CI Failures
- Coverage regression detected - review recent changes
- Check `coverage.json` for detailed breakdown
- Use HTML report to identify uncovered lines
