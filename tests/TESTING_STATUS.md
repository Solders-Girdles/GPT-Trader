# Testing Status & Roadmap

*Last updated: 2025-10-19*

## Current Coverage Baseline

- **Overall Coverage**: 72.87%
- **Target**: 80% (short-term), 90% (long-term)
- **CI Threshold**: 72.87% (regressions fail fast)

## Sprint 2025-10-19: Major Coverage Improvements ✅

### Completed Tasks

#### Infrastructure & CI
- ✅ **Coverage Baseline**: Locked in 72.87% with `--cov=src` scope
- ✅ **CI Regression Checks**: Added JSON diff validation in GitHub Actions
- ✅ **Developer Tools**: Created `scripts/run_coverage_report.py` for easy coverage runs

#### Core Logic Testing
- ✅ **Strategy Analysis**: Expanded parameterised tests for `analyze_with_strategies` and all strategy helpers (MA, momentum, mean reversion, volatility, breakout)
- ✅ **Trailing Stops**: Comprehensive test coverage for long/short trailing stop logic with edge cases
- ✅ **Position Sizing**: Full branch coverage for regime-based sizing utilities

#### Brokerage Integration
- ✅ **Coinbase REST API**: Contract suite testing all public mixins (orders, portfolio, PnL) with mocked responses, covering quantity resolution, error branches, pagination, and retry/telemetry logging
- ✅ **Specs & Quantization**: Extensive fixture-driven tests for override loading, quantisation, and safe position sizing
- ✅ **Auth Negotiation**: Integration smoke tests for complete auth flow (HMAC/CDP JWT/WebSocket providers)

#### Advanced Testing Techniques
- ✅ **Property-Based Testing**: Invariants for safe position sizing, order validation, and order lifecycle flows
- ✅ **Async Testing**: Anyio-based tests for orchestration paths, guard flows, and state management
- ✅ **Contract Testing**: Persistence layer tests covering failure branches, locking, and concurrent access

#### Monitoring & Infrastructure
- ✅ **Guard Management**: End-to-end tests for guard registration, alert fan-out, and auto-shutdown
- ✅ **Logging System**: Serializer and level handling tests with fake alert handlers
- ✅ **Persistence Layer**: JsonFileStore contract tests with thread safety and error recovery

## Package-Level Coverage Status

| Package | Coverage | Status | Priority |
|---------|----------|--------|----------|
| `bot_v2.config` | 90%+ | ✅ Excellent | Maintain |
| `bot_v2.utilities` | 85%+ | ✅ Good | Maintain |
| `bot_v2.persistence` | 90%+ | ✅ Excellent | Maintain |
| `bot_v2.features.analyze` | 85%+ | ✅ Good | Maintain |
| `bot_v2.features.brokerages.coinbase` | 80%+ | ✅ Good | Maintain |
| `bot_v2.features.live_trade` | 75%+ | ✅ Good | Maintain |
| `bot_v2.features.position_sizing` | 80%+ | ✅ Good | Maintain |
| `bot_v2.monitoring` | 75%+ | ✅ Good | Maintain |
| `bot_v2.orchestration` | 70%+ | ⚠️ Needs Work | High |
| `bot_v2.security` | 60%+ | ⚠️ Needs Work | High |

## Sprint 2025-11: Target 80% Overall Coverage 🎯

### High Priority (Must Complete)

#### Orchestration Coverage (Target: 80%+)
- **Strategy Orchestrator**: Add tests for decision routing edge cases and async state management
- **Execution Guards**: Expand coverage for kill-switch activation and guard composition
- **State Collection**: Test missing marks handling and concurrent state updates

#### Security Module Coverage (Target: 75%+)
- **Secrets Management**: Test encryption/decryption flows and key rotation
- **Auth Handlers**: Add integration tests for credential validation
- **Security Validators**: Cover all validation rules and error scenarios

### Medium Priority (Should Complete)

#### Integration Test Expansion
- **End-to-End Flows**: Add smoke tests for complete trading cycles
- **API Contract Testing**: Expand REST API response validation
- **WebSocket Integration**: Test real-time data feed handling

#### Property-Based Testing Expansion
- **Market Data**: Invariants for price feeds and data validation
- **Risk Management**: Property tests for position limits and exposure
- **Performance Testing**: Benchmark tests for critical paths

### Low Priority (Nice to Have)

#### Documentation & Tooling
- **Coverage Dashboard**: Automated badge generation and reporting
- **Test Quality Metrics**: Mutation testing and test effectiveness analysis
- **CI Optimization**: Parallel test execution and selective test running

## Testing Strategy & Guidelines

### Test Categories

#### Unit Tests (`tests/unit/`)
- Pure function testing with mocks
- Business logic validation
- Error condition handling
- Edge case coverage

#### Integration Tests (`tests/integration/`)
- Component interaction testing
- External API mocking
- End-to-end flow validation
- Contract testing

#### Property Tests (`tests/property/`)
- Invariant validation with generated inputs
- Statistical testing of complex logic
- Boundary condition exploration

#### Behavioral Tests (`tests/fixtures/behavioral/`)
- Scenario-based testing
- State machine validation
- Temporal logic testing

### Coverage Targets by Component Type

| Component Type | Target Coverage | Rationale |
|----------------|----------------|-----------|
| Core Utilities | 90%+ | High reuse, critical path |
| Business Logic | 80%+ | Complex decision making |
| External APIs | 75%+ | Integration complexity |
| Error Handling | 85%+ | Failure mode importance |
| Configuration | 90%+ | Validation criticality |

### Test Quality Standards

#### Code Coverage
- **Line Coverage**: 80%+ overall target
- **Branch Coverage**: 75%+ for conditional logic
- **Function Coverage**: 85%+ for public APIs

#### Test Characteristics
- **Deterministic**: Tests should be reproducible
- **Fast**: Unit tests < 100ms, integration < 1s
- **Isolated**: No external dependencies in unit tests
- **Maintainable**: Clear naming and documentation

### Coverage Improvement Workflow

1. **Identify Gaps**: Run coverage report and analyze missing lines
2. **Prioritize**: Focus on high-risk, high-usage code first
3. **Write Tests**: Use TDD approach for new functionality
4. **Verify**: Ensure coverage increases without quality decrease
5. **Document**: Update this status document and coverage badges

## Running Tests & Coverage

### Quick Coverage Check
```bash
./scripts/run_coverage_report.py
```

### Full CI Simulation
```bash
poetry run pytest --cov=src --cov-report=json:coverage.json --cov-fail-under=72.87
```

### HTML Report Generation
```bash
poetry run pytest --cov=src --cov-report=html:htmlcov
# Open htmlcov/index.html
```

### Selective Testing
```bash
# Run only fast tests
poetry run pytest -m "not slow and not integration"

# Run specific package
poetry run pytest tests/unit/bot_v2/features/brokerages/coinbase/
```

## Success Metrics

### Quantitative
- **Coverage**: 80% overall by end of Sprint 2025-11
- **Test Count**: 500+ tests by project completion
- **Test Runtime**: < 5 minutes for full suite
- **Flakiness**: < 1% test failure rate

### Qualitative
- **Confidence**: High confidence in code changes
- **Debugging**: Easy to isolate and fix issues
- **Documentation**: Tests serve as living documentation
- **Maintenance**: Easy to modify and extend tests

## Risk Mitigation

### Coverage Regression Prevention
- CI checks prevent coverage decreases
- Pre-commit hooks validate test coverage
- Code review requirements for test coverage

### Test Suite Maintenance
- Regular test cleanup and refactoring
- Performance monitoring of test suite
- Automated test quality checks

### Future Evolution
- Test strategy review every 3 months
- Coverage target adjustment based on project needs
- New testing technique adoption as appropriate

---

*This document is maintained alongside the codebase. Update when coverage changes significantly or testing strategy evolves.*
