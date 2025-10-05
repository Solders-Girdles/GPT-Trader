# Coverage Heatmap

**Generated**: 2025-10-05
**Purpose**: Identify testing gaps per feature slice

---

## Summary

**Overall Coverage**: 89.50%
**Total Statements**: 23,881
**Executed**: 21,373
**Missing**: 2,508

### Coverage Status Legend

- 🟢 **Excellent** (≥90%): Well-tested, minimal gaps
- 🟡 **Good** (70-89%): Adequate coverage, some gaps
- 🔴 **Needs Work** (<70%): Significant testing gaps

---

## Per-Slice Coverage

| Slice | Coverage | Statements | Executed | Missing | Status |
|-------|----------|------------|----------|---------|--------|
| core | 25.0% | 8 | 2 | 6 | 🔴 |
| features/adaptive_portfolio | 83.0% | 1,040 | 863 | 177 | 🟡 |
| state | 85.3% | 4,239 | 3,617 | 622 | 🟡 |
| config | 86.5% | 289 | 250 | 39 | 🟡 |
| features/brokerages | 86.7% | 2,744 | 2,380 | 364 | 🟡 |
| errors | 86.9% | 259 | 225 | 34 | 🟡 |
| data_providers | 88.0% | 415 | 365 | 50 | 🟡 |
| orchestration | 88.6% | 3,538 | 3,135 | 403 | 🟡 |
| security | 88.6% | 493 | 437 | 56 | 🟡 |
| monitoring | 88.8% | 2,138 | 1,898 | 240 | 🟡 |
| features/position_sizing | 89.2% | 752 | 671 | 81 | 🟡 |
| logging | 89.8% | 59 | 53 | 6 | 🟡 |
| features/live_trade | 92.3% | 3,786 | 3,495 | 291 | 🟢 |
| persistence | 92.5% | 199 | 184 | 15 | 🟢 |
| features/analyze | 94.4% | 589 | 556 | 33 | 🟢 |
| cli | 95.2% | 609 | 580 | 29 | 🟢 |
| features/data | 96.4% | 529 | 510 | 19 | 🟢 |
| utilities | 96.4% | 56 | 54 | 2 | 🟢 |
| features/paper_trade | 96.8% | 989 | 957 | 32 | 🟢 |
| features/strategies | 98.0% | 198 | 194 | 4 | 🟢 |
| features/optimize | 98.9% | 459 | 454 | 5 | 🟢 |
| features/strategy_tools | 100.0% | 128 | 128 | 0 | 🟢 |
| types | 100.0% | 65 | 65 | 0 | 🟢 |
| validation | 100.0% | 300 | 300 | 0 | 🟢 |

---

## Detailed Slice Analysis

### 🔴 core (25.0%)

**Priority**: HIGH - Needs immediate attention

#### Lowest Coverage Modules

| Module | Coverage | Statements | Missing |
|--------|----------|------------|---------|
| __main__.py | 0.0% | 2 | 2 |
| cli.py | 0.0% | 4 | 4 |
| __init__.py | 100.0% | 2 | 0 |


---

## Testing Backlog (Priority Order)

Based on coverage gaps and module criticality:

1. **core** - 25.0% (2 modules below 70%)

---

## Recommended Actions

### Immediate (This Sprint)

2. **Establish Baseline Tests**
   - Each module should have at least 70% coverage
   - Focus on happy path + error handling

3. **Integration Tests**
   - Build scenario tests for broker/orchestration interactions
   - Use recorded Coinbase fixtures for deterministic tests

### Phase 1 Goals

- [ ] Bring all feature slices to ≥80% coverage
- [ ] Add integration tests for orchestration layer
- [ ] Establish coverage gates in CI (fail if coverage drops)

### Phase 2 Goals

- [ ] Achieve ≥90% coverage across all features
- [ ] Add property-based tests for critical calculations (risk, fees)
- [ ] Build comprehensive scenario test suite

---

**Next Steps**: See `docs/testing/coverage_backlog.md` for detailed module-level tasks
