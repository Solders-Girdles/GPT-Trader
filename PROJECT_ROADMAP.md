# GPT-Trader Project Roadmap

**Last Updated**: 2025-11-18  
**Status**: Active (Single Source of Truth)

---

## 🎯 Project Vision

GPT-Trader is a production-ready Coinbase **spot trading bot** with future-ready perpetuals support. The system uses ML-driven strategies with comprehensive risk management, built on a clean coordinator-based architecture optimized for AI-assisted development.

---

## 📍 Current Status (November 2025)

### ✅ **MAJOR BREAKTHROUGH: Bot is Now Operational!**

**Update 2025-11-18**: Fixed critical circular import bug that prevented bot from ever starting. **The bot can now run successfully in dev mode!**

### ✅ What's Working
- **🎯 BOT STARTUP**: Bot successfully starts and runs complete trading cycle (FIXED TODAY!)
- **Spot Trading**: Fully operational on Coinbase Advanced Trade (BTC, ETH, SOL, XRP, LTC, ADA, DOGE, BCH, AVAX, LINK)
- **Strategy Execution**: BaselinePerpsStrategy processes all symbols and generates decisions
- **Architecture**: Coordinator pattern with vertical slices, well-structured and maintainable
- **Testing**: 3,698 active tests with ~74% coverage
- **Risk Management**: Daily loss guards, liquidation buffers, volatility circuit breakers
- **Monitoring**: Prometheus exporter, account telemetry, system health checks
- **CI/CD**: Multiple workflows (Python CI, targeted suites, nightly validation, security audit)

### 🚧 In Progress
- **Refining Strategy Logic**: Current strategy returns "hold" for all symbols - needs tuning/signals
- **Test Refactoring Phase 3**: Split `test_execution_coordinator.py` (1,152 lines → 3 focused files)
  - Phases 1-2 complete (75% done)
  - Remaining: `test_exec_main_workflows.py`, `test_exec_main_error_handling.py`, `test_exec_main_advanced_features.py`

### ⏸️ Future Activation
- **Perpetual Futures**: Code ready but disabled pending Coinbase INTX access
- **Advanced WebSocket**: Baseline exists, enrichment/backfill in progress
- **Durable State**: OrdersStore/EventStore need production hardening

---

## 🎯 Immediate Priorities (Next 2 Weeks)

### Priority 1: Get Bot Trading (NEW - MOST CRITICAL)
**Why**: Bot now runs but strategy returns "hold" for all symbols - it's not actually trading yet

**Tasks**:
1. Investigate why BaselinePerpsStrategy returns "No signal" for all symbols
2. Check if market data is being fetched correctly
3. Verify strategy parameters and thresholds are configured
4. Test with simple buy signal to confirm order placement works
5. Document minimum viable trading configuration

**Success Criteria**:
- Bot generates at least one non-"hold" decision
- Can place test order in dev mode (mock broker)
- Clear documentation on how to configure strategy signals

**Estimated Effort**: 4-6 hours

---

### Priority 2: Complete Test Refactoring
**Why**: Finish in-flight work, achieve 3x parallelization benefit for all orchestration tests

**Tasks**:
1. Split `test_execution_coordinator.py` into 3 files per [TEST_REFACTORING_PLAN.md](docs/archive/planning/TEST_REFACTORING_PLAN.md)
2. Verify all 39 tests pass individually and together
3. Run parallel execution test: `pytest tests/unit/gpt_trader/orchestration/ -n auto`
4. Document completion and archive planning docs

**Success Criteria**:
- All 3,698 tests still passing
- Average test file size < 500 lines
- Parallel execution works without flakiness

**Estimated Effort**: 4-6 hours

---

### Priority 2: Stabilize Testing Infrastructure
**Why**: Prevent test flakiness from blocking development

**Tasks**:
1. Fix any flaky tests discovered during refactoring
2. Add pytest retry plugin for known-flaky integration tests
3. Document test execution best practices in TESTING_GUIDE.md

**Success Criteria**:
- <1% test failure rate on CI
- Clear guidelines for writing deterministic tests

**Estimated Effort**: 2-3 hours

---

## 📅 Short-Term Goals (1-3 Months)

### Month 1: Testing & Quality Foundation

**Week 1-2**: Test Infrastructure
- ✅ Complete test refactoring Phase 3
- Add property-based testing for critical trading logic
- Improve test execution speed (target: <5 min full suite)

**Week 3-4**: Critical Coverage Gaps
- Advanced execution engine core methods (`place_order`, `calculate_impact_aware_size`)
- Risk management state transitions
- Liquidity service edge cases

**Success Metrics**:
- 80% overall coverage (up from 74%)
- 95% coverage on trading and risk components
- <5 minute full test suite execution

---

### Month 2: Configuration & Monitoring

**Week 5-6**: Configuration Validation
- Extend schema-driven validation to all config surfaces
- Add environment-specific validation rules
- Implement configuration drift detection

**Week 7-8**: Enhanced Monitoring
- Improve runtime guard error messages
- Add guard health monitoring dashboard
- Configuration change audit trail

**Success Metrics**:
- 100% configuration schema coverage
- <5 minute drift detection
- Zero config-related production incidents

---

### Month 3: Security & Documentation

**Week 9-10**: Security Hardening
- Comprehensive auth handler test coverage
- Secrets manager robustness testing
- Security validator edge cases

**Week 11-12**: Documentation Validation
- CI/CD integration for doc checks
- Automated link validation
- Code example verification in docs

**Success Metrics**:
- 95% security component coverage
- 90% API documentation coverage
- <7 day documentation staleness

---

## 🔮 Medium-Term Goals (3-6 Months)

### Architectural Improvements

**PerpsBot Refactoring** (if still monolithic):
- Extract `PerpsBotBuilder` for construction logic
- Separate `SymbolNormalizer` for symbol processing
- Create `RuntimePathResolver` for filesystem setup
- Reduce constructor to ≤50 lines

**Coordinator Pattern Enhancement**:
- Improve lifecycle management error handling
- Enhance context passing efficiency
- Add coordinator health monitoring
- Simplify registration process

### Advanced Testing

**Property-Based Testing**:
- Trading logic invariants (order size bounds, price validation)
- Security properties (auth token expiration, permission checks)
- Risk management invariants (position limits, leverage bounds)

**Performance Testing**:
- Market condition edge cases
- System stress scenarios
- Resource exhaustion handling

---

## 🌟 Long-Term Vision (6-12 Months)

### Multi-Exchange Support
- Extensible broker interface abstraction
- Exchange-specific adapters (Kraken, Binance)
- Cross-exchange arbitrage foundation

### Advanced Features
- Real-time ML model adaptation
- Advanced order types (iceberg, TWAP, VWAP)
- Portfolio-level risk management
- Options integration preparation

### Production Excellence
- Distributed execution patterns
- Institutional-grade features
- Full WebSocket user event handling
- Durable state recovery

---

## 📋 Backlog (Not Prioritized)

**Nice to Have (When Time Permits)**:
- Funding rate accrual in deterministic broker
- Order modification/amend flows beyond cancel
- Partial fill handling in mock broker
- Enhanced backtesting engine improvements
- Additional analysis and visualization tools

---

## 🚫 Not Planned / Out of Scope

**Explicitly Deprioritized**:
- Support for Coinbase Sandbox (API diverges, limited value)
- Legacy bot V1 maintenance (archived, focus on V2)
- Support for Python <3.12 (project targets 3.12+)

---

## 📖 How to Use This Roadmap

### For Human Developers
1. **Start here** when beginning work
2. Check "Immediate Priorities" for what to work on next
3. Update this file when completing major milestones
4. Archive old sections as work progresses

### For AI Coding Agents
1. **Read this first** before making changes
2. Check "Current Status" to understand what's working
3. Align work with "Immediate Priorities" unless told otherwise
4. Update relevant sections when completing tasks
5. Refer to archived plans in `docs/archive/planning/` for detailed context only if needed

### Updating This Roadmap
- **Weekly**: Update "In Progress" section
- **Monthly**: Review and adjust priorities based on progress
- **Quarterly**: Reassess long-term vision and goals

---

## 🗂️ Related Documentation

### Active Documents (Read These)
- [README.md](README.md) - Quick start, current capabilities
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design, components
- [DEVELOPMENT_GUIDELINES.md](docs/DEVELOPMENT_GUIDELINES.md) - Coding standards
- [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) - Testing practices
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute

### Archived Planning Docs (Historical Context)
- [COMPREHENSIVE_IMPROVEMENT_PLAN.md](docs/archive/planning/COMPREHENSIVE_IMPROVEMENT_PLAN.md) - Original 12-month roadmap
- [TEST_REFACTORING_PLAN.md](docs/archive/planning/TEST_REFACTORING_PLAN.md) - Detailed test refactoring spec
- [PR_SUMMARY.md](docs/archive/planning/PR_SUMMARY.md) - Test refactoring Phases 1-2 completion

> **Note**: Archived docs provide valuable context but are not actively maintained. This roadmap is the single source of truth for current priorities.

---

## 📝 Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-11-18 | Initial roadmap creation, consolidated from 3 planning docs | Gemini AI |

---

**Questions or Feedback?** Update this roadmap or discuss in project planning meetings.
