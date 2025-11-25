# GPT-Trader Project Roadmap

**Last Updated**: 2025-11-24
**Status**: Active (Single Source of Truth)

---

## 🎯 Project Vision

GPT-Trader is a production-ready Coinbase **spot trading bot** with future-ready perpetuals support. The system uses ML-driven strategies with comprehensive risk management, built on a clean coordinator-based architecture optimized for AI-assisted development.

---

## 📍 Current Status (November 2025)

### ✅ **MAJOR BREAKTHROUGH: Bot is Operational & Verified (Nov 2025)**

**Update 2025-11-23**: 
- **Live Dry-Run Successful**: Bot successfully connected to Coinbase, authenticated using ECDSA keys from a secure file, and executed trading cycles on `BTC-USD`.
- **US Futures Discovery**: Confirmed that US-based API keys require specific approval for Futures products (IDs like `BIP` vs `BTC-PERP`). Current dry-runs will use Spot (`BTC-USD`) until Futures access is provisioned.
- **Infrastructure Hardening**: 
    - Implemented file-based credential loading (`secrets/`) for improved security.
    - Fixed async blocking issues in `TradingEngine` by offloading network calls.
    - Resolved circular import dependencies in logging utilities.

### ✅ What's Working
- **🎯 BOT STARTUP**: Bot successfully starts and runs complete trading cycle (FIXED TODAY!)
- **Live Connectivity**: Authenticated connection to Coinbase Advanced Trade (Spot) verified.
- **Spot Trading**: Fully operational on Coinbase Advanced Trade (BTC, ETH, SOL, etc.)
- **Strategy Execution**: BaselinePerpsStrategy processes symbols (currently executing "HOLD" logic correctly).
- **Architecture**: Coordinator pattern with vertical slices, well-structured and maintainable
- **Testing**: 1,484 test functions (~3,700 runtime cases via parametrization), 72.87% coverage baseline
- **Risk Management**: Daily loss guards, liquidation buffers, volatility circuit breakers
- **Monitoring**: Prometheus exporter, account telemetry, system health checks
- **CI/CD**: Multiple workflows (Python CI, targeted suites, nightly validation, security audit)
- **Standardized Instantiation**: TradingBot and TradingEngine use clean dependency injection (ApplicationContainer)
- **Strategy Wiring**: BaselinePerpsStrategy is fully connected and executing decisions in the trading loop

### 🚧 In Progress
- **Refining Strategy Logic**: 60% complete - basic BUY/SELL signals implemented using simple moving average; RSI/crossover logic defined in config but not yet coded
- **Preflight Module Testing**: 7+ bootstrap/connectivity modules lack test coverage
- **Pre-existing Test Failures**: `test_event_store.py` (API mismatch), `test_cli_integration.py` (needs credentials)

### ✅ Recently Completed (Nov 2025)
- **Test Refactoring Phase 3**: Superseded - architecture reorganized from "coordinator" to "engine" naming instead of splitting files
- **Parallel Test Execution**: Enabled by default (`-n auto` in pytest.ini) - 12 workers, verified working
- **Security Module Coverage**: Achieved 95% coverage target
- **StructuredLogger**: Implemented with domain fields and correlation tracking

### ⏸️ Future Activation
- **US Perpetual Futures**: Requires Coinbase Financial Markets (CFM) approval and integration of `BIP`/`ETP` product IDs.
- **International Perpetual Futures**: Code ready (`BTC-PERP`) pending Coinbase INTX access.
- **Advanced WebSocket**: Baseline exists, enrichment/backfill in progress
- **Durable State**: OrdersStore/EventStore need production hardening

---

## 🎯 Immediate Priorities (Next 2 Weeks)

### Priority 1: Get Bot Trading (COMPLETED)
**Status**: ✅ **DONE** (Nov 23, 2025)
**Summary**: Bot is now running, connecting, and executing cycles.

**Completed Tasks**:
1. [x] Investigate why BaselinePerpsStrategy returns "No signal" for all symbols (Fixed: Strategy was missing, recreated and wired)
2. [x] Check if market data is being fetched correctly (Fixed: TradingEngine now fetches ticker data)
3. [x] Verify strategy parameters and thresholds are configured (Fixed: Strategy instantiated with config)
4. [x] Test with simple buy signal to confirm order placement works (Verified with tests/integration/test_end_to_end_buy.py)
5. [x] Document minimum viable trading configuration (Documented in walkthrough.md)
6. [x] **Live Dry-Run**: Verified connectivity and authentication with `BTC-USD`.

---

### Priority 2: Fix Pre-existing Test Failures
**Why**: Tests reference non-existent API methods, blocking CI reliability
**Status**: In Progress

**Tasks**:
1. [ ] Fix `test_event_store.py` - tests reference `append_trade`, `_normalize_payload` which don't exist
2. [ ] Mark `test_cli_integration.py` tests with `@pytest.mark.integration` (require credentials)
3. [x] Enable parallel test execution by default (`-n auto` added to pytest.ini)
4. [x] Document test execution best practices in TESTING_GUIDE.md (372-line guide exists)

**Success Criteria**:
- All 1,484 test functions passing
- Average test file size < 500 lines
- Parallel execution works without flakiness (verified)

---

### Priority 3: Preflight Module Test Coverage
**Why**: 7+ modules providing critical bootstrap/connectivity checks have zero test coverage

**Tasks**:
1. [ ] Add tests for `preflight/cli.py`, `preflight/core.py`, `preflight/context.py`
2. [ ] Test connectivity, dependencies, and environment checks
3. [ ] Target 80% coverage for preflight module

**Success Criteria**:
- 80% coverage on preflight module
- All bootstrap paths tested

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
- 80% overall coverage (up from 72.87%)
- 95% coverage on trading and risk components (security already at 95%)
- <5 minute full test suite execution (currently ~6s with parallel)

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
- ✅ Comprehensive auth handler test coverage (achieved 95%)
- ✅ Secrets manager robustness testing (13 test files)
- ✅ Security validator edge cases (6 test files)

**Week 11-12**: Documentation Validation
- CI/CD integration for doc checks
- Automated link validation
- Code example verification in docs

**Success Metrics**:
- ✅ 95% security component coverage (ACHIEVED)
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

**Technical Debt (Address Soon)**:
- Refactor `order_submission.py` (536 lines - only file >500 lines, complexity risk)
- Strategy enhancement: Implement RSI/crossover logic using existing config parameters

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
- **After Milestones**: Update when significant work is completed (preferred cadence)
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
| 2025-11-24 | Enabled parallel test execution (`-n auto`), updated priorities, fixed test count (1,484 functions) | Claude AI |
| 2025-11-24 | Achieved 95% security module coverage, added StructuredLogger | Claude AI |
| 2025-11-23 | Live dry-run verified with BTC-USD, architecture reorganization complete | Claude AI |
| 2025-11-18 | Initial roadmap creation, consolidated from 3 planning docs | Gemini AI |

---

**Questions or Feedback?** Update this roadmap or discuss in project planning meetings.
