# Phase 2 - Planning & Priorities

**Status**: 📋 Planning
**Start Date**: TBD
**Phase 1 Completion**: 2025-10-05 ✅
**Prerequisites**: All Phase 1 objectives met

---

## 🎯 Phase 2 Mission

**Primary Goal**: Enhance observability and address remaining technical debt while maintaining the strong test coverage and stability established in Phase 1.

**Key Themes**:
1. **Observability First** - Metrics, dashboards, monitoring
2. **Technical Debt Reduction** - Address deferred items
3. **Production Readiness** - Move toward deployable state
4. **Maintain Quality** - Keep 100% test coverage

---

## 📊 Phase 1 Foundation (What We Built)

### Strong Base to Build On:
- ✅ 59/59 characterization tests (100% coverage, 0.32s runtime)
- ✅ Modular test suite architecture
- ✅ Clean builder pattern implementation
- ✅ Background task lifecycle management validated
- ✅ Operational stability confirmed (2H soak: 0 errors, 4.5% memory growth)

### Known Gaps (From Phase 1):
- ❌ No Prometheus metrics endpoint
- ❌ No Grafana dashboards
- ❌ Mock broker only (no real API validation)
- ❌ No HTTP server for health/metrics
- ❌ Deferred full 48H drift review

---

## 🔍 Phase 2 Candidate Areas

### 1. Observability & Metrics (HIGH PRIORITY)

**Current State**:
- Bot runs with structured JSON logging only
- No Prometheus metrics exposed
- No Grafana dashboards
- Health checks disabled (no HTTP endpoint)

**Proposed Work**:
- [ ] Implement Prometheus metrics endpoint
  - Expose `/metrics` on port 9090
  - Export key metrics (uptime, memory, CPU, cycle time, etc.)
  - Track order execution metrics
  - Monitor background task health

- [ ] Add HTTP health endpoint
  - Implement `/health` endpoint on port 8080
  - Return bot status, component health
  - Enable proper Docker health checks

- [ ] Create Grafana dashboards
  - Bot performance dashboard
  - Order execution dashboard
  - Risk management dashboard
  - Infrastructure health dashboard

- [ ] Set up alerting rules
  - Memory leak detection
  - CPU spike alerts
  - Background task failures
  - Order execution errors

**Effort**: Medium (2-4 hours)
**Impact**: High (enables production monitoring)
**Risk**: Low (additive, doesn't touch core logic)

---

### 2. Real Broker Integration (MEDIUM PRIORITY)

**Current State**:
- Using mock broker for safety
- No real API latency metrics
- No WebSocket streaming validation
- Limited production signal

**Proposed Work**:
- [ ] Configure Coinbase Advanced Trade broker for staging
  - Use staging API credentials
  - Enable WebSocket streaming
  - Configure rate limit handling
  - Test order preview/execution (paper mode)

- [ ] Add broker integration tests
  - API connectivity tests
  - WebSocket stability tests
  - Rate limit compliance tests
  - Order lifecycle tests (preview, place, cancel)

- [ ] Streaming service validation
  - Enable real-time mark updates
  - Measure WebSocket lag
  - Test reconnection logic
  - Validate mark window updates

**Effort**: Medium (3-5 hours)
**Impact**: High (production-realistic metrics)
**Risk**: Medium (requires API credentials, rate limits)

---

### 3. Technical Debt Cleanup (MEDIUM PRIORITY)

**From Phase 1 Drift Review**:
- [ ] Unit test import errors (seen during Docker build)
  - Fix imports for strategy tests
  - Update test fixtures
  - Ensure all unit tests pass in Docker

- [ ] Test stage failures in Dockerfile
  - Investigate strategy test import errors
  - Fix adaptive_portfolio test issues
  - Enable testing stage in CI/CD

**From Refactoring Backlog**:
- [ ] Review and address TODO comments
- [ ] Clean up deprecated code paths
- [ ] Update stale documentation
- [ ] Consolidate duplicate code

**Effort**: Low-Medium (2-4 hours)
**Impact**: Medium (code quality, maintainability)
**Risk**: Low (mostly cleanup)

---

### 4. Performance Optimization (LOW PRIORITY)

**Potential Improvements**:
- [ ] Optimize mark update frequency
- [ ] Cache frequently accessed data
- [ ] Reduce unnecessary logging
- [ ] Optimize database queries

**Note**: Phase 1 showed excellent performance (1.2% CPU, 2.8GB memory stable), so this is low priority unless bottlenecks emerge.

**Effort**: Variable
**Impact**: Low (already performant)
**Risk**: Medium (optimization can introduce bugs)

---

### 5. Additional Refactoring (DEFERRED)

**Candidates from Surveys**:
- Advanced execution analysis
- Liquidity service refactoring
- PnL tracker improvements
- Portfolio valuation enhancements

**Recommendation**: Defer to Phase 3+
- Phase 2 focus: Observability + production readiness
- Major refactors can wait until monitoring is solid
- Current code is stable and well-tested

---

## 🎯 Recommended Phase 2 Scope

### Core Scope (Must-Have):

1. **Metrics Endpoint Implementation** (3-4 hours)
   - Prometheus metrics exporter
   - HTTP health endpoint
   - Basic system metrics
   - Order/cycle metrics

2. **Grafana Dashboards** (2-3 hours)
   - Bot performance dashboard
   - Order execution dashboard
   - Infrastructure health dashboard

3. **Unit Test Fixes** (1-2 hours)
   - Fix strategy test imports
   - Enable testing stage in Docker
   - Ensure all tests pass

**Total Core Effort**: 6-9 hours

### Extended Scope (Nice-to-Have):

4. **Real Broker Integration** (3-5 hours)
   - Staging Coinbase credentials
   - WebSocket streaming enabled
   - Broker integration tests

5. **Alert Rules** (1-2 hours)
   - Prometheus alert definitions
   - Memory/CPU thresholds
   - Error rate alerts

**Total Extended Effort**: 4-7 hours

### Phase 2 Total: 10-16 hours (vs Phase 1: 5 hours)

---

## 📋 Success Criteria

### Must Meet:
- [ ] Prometheus metrics endpoint functional
- [ ] HTTP health endpoint implemented
- [ ] At least 2 Grafana dashboards deployed
- [ ] All unit tests passing (including in Docker)
- [ ] 100% test coverage maintained (59+ tests)
- [ ] No new critical errors introduced

### Nice-to-Have:
- [ ] Real broker integrated (staging)
- [ ] WebSocket streaming validated
- [ ] Alert rules configured
- [ ] Technical debt reduced by 50%

### Quality Gates:
- [ ] All tests pass (no regressions)
- [ ] Memory usage stable (< 10% growth from Phase 1)
- [ ] CPU usage stable (< 5% average)
- [ ] Documentation updated
- [ ] No deprecation warnings

---

## 🚀 Execution Plan

### Phase 2A: Observability (Week 1)

**Goals**:
1. Implement metrics endpoint
2. Deploy Grafana dashboards
3. Enable health checks

**Deliverables**:
- Working `/metrics` endpoint
- 2+ Grafana dashboards
- HTTP health endpoint
- Updated Docker config

### Phase 2B: Validation (Week 2)

**Goals**:
1. Fix unit test issues
2. Run full 48H drift review (with metrics)
3. Optional: Real broker integration

**Deliverables**:
- All tests passing
- 48H drift review complete
- Production readiness assessment
- Phase 2 completion report

---

## 🎓 Lessons from Phase 1 to Apply

### What Worked:
1. **Pragmatic Scoping**: Short soak vs 48H (applied same logic to Phase 2)
2. **Strong Testing**: 100% coverage provided confidence (maintain this)
3. **Clear Documentation**: Well-documented decisions (continue this)
4. **Under-Promise**: 5h vs 12-20h budgeted (realistic Phase 2 estimates)

### What to Improve:
1. **Test Infrastructure**: Fix Docker test stage upfront
2. **Health Monitoring**: Implement metrics from day 1
3. **Real Workloads**: Use real broker earlier for realistic signals

---

## 🤔 Open Questions

### For Discussion:

1. **Broker Credentials**:
   - Do we have staging Coinbase API credentials available?
   - What rate limits apply to staging?
   - Paper trading mode enabled?

2. **Observability Stack**:
   - Are there existing Grafana instances to connect to?
   - Should we use Prometheus pushgateway or exporter?
   - Any organization-wide metrics standards?

3. **Deployment Target**:
   - Is production deployment a Phase 2 goal?
   - Or just "production-ready" with deployment in Phase 3?
   - What's the minimum viable production feature set?

4. **Testing Strategy**:
   - Should we add integration tests with real broker?
   - Or keep mocks and defer to staging validation?
   - What's the right balance of speed vs realism?

---

## 📊 Risk Assessment

### Low Risk:
- ✅ Metrics endpoint (additive, well-understood)
- ✅ Grafana dashboards (visualization only)
- ✅ Unit test fixes (isolated, testable)

### Medium Risk:
- ⚠️ Real broker integration (API limits, credentials, costs)
- ⚠️ Health endpoint (potential port conflicts, routing)
- ⚠️ Alert rules (false positives, tuning required)

### High Risk:
- ❌ Major refactoring (deferred to Phase 3+)
- ❌ Architecture changes (not in Phase 2 scope)
- ❌ Production deployment (validation only in Phase 2)

**Mitigation**: Start with low-risk items, add medium-risk incrementally, defer high-risk

---

## 🎯 Recommendation

### Proposed Phase 2 Approach:

**Week 1: Observability Core**
1. Implement Prometheus metrics endpoint (3-4h)
2. Create Grafana dashboards (2-3h)
3. Add HTTP health endpoint (1h)
4. Fix unit test issues (1-2h)

**Week 2: Validation & Polish**
1. Run 48H drift review with metrics (2h setup + 48h soak)
2. Optional: Real broker integration (3-5h)
3. Document findings (1-2h)
4. Phase 2 completion report (1h)

**Total Effort**: 10-16 hours over 2 weeks
**Risk Level**: Low-Medium
**Success Probability**: High (building on Phase 1 success)

---

## ✅ Ready to Begin?

**Prerequisites Check**:
- ✅ Phase 1 complete
- ✅ Test suite stable (59/59 passing)
- ✅ Deployment infrastructure working
- ✅ Team capacity available

**Next Steps**:
1. Review and approve Phase 2 scope
2. Clarify open questions (credentials, deployment target)
3. Allocate time/resources
4. Begin with metrics endpoint implementation

**Estimated Timeline**: 2 weeks (10-16 hours)
**Target Completion**: Mid-October 2025

---

**Prepared By**: Phase 2 Planning Team
**Planning Date**: 2025-10-05
**Approval Status**: Pending stakeholder review
**Next Action**: Scope approval & kickoff
