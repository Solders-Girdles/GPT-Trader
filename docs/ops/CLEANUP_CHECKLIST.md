# Operational Audit & Cleanup Checklist - Q4 2025

**Branch:** `cleanup/operational-audit-q4-2025`
**Start Date:** 2025-10-05
**Target Completion:** 2025-11-02 (4 weeks + buffer)
**Status:** ✅ Week 1 In Progress

---

## 📋 Tracker Template

| Task | Owner | Status | Dependencies | Target Date | Risk Level | Decision Log |
|------|-------|--------|--------------|-------------|------------|--------------|
| W1: Validation checks | Architecture | ✅ Complete | - | 2025-10-05 | 🟢 LOW | backups/ confirmed active, BackupScheduler in use |
| W1: Create tracking docs | Architecture | 🔄 In Progress | - | 2025-10-06 | 🟢 LOW | - |
| W1: Dependency audit | Platform | ⏳ Pending | - | 2025-10-07 | 🟢 LOW | - |
| W2: Retention policy | Ops | ⏳ Pending | W1 complete | 2025-10-13 | 🟢 LOW | - |
| W2: Cleanup script | Ops | ⏳ Pending | Retention policy | 2025-10-14 | 🟡 MEDIUM | Must include dry-run |
| W2: Integration test plan | QA + Trading | ⏳ Pending | - | 2025-10-15 | 🟢 LOW | - |
| W2: Sandbox access | Trading Ops | ⏳ Pending | - | 2025-10-15 | 🟡 MEDIUM | Blocker for live tests |
| W3: Integration tests | QA | ⏳ Pending | Test plan | 2025-10-22 | 🟡 MEDIUM | Start with mocks |
| W3: Pytest markers | QA | ⏳ Pending | - | 2025-10-20 | 🟢 LOW | - |
| W3: Update ARCHITECTURE.md | Architecture | ⏳ Pending | - | 2025-10-22 | 🟢 LOW | - |
| W3: Update README.md | Architecture | ⏳ Pending | - | 2025-10-22 | 🟢 LOW | - |
| W3: Update REFACTORING_RUNBOOK | Architecture | ⏳ Pending | - | 2025-10-22 | 🟢 LOW | - |
| W4: Pre-commit setup | Platform | ⏳ Pending | - | 2025-10-29 | 🟡 MEDIUM | Incremental apply |
| W4: Trading ops validation | Trading Ops | ⏳ Pending | Sandbox access | 2025-10-30 | 🟡 MEDIUM | Requires environment |
| W4: Monitoring validation | Ops | ⏳ Pending | - | 2025-10-30 | 🟢 LOW | - |
| W4: Governance docs | Platform | ⏳ Pending | - | 2025-10-31 | 🟢 LOW | - |
| W4: Feature flag cleanup | Architecture | ⏳ Pending | - | 2025-10-31 | 🟡 MEDIUM | Remove retired flags |

---

## Week 1: Reconcile & Map (Oct 5-11)

### ✅ Validation Checks
- [x] Check backups/ directory usage → **ACTIVE (Oct 4 timestamps)**
- [x] Locate BackupScheduler references → **Found in operations.py:221**
- [x] Identify operational directories → **backups/, logs/, cache/, data/, data_storage/**
- [x] Review dependency constraints → **numpy<2, websockets<16**

### 🔄 Documentation Tasks
- [x] Create CLEANUP_CHECKLIST.md (this document)
- [ ] Create CLEANUP_PROGRESS_REPORT.md
- [ ] Create CODEBASE_HEALTH_ASSESSMENT.md
- [ ] Document dependency policy (numpy, websockets constraints)

### 🔍 Key Findings
1. **backups/** is ACTIVE - managed by BackupScheduler, contains timestamped backups
2. **logs/, cache/** are empty - safe for retention policy
3. **data/, data_storage/** contain structured data (universe/, metadata/, ohlcv/)
4. **Pre-commit already in dev dependencies** - ready for activation
5. **responses library available** - can mock Coinbase API for tests

---

## Week 2: Operational Audit & Test Design (Oct 12-18)

### Retention Policy Design
- [ ] Define lifecycle: runtime → archival → deletion
- [ ] Set triggers: age-based, size-based, event-based
- [ ] Document recovery SLA
- [ ] Create audit trail spec (what/when/why)

### Cleanup Script (cleanup_artifacts.sh)
- [ ] Implement --dry-run mode (default)
- [ ] Add --confirm interactive mode
- [ ] Set --age-threshold parameter
- [ ] Add --backup-first archival
- [ ] Include --exclude-patterns safelist
- [ ] Test on empty directories first

### Integration Test Plan
- [ ] Review tests/integration/ structure
- [ ] Design broker streaming failover tests
- [ ] Design guardrails integration tests
- [ ] Design WebSocket/REST fallback tests
- [ ] Document mock strategy using responses library
- [ ] Confirm Coinbase sandbox access

### ⚠️ End of Week 2 Review Gate
- [ ] Retention policy approved by ops team
- [ ] Cleanup script dry-run successful
- [ ] Test plan reviewed by QA + trading engineers
- [ ] No blockers for Week 3

---

## Week 3: Integration Coverage & Doc Sync (Oct 19-25)

### Integration Tests
**tests/integration/brokerages/**
- [ ] test_coinbase_streaming_failover.py

**tests/integration/orchestration/**
- [ ] test_guardrails_integration.py
- [ ] test_broker_outage_handling.py

**tests/integration/streaming/**
- [ ] test_websocket_rest_fallback.py

### Pytest Configuration
- [ ] Update pytest.ini with markers:
  - `@pytest.mark.integration`
  - `@pytest.mark.slow`
  - `@pytest.mark.soak`

### Documentation Sync
- [ ] ARCHITECTURE.md - Update Phase 0-3 extractions
  - MarketDataService moved to features/market_data/
  - StreamingService extraction
  - PerpsBotBuilder pattern
- [ ] README.md - Validate quickstart command
  - `poetry run perps-bot --profile dev --dev-fast`
- [ ] REFACTORING_2025_RUNBOOK.md - Feature flag audit
  - Mark retired: USE_NEW_MARKET_DATA_SERVICE
  - Mark retired: USE_NEW_STREAMING_SERVICE
  - Document active: USE_PERPS_BOT_BUILDER

### ⚠️ End of Week 3 Review Gate
- [ ] Integration tests passing in CI
- [ ] Doc updates reviewed by architecture lead
- [ ] No production incidents
- [ ] Pre-commit strategy validated

---

## Week 4: Governance & Validation (Oct 26 - Nov 2)

### Pre-commit Activation (Incremental)
**Phase 1: Impact Assessment**
- [ ] Run `pre-commit run --all-files --show-diff-on-failure > precommit_report.txt`
- [ ] Review report, identify exclusions needed

**Phase 2: Apply Fixes**
- [ ] `pre-commit run black --all-files` (formatting)
- [ ] `pre-commit run ruff --all-files` (linting)
- [ ] `pre-commit run mypy --all-files` (typing)

**Phase 3: Commit in Batches**
- [ ] Commit formatting changes
- [ ] Commit linting fixes
- [ ] Commit type improvements

### Trading Operations Validation
- [ ] `poetry run perps-bot --account-snapshot` (fees, limits, permissions)
- [ ] Review `config/risk/spot_top10.json` (symbol limits)
- [ ] Audit `features/live_trade/risk/` (hardcoded limits)
- [ ] Run `scripts/deploy_sandbox_soak.sh` (deployment automation)
- [ ] Execute SOAK_TEST_QUICKSTART.md steps
- [ ] Check Grafana dashboards (monitoring/grafana/dashboards/)
- [ ] Test Prometheus alerts (monitoring/alertmanager/alertmanager.yml)

### Monitoring Documentation Validation
- [ ] docs/MONITORING_PLAYBOOK.md
- [ ] docs/guides/monitoring.md
- [ ] docs/ops/operations_runbook.md

### Governance Documentation
- [ ] Create docs/ops/governance.md
  - Pre-commit usage guidelines
  - Dependency update process
  - CI expectations
- [ ] Create docs/ops/dependency_policy.md
  - Pinned version rationale
  - Review cadence
  - Update procedures

### Feature Flag Cleanup
- [ ] Search for USE_NEW_MARKET_DATA_SERVICE references
- [ ] Search for USE_NEW_STREAMING_SERVICE references
- [ ] Remove conditional branches if fully retired
- [ ] Update REFACTORING_2025_RUNBOOK.md status

---

## 🚦 Risk Assessment

| Phase | Risk Level | Mitigation |
|-------|------------|------------|
| Week 1 | 🟢 LOW | Discovery only, no changes |
| Week 2 | 🟢 LOW | Dry-run mode, interim review |
| Week 3 | 🟡 MEDIUM | Integration tests might be flaky; use mocks initially |
| Week 4 | 🟡 MEDIUM | Pre-commit formatting; apply incrementally |

**Elevated Risks:**
1. **Operational Directory Deletion** - backups/ active
   - ✅ Mitigation: Confirmed BackupScheduler behavior; Week 2 dry-run validates
2. **Pre-commit Widespread Changes** - 692 Python files
   - ⏳ Mitigation: Apply by tool (black → ruff → mypy), commit in batches
3. **Soak Test Environment** - Sandbox might not be provisioned
   - ⏳ Mitigation: Validate access during Week 2; use local mocks if unavailable

---

## 📊 Success Criteria

- [ ] All operational directories have documented retention policies
- [ ] Cleanup script tested and operational (dry-run validated)
- [ ] Integration test coverage >70% for critical paths
- [ ] All docs reflect current architecture (Phases 0-3)
- [ ] Pre-commit hooks active without breaking CI
- [ ] Trading ops validated in sandbox environment
- [ ] Monitoring documentation current and accurate
- [ ] Governance processes documented and communicated
- [ ] Retired feature flags removed from codebase

---

## 📅 Next Steps After Completion

1. Merge all PRs to main (after review)
2. Update CHANGELOG.md with cleanup summary
3. Archive plan documents to `docs/archive/cleanup-2025-q4/`
4. Conduct retrospective
5. Update runbooks with lessons learned
