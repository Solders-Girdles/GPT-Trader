# Dependency Management Policy

**Last Updated:** 2025-10-05
**Owner:** Platform Team
**Status:** Active

---

## Overview

This document defines the dependency management strategy for GPT-Trader, including version constraints, update procedures, and constraint rationale. All dependency decisions should be documented here to maintain institutional knowledge.

---

## Pinned Major Versions

### numpy < 2.0.0

**Constraint:** `numpy>=1.26.4,<2.0.0`

**Reason:**
- NumPy 2.0 introduces breaking changes to the array API
- pandas 2.x (our current version) has limited NumPy 2.0 support
- pydantic-core has compatibility issues with NumPy 2.0 array handling

**Impact:**
- All numerical computations use NumPy 1.x API
- pandas DataFrame operations remain stable
- No immediate performance penalty (NumPy 2.0 optimizations not critical)

**Review Schedule:** Q1 2026 when pandas 2.4+ stabilizes NumPy 2.0 support

**Update Checklist (when ready):**
- [ ] Verify pandas >=2.4.0 released with NumPy 2.0 support
- [ ] Check pydantic-core changelog for NumPy 2.0 compatibility
- [ ] Run full test suite with NumPy 2.0 in dev environment
- [ ] Benchmark performance impact (portfolio valuation, PnL calculations)
- [ ] Update constraint to `numpy>=2.0.0,<3.0.0`

---

### websockets 12.0-16.0 (currently constrained to <16.0)

**Constraint:** `websockets>=12.0,<16.0`

**Reason:**
- coinbase-advanced-py 1.8.2 requires websockets <14.0
- Our codebase doesn't directly use websockets features (only via coinbase-advanced-py)
- Historical context: Downgraded from 15.1 to 13.1 on Oct 4, 2025 for coinbase-advanced-py compatibility

**Impact:**
- None on our codebase (no direct websockets usage found)
- Coinbase streaming functionality fully operational

**Review Schedule:** When coinbase-advanced-py supports websockets 14.0+ (monitor releases)

**Update Checklist (when ready):**
- [ ] Check coinbase-advanced-py release notes for websockets >=14.0 support
- [ ] Test Coinbase WebSocket streaming with updated version
- [ ] Run integration tests: `pytest -m "brokerages and integration"`
- [ ] Validate streaming failover scenarios
- [ ] Update constraint if coinbase-advanced-py allows

---

### Python >=3.12,<3.13

**Constraint:** `python>=3.12,<3.13`

**Reason:**
- Project standardized on Python 3.12 features (structural pattern matching, generics, etc.)
- 3.13 introduces experimental JIT, free-threaded mode - compatibility uncertain
- Dev tooling (mypy, ruff, black) fully support 3.12

**Impact:**
- Development and production environments use Python 3.12.x
- CI/CD pipelines pinned to 3.12
- Type hints use 3.12 syntax

**Review Schedule:** Q2 2026 when Python 3.13 reaches maturity, tooling catches up

**Update Checklist (when ready):**
- [ ] Verify all dependencies support Python 3.13
- [ ] Test with free-threaded mode disabled (--disable-gil flag)
- [ ] Benchmark performance (JIT may improve portfolio calculations)
- [ ] Update type hints if new syntax available
- [ ] Update constraint to `python>=3.13,<3.14`

---

### coinbase-advanced-py >=1.8.2,<2.0.0

**Constraint:** `coinbase-advanced-py>=1.8.2,<2.0.0`

**Reason:**
- 1.8.2 provides stable Coinbase Advanced Trade API support
- REST and WebSocket streaming validated against production flows
- Breaking changes expected in 2.0 (API alignment)

**Impact:**
- All Coinbase brokerage operations (REST, streaming, auth)
- Rate limit tracking, error handling aligned to 1.8.x behavior
- 95% target coverage against production API flows

**Review Schedule:** When 2.0.0 released (monitor coinbase-advanced-py releases)

**Update Checklist (when ready):**
- [ ] Review 2.0.0 breaking changes and migration guide
- [ ] Update `features/brokerages/coinbase/` adapter layer
- [ ] Update rate limit tracker for API changes
- [ ] Run Coinbase API audit: `pytest -m "real_api"` (sandbox)
- [ ] Execute soak tests in sandbox environment
- [ ] Update constraint after validation

---

## Dependency Update Cadence

### Security Patches (Critical)
**Timeline:** Within 7 days of CVE publication

**Process:**
1. Monitor GitHub security advisories and Dependabot alerts
2. Review CVE severity (CVSS score)
3. If score ≥7.0 (High):
   - Update immediately in hotfix branch
   - Run fast test suite: `pytest --maxfail=10`
   - Deploy to staging, validate critical paths
   - Merge and deploy to production within 7 days
4. If score <7.0 (Medium/Low):
   - Include in next monthly dependency review

**Approval:** Platform lead + Ops approval for production deploy

---

### Minor Versions (Monthly)
**Timeline:** First week of each month

**Process:**
1. **Monday (Day 1):** Generate dependency report
   ```bash
   poetry show --outdated
   ```
2. **Tuesday (Day 2):** Review changelogs for each outdated package
   - Prioritize: coinbase-advanced-py, pandas, pytest, ruff
   - Flag breaking changes or deprecations
3. **Wednesday (Day 3):** Update in dev branch, run full test suite
   ```bash
   poetry update <package>
   pytest --maxfail=1000
   ```
4. **Thursday (Day 4):** Deploy to staging, monitor for 24 hours
5. **Friday (Day 5):** Merge to main if green, schedule production deploy

**Approval:** Platform lead review, QA sign-off

---

### Major Versions (Quarterly)
**Timeline:** Quarterly planning (Jan, Apr, Jul, Oct)

**Process:**
1. **Week 1:** Enumerate major version updates available
   - Example: pandas 2.x → 3.x, pydantic 2.x → 3.x
2. **Week 2:** Create RFC document for each major update
   - Breaking changes analysis
   - Migration effort estimation (dev days)
   - Risk assessment (test coverage, production impact)
3. **Week 3:** Team review, prioritize based on:
   - Security benefits
   - Performance improvements
   - Feature enablement
   - Community momentum (deprecation timeline)
4. **Week 4:** Schedule major updates in quarterly roadmap
   - Allocate dedicated dev time (no feature work during update week)
   - Plan rollback strategy (feature flags, blue/green deploy)

**Approval:** Architecture lead + Platform lead + Trading ops sign-off

---

## Constraint Documentation Template

When adding a new constraint, document using this template:

```markdown
### package-name version-constraint

**Constraint:** `package-name>=X.Y.Z,<A.B.C`

**Reason:**
- Why is this constraint necessary?
- What breaks without it?
- What compatibility issues exist?

**Impact:**
- Which modules/features depend on this constraint?
- What functionality is affected?
- Performance implications?

**Review Schedule:** When to revisit (date or milestone)

**Update Checklist (when ready):**
- [ ] Specific validation steps
- [ ] Tests to run
- [ ] Metrics to monitor
- [ ] Stakeholders to notify
```

---

## Pre-flight Checklist (Before Updating Dependencies)

Before running `poetry update`, ensure:

- [ ] Current branch is clean: `git status`
- [ ] All tests passing: `pytest --maxfail=1`
- [ ] Review changelog/release notes for breaking changes
- [ ] Check if any pinned constraints conflict with update
- [ ] Backup current poetry.lock: `cp poetry.lock poetry.lock.backup`
- [ ] Plan rollback: know how to restore previous state

---

## Rollback Procedure

If dependency update causes failures:

1. **Immediate rollback:**
   ```bash
   git restore poetry.lock
   poetry install --sync
   pytest --maxfail=10  # Verify restoration
   ```

2. **Document failure:**
   - Create issue with: package name, version attempted, error logs
   - Tag as `dependency-update-failed`
   - Assign to platform team

3. **Root cause analysis:**
   - Identify breaking change (changelog review)
   - Determine if our code needs adaptation
   - Estimate fix effort vs. staying on old version

4. **Decision:**
   - **Fix our code:** Schedule in sprint, update with fix
   - **Stay on old version:** Document constraint (this file), schedule quarterly review

---

## Dependency Hygiene

### .gitignore (Tool Caches)
These directories are safe to exclude from version control:
```
.mypy_cache/
.pytest_cache/
.ruff_cache/
__pycache__/
*.pyc
```

### poetry.lock
- **Always commit** poetry.lock to ensure deterministic installs
- **Never manually edit** poetry.lock (use `poetry update` or `poetry add`)
- **Review diffs** before committing to catch unexpected version jumps

### pyproject.toml Hygiene
- **Group dependencies** logically:
  - Core runtime: `[project] dependencies`
  - Development: `[tool.poetry.group.dev.dependencies]`
  - Optional: `[project.optional-dependencies]`
- **Avoid overly strict constraints** unless necessary (e.g., `>=X.Y` not `==X.Y.Z`)
- **Document constraints** in this file if pinning <major version

---

## Automation (Future)

### Dependabot / Renovate (Week 4 Investigation)

**Recommendation:** Enable automated dependency PRs

**Configuration:**
```yaml
# .github/dependabot.yml (example)
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "automated"
    reviewers:
      - "platform-team"
```

**Benefits:**
- Automatic PRs for security patches
- Changelog links in PR descriptions
- CI runs on every update PR

**Risks:**
- PR noise (can be 10+ PRs/week)
- Auto-merge risk (needs review gates)

**Week 4 Action:** Evaluate Dependabot vs. Renovate, create pilot configuration

---

## Related Documents

- [CLEANUP_CHECKLIST.md](CLEANUP_CHECKLIST.md) - Operational audit plan
- [CODEBASE_HEALTH_ASSESSMENT.md](CODEBASE_HEALTH_ASSESSMENT.md) - Dependency health metrics
- [pyproject.toml](/pyproject.toml) - Actual dependency specifications
- [REFACTORING_2025_RUNBOOK.md](../architecture/REFACTORING_2025_RUNBOOK.md) - Architecture context

---

**Questions or Exceptions:** Contact Platform Team or create issue tagged `dependency-policy`
