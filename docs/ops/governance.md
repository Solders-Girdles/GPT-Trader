# Development Governance & Standards

**Last Updated:** 2025-10-05
**Owner:** Platform Team
**Status:** Active

---

## Overview

This document defines development standards, quality gates, and operational processes for the GPT-Trader project. All contributors must follow these guidelines to maintain code quality, test coverage, and production stability.

---

## Table of Contents

1. [Pre-commit Hooks](#pre-commit-hooks)
2. [CI/CD Expectations](#cicd-expectations)
3. [Dependency Management](#dependency-management)
4. [Code Quality Standards](#code-quality-standards)
5. [Testing Requirements](#testing-requirements)
6. [Configuration Management](#configuration-management)
7. [Deployment Process](#deployment-process)

---

## Pre-commit Hooks

### Current Configuration

Pre-commit hooks are **active and enforced** on all commits. Configuration: `.pre-commit-config.yaml`

**Active Hooks (11 total):**

| Hook | Purpose | Status | Enforcement |
|------|---------|--------|-------------|
| `trailing-whitespace` | Remove trailing whitespace | ✅ Passing | Auto-fix |
| `end-of-file-fixer` | Ensure newline at EOF | ✅ Passing | Auto-fix |
| `check-yaml` | Validate YAML syntax | ✅ Passing | Fail on error |
| `check-added-large-files` | Block large files (>500KB) | ✅ Passing | Fail on error |
| `mixed-line-ending` | Enforce consistent line endings | ✅ Passing | Auto-fix |
| `black` | Code formatting (PEP 8) | ✅ Passing | Auto-fix |
| `ruff` | Fast Python linting | ✅ Passing | Auto-fix + fail |
| `pyupgrade` | Upgrade Python syntax (3.12+) | ✅ Passing | Auto-fix |
| `config-doctor` | Validate config files | ✅ Passing | Fail on error |
| `test-hygiene` | Enforce test file limits | ⚠️ 12 files | Fail on error |
| `forbid-bytecode` | Block .pyc commits | ✅ Passing | Fail on error |

### Hook Details

#### 1. Code Formatters (Auto-fix)

**black** - Python code formatter
- Enforces PEP 8 with 88-character line length
- Auto-formats on commit
- Configuration: `pyproject.toml` → `[tool.black]`
- Run manually: `poetry run black .`

**ruff** - Fast Python linter
- Replaces flake8, isort, pyupgrade (faster)
- Auto-fixes import sorting, unused imports, etc.
- Configuration: `pyproject.toml` → `[tool.ruff]`
- Run manually: `poetry run ruff check . --fix`

**pyupgrade** - Python syntax modernizer
- Upgrades syntax to Python 3.12+ features
- Converts old-style type hints, f-strings, etc.
- Run manually: `poetry run pyupgrade --py312-plus <file>`

#### 2. Validation Hooks (Fail on Error)

**config-doctor** - Configuration validator
- Validates YAML/JSON config files
- Checks for duplicate configurations
- Enforces risk profile schemas
- Run manually: `poetry run python scripts/tools/config_doctor.py --check all --strict`

**test-hygiene** - Test file quality enforcer
- Enforces 240-line limit per test file (maintainability)
- Detects `time.sleep()` usage (should use `fake_clock` fixture)
- Prevents test bloat and flaky time-dependent tests
- Run manually: `scripts/ci/check_test_hygiene.py tests/`

**Current Issues:**
- 12 pre-existing test files exceed 240 lines (splitting deferred as low-priority enhancement)
- Week 3 integration tests (3 files) allowlisted with justification (comprehensive scenario coverage)
- 2 files use `time.sleep()` without `fake_clock` fixture (low-priority enhancement)

### Usage Guidelines

#### Installing Pre-commit Hooks

```bash
# First-time setup (after cloning repo)
poetry install
poetry run pre-commit install

# Verify installation
poetry run pre-commit run --all-files
```

#### Running Hooks Manually

```bash
# Run all hooks on all files
poetry run pre-commit run --all-files

# Run specific hook
poetry run pre-commit run black --all-files
poetry run pre-commit run config-doctor --all-files

# Run on specific files
poetry run pre-commit run --files tests/unit/bot_v2/test_example.py
```

#### Bypassing Hooks (Exceptional Cases Only)

```bash
# Skip pre-commit hooks (NOT recommended)
git commit --no-verify -m "message"
```

**⚠️ Only bypass hooks when:**
- Urgent hotfix for production incident (document in commit message)
- Known test-hygiene violation justified (e.g., comprehensive integration tests)
- Hook itself is broken (report issue immediately)

**⚠️ Never bypass for:**
- "Saving time" - hooks run in <10 seconds
- Formatting disagreements - black is non-negotiable
- Test failures unrelated to changes - fix root cause first

#### Test Hygiene Allowlist

Test files exceeding 240 lines may be allowlisted if justified (comprehensive scenario coverage, integration tests). To allowlist:

1. Add justification comment to test file:
   ```python
   # test-hygiene: allowlist - Comprehensive integration scenarios (8 critical paths)
   ```

2. Update `scripts/ci/check_test_hygiene.py` allowlist:
   ```python
   ALLOWLIST = {
       "tests/integration/brokerages/test_coinbase_streaming_failover.py": "6 WebSocket failover scenarios",
       # ... add your file
   }
   ```

3. Document in commit message why 240-line limit doesn't apply

---

## CI/CD Expectations

### GitHub Actions Workflows

**Primary Workflows:**

| Workflow | Trigger | Purpose | Coverage |
|----------|---------|---------|----------|
| `ci.yml` | PR, Push to main/develop | Main CI pipeline | Linting, testing, coverage |
| `bot-v2.yml` | PR, Push | Bot-specific tests | Integration, scenario tests |
| `config-validation.yml` | Config file changes | Validate configurations | Risk profiles, env files |
| `coverage-check.yml` | PR | Enforce coverage thresholds | ≥60% required |
| `nightly_validation.yml` | Nightly cron | Full regression suite | Slow tests, performance |
| `security-audit.yml` | Daily | Dependency vulnerability scan | CVE monitoring |

### CI Requirements (Must Pass)

All PRs must pass these checks before merge:

#### 1. Linting (ruff)
```bash
poetry run ruff check .
```
- Zero linting errors allowed
- Auto-fixes applied via pre-commit

#### 2. Formatting (black)
```bash
poetry run black --check .
```
- All code must be black-formatted
- Pre-commit auto-formats, CI validates

#### 3. Type Checking (mypy)
```bash
poetry run mypy src --ignore-missing-imports
```
- Type hints enforced in `src/bot_v2/`
- Gradual typing: ignore missing imports for now
- Future: Strict mode once coverage >80%

#### 4. Unit Tests (pytest)
```bash
poetry run pytest -m "not slow and not performance" --cov=src/bot_v2 --cov-fail-under=60
```
- **Coverage requirement: ≥60%**
- Slow tests excluded in PR CI (run nightly)
- Fast feedback loop (<2 min)

#### 5. Selective Test Runner (PRs only)
- Dependency analysis determines affected tests
- Runs only impacted test modules
- Max selective ratio: 70% (fallback to full suite if >70% affected)
- Auto-runs full suite for interface changes

#### 6. Coinbase Core Checks
```bash
poetry run pytest tests/unit/bot_v2/features/brokerages/coinbase -q
```
- Critical path: Coinbase broker integration
- Must pass on every commit
- Real API tests run nightly (not in PR CI)

### CI Performance Benchmarks

**Target CI Duration:**
- PR CI: <5 minutes (selective tests)
- Main CI: <10 minutes (full unit suite, no slow tests)
- Nightly: <30 minutes (full regression + slow tests)

**Optimization Strategies:**
- Selective test runner based on dependency graph
- Cached Poetry virtualenvs (GitHub Actions cache)
- Parallel test execution (pytest-xdist)
- Skip slow/performance tests in PR CI

### CI Failure Response

**If CI fails:**

1. **Linting/Formatting:** Run pre-commit locally, push fix
   ```bash
   poetry run pre-commit run --all-files
   git add -u && git commit -m "style: Fix linting/formatting"
   ```

2. **Type Errors:** Fix type hints, re-run mypy
   ```bash
   poetry run mypy src --ignore-missing-imports
   ```

3. **Test Failures:**
   - Reproduce locally: `poetry run pytest <failing_test>`
   - Fix root cause (don't skip tests)
   - Add regression test if bug found

4. **Coverage Drop:** Add tests for new code
   ```bash
   poetry run pytest --cov=src/bot_v2 --cov-report=html
   open htmlcov/index.html  # Identify uncovered lines
   ```

### Merge Requirements

**All of these must be ✅ before merge:**
- [ ] All CI checks passing
- [ ] Code review approved (1+ reviewer)
- [ ] Pre-commit hooks passing locally
- [ ] Coverage ≥60% maintained
- [ ] No merge conflicts with base branch
- [ ] PR description explains changes (why, what, how)

---

## Dependency Management

### Policy Document

**Primary Reference:** [dependency_policy.md](dependency_policy.md)

Comprehensive dependency management strategy is documented in `docs/ops/dependency_policy.md`. Key highlights:

### Pinned Constraints (Active)

| Package | Constraint | Reason | Review Date |
|---------|-----------|--------|-------------|
| `numpy` | `<2.0.0` | Pandas/pydantic compatibility | Q1 2026 |
| `websockets` | `<16.0` | coinbase-advanced-py requirement | When coinbase-advanced-py supports 14.0+ |
| `python` | `>=3.12,<3.13` | Project standard, tooling support | Q2 2026 |
| `coinbase-advanced-py` | `<2.0.0` | API stability | When 2.0 released |

### Update Cadence

- **Security Patches:** Within 7 days (CVSS ≥7.0)
- **Minor Versions:** Monthly (first week)
- **Major Versions:** Quarterly planning

### Quick Commands

```bash
# Check for outdated dependencies
poetry show --outdated

# Update specific package (minor version)
poetry update <package>

# Update all (within constraints)
poetry update

# Lock file only (no install)
poetry lock --no-update
```

### Pre-flight Checklist

Before updating dependencies:
- [ ] Branch is clean: `git status`
- [ ] Tests passing: `pytest --maxfail=1`
- [ ] Review changelog for breaking changes
- [ ] Backup lock file: `cp poetry.lock poetry.lock.backup`

### Rollback Procedure

If update fails:
```bash
git restore poetry.lock
poetry install --sync
pytest --maxfail=10  # Verify restoration
```

---

## Code Quality Standards

### File Organization

**Python Modules:**
- Max 500 lines per module (enforce via reviews)
- Extract to subpackages if exceeded
- One class per file for large classes (>200 lines)

**Test Files:**
- Max 240 lines per test file (enforced by `test-hygiene` hook)
- Use fixtures for setup/teardown
- Allowlist if comprehensive integration tests

**Configuration Files:**
- YAML preferred over JSON (human-readable)
- Validate with `config-doctor` hook
- No duplicate configurations

### Code Style

**Formatting:**
- Black (88-character line length)
- Auto-formatted via pre-commit
- No manual formatting debates

**Linting:**
- Ruff (replaces flake8, isort, pyupgrade)
- Zero linting errors in CI
- Auto-fixes applied when possible

**Type Hints:**
- Required for new code in `src/bot_v2/`
- Gradual typing: existing code may omit hints
- Use `mypy` to validate

**Naming Conventions:**
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Documentation Standards

**Docstrings:**
- Required for public functions/classes
- Google-style format:
  ```python
  def execute_order(symbol: str, side: str) -> OrderResult:
      """Execute a trading order.

      Args:
          symbol: Trading pair (e.g., "BTC-USD")
          side: Order side ("buy" or "sell")

      Returns:
          OrderResult with execution details

      Raises:
          RiskGuardError: If pre-trade checks fail
      """
  ```

**Inline Comments:**
- Explain "why", not "what" (code is self-documenting)
- Use sparingly (prefer clear variable names)
- Update when code changes

**Architecture Docs:**
- Update `docs/ARCHITECTURE.md` for structural changes
- Document feature flags in `REFACTORING_2025_RUNBOOK.md`
- Update `docs/ops/` for operational changes

---

## Testing Requirements

### Coverage Thresholds

| Scope | Minimum Coverage | Current |
|-------|-----------------|---------|
| Overall | 60% | 87.52% ✅ |
| New Code | 80% | Enforced in reviews |
| Critical Paths | 90% | Brokerages, risk management |

### Test Categories (Markers)

| Marker | Purpose | Run Frequency | Typical Duration |
|--------|---------|---------------|------------------|
| `unit` | Fast unit tests | Every commit | <1 min |
| `integration` | Integration tests | PR CI (selective) | 2-5 min |
| `slow` | Slow tests (I/O, network) | Nightly | 10-30 min |
| `performance` | Performance benchmarks | Weekly | 5-10 min |
| `real_api` | Live API tests (sandbox) | Manual | 5-15 min |
| `soak` | Extended stability tests | Manual | Hours/days |

### Test Structure

**Unit Tests:**
```python
# tests/unit/bot_v2/features/live_trade/test_risk.py
def test_position_size_calculation():
    """Test Kelly Criterion position sizing."""
    sizer = PositionSizer(...)
    result = sizer.calculate(...)
    assert result.size == Decimal("0.1")
```

**Integration Tests:**
```python
# tests/integration/brokerages/test_coinbase.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_order_lifecycle():
    """Test full order lifecycle with Coinbase adapter."""
    # Setup, execute, verify
```

**Fixtures:**
- Use fixtures for common test setup
- Share fixtures via `conftest.py`
- Mock external dependencies (broker API, market data)

### Running Tests Locally

```bash
# Fast unit tests only
poetry run pytest -m "not slow and not performance"

# Specific test file
poetry run pytest tests/unit/bot_v2/features/live_trade/test_risk.py

# Integration tests (selective)
poetry run pytest -m integration

# Coverage report
poetry run pytest --cov=src/bot_v2 --cov-report=html
open htmlcov/index.html

# Verbose mode (see test names)
poetry run pytest -v

# Stop on first failure
poetry run pytest -x
```

---

## Configuration Management

### Configuration Files

**Risk Profiles:**
- Location: `config/risk/`
- Format: YAML (preferred) or JSON
- Validation: `config-doctor` hook
- Schema: Defined in `scripts/tools/config_doctor.py`

**Trading Profiles:**
- Location: `config/profiles/`
- Format: YAML
- Examples: `dev_entry.yaml`, `canary.yaml`, `spot.yaml`
- Validation: YAML syntax check

**Environment Variables:**
- Location: `.env` (git-ignored), `config/environments/.env.*`
- Never commit secrets
- Template: `.env.example` (commit this)
- Load via `python-dotenv`

### Config Validation

```bash
# Validate all configs
poetry run python scripts/tools/config_doctor.py --check all --strict

# Validate specific config
poetry run python scripts/tools/config_doctor.py --check risk --strict

# Run via pre-commit
poetry run pre-commit run config-doctor --all-files
```

### Config Hygiene

**Best Practices:**
- ✅ Use YAML for human-edited configs
- ✅ JSON for machine-generated configs only
- ✅ No duplicate configs (one source of truth)
- ✅ Validate schema before commit
- ❌ Never commit `.env` with secrets
- ❌ No hardcoded credentials in configs

---

## Deployment Process

### Environments

| Environment | Branch | Purpose | Approval Required |
|-------------|--------|---------|-------------------|
| `dev` | feature branches | Local development | None |
| `staging` | `develop` | Integration testing | Platform lead |
| `canary` | `main` (subset) | Limited production rollout | Ops + Platform |
| `production` | `main` | Live trading | Ops + Platform + Trading |

### Deployment Gates

**Staging Deployment:**
- [ ] All CI checks passing
- [ ] Code review approved
- [ ] Integration tests passing
- [ ] Manual smoke test in dev environment

**Production Deployment:**
- [ ] Staging validated for 24+ hours
- [ ] No open P0/P1 incidents
- [ ] Canary deployment successful (if available)
- [ ] Rollback plan documented
- [ ] On-call engineer identified

### Deployment Commands

```bash
# Deploy to staging
git checkout develop
git pull origin develop
# CI automatically deploys to staging on push

# Deploy to production (manual)
git checkout main
git merge develop --ff-only
git push origin main
# Notify ops team, monitor deployment
```

### Rollback Procedure

If deployment causes issues:

1. **Immediate rollback:**
   ```bash
   git revert <bad-commit-sha>
   git push origin main
   ```

2. **Incident response:**
   - Create P0 incident ticket
   - Notify trading ops and platform team
   - Document root cause in post-mortem

3. **Prevention:**
   - Add regression test
   - Update deployment checklist
   - Review approval process

---

## Exception Process

### Requesting Governance Exception

If you need to bypass governance (e.g., skip test-hygiene, deploy without staging):

1. **Create exception request:**
   - Document: Why is exception needed?
   - Impact: What risks are introduced?
   - Mitigation: How will you minimize risk?
   - Duration: Temporary or permanent exception?

2. **Approval required:**
   - Platform lead approval (required)
   - Architecture lead (for architectural exceptions)
   - Ops lead (for deployment exceptions)

3. **Document exception:**
   - Add to code comment (for test-hygiene allowlist)
   - Add to commit message (for --no-verify commits)
   - Create follow-up issue to resolve (if temporary)

---

## Related Documents

**Operations & Infrastructure:**
- [operations_runbook.md](operations_runbook.md) - Operational procedures and incident response
- [dependency_policy.md](dependency_policy.md) - Dependency management strategy
- [CLEANUP_CHECKLIST.md](CLEANUP_CHECKLIST.md) - Operational audit plan

**Architecture & Monitoring:**
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [REFACTORING_2025_RUNBOOK.md](../architecture/REFACTORING_2025_RUNBOOK.md) - Refactoring guide
- [MONITORING_PLAYBOOK.md](../MONITORING_PLAYBOOK.md) - Monitoring and alerting playbook
- [monitoring.md](../guides/monitoring.md) - Monitoring setup and configuration

---

## Contact & Questions

**Questions or clarifications:** Create issue tagged `governance` or contact Platform Team

**Governance updates:** Submit PR to this document with justification

**Last Review:** 2025-10-05 (Week 4, Operational Audit Q4 2025)
