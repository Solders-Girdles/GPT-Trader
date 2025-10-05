# Codebase Cleanup - Quick Action Checklist

**Based on**: [Codebase Health Assessment](docs/CODEBASE_HEALTH_ASSESSMENT.md)
**Priority**: Critical items first, then important, then nice-to-have

---

## 🔴 Critical (Do Now - 1 hour total)

### ✅ 1. Fix Failing Test (1 min)

**Issue**: CLI argument count test failing

**File**: `tests/unit/bot_v2/cli/test_argument_groups.py`

**Fix**:
```bash
# Option 1: Quick fix
sed -i.bak 's/assert len(BOT_CONFIG_ARGS) == 11/assert len(BOT_CONFIG_ARGS) == 12/' tests/unit/bot_v2/cli/test_argument_groups.py

# Option 2: Manual edit
# Line 91: Change `assert len(BOT_CONFIG_ARGS) == 11` to `assert len(BOT_CONFIG_ARGS) == 12`
```

**Verify**:
```bash
poetry run pytest tests/unit/bot_v2/cli/test_argument_groups.py::TestArgumentGroups::test_bot_config_args_count -v
```

---

### ✅ 2. Update Security Dependencies (30 min)

**Critical packages**:
```bash
# Update security-critical packages
poetry update certifi cryptography coinbase-advanced-py

# Verify no breaking changes
poetry run pytest tests/ -x --tb=short
```

**If tests fail**:
1. Check `poetry.lock` diff for version changes
2. Review changelog for breaking changes
3. Revert with `git checkout poetry.lock` if needed

---

### ✅ 3. Fix Poetry Configuration (10 min)

**Issue**: Duplicate fields between `[project]` and `[tool.poetry]`

**File**: `pyproject.toml`

**Fix**: Keep only `[project]` section (PEP 621 standard), remove from `[tool.poetry]`:

```bash
# Backup first
cp pyproject.toml pyproject.toml.backup

# Edit pyproject.toml manually to remove duplicates from [tool.poetry]:
# - name
# - version
# - description
# - authors
```

**Verify**:
```bash
poetry check
# Should show no warnings about duplicate fields
```

---

### ✅ 4. Stage Recent Work (15 min)

**Untracked files from Phase 3.2/3.3**:

```bash
# Stage soak test documentation
git add docs/testing/prometheus_queries.md
git add docs/testing/sandbox_deployment_checklist.md
git add docs/testing/sandbox_soak_test_plan.md
git add SOAK_TEST_QUICKSTART.md

# Stage monitoring configs
git add monitoring/alertmanager/alertmanager.yml
git add monitoring/grafana/dashboards/
git add monitoring/grafana/provisioning/

# Stage deployment scripts
git add scripts/README.md
git add scripts/*.sh

# Stage new source files
git add src/bot_v2/monitoring/metrics_server.py
git add src/bot_v2/orchestration/guardrails.py

# Stage new test files
git add tests/unit/bot_v2/monitoring/test_metrics_server.py
git add tests/unit/bot_v2/orchestration/test_broker_health.py
git add tests/unit/bot_v2/orchestration/test_broker_selection.py
git add tests/unit/bot_v2/orchestration/test_guardrails.py

# Stage environment template
git add .env.sandbox.example

# Review what will be committed
git status

# Commit
git commit -m "feat(phase-3): Add Phase 3.2/3.3 guardrails, streaming validation, and soak test infrastructure

- Add guardrails framework with order caps, daily loss limit, and circuit breaker
- Implement streaming metrics and REST fallback for WebSocket resilience
- Add comprehensive soak test documentation and deployment automation
- Update Prometheus alerts for streaming and guardrail monitoring
- Add Grafana dashboards for bot health and trading activity
- Create deployment scripts for sandbox soak testing
"
```

---

## 🟡 Important (Do This Week - 3-4 hours)

### ✅ 5. Update Remaining Dependencies (1 hour)

**Safe patch updates**:
```bash
# Update patch versions (low risk)
poetry update coverage hypothesis identify pydantic propcache protobuf pycparser

# Update minor versions (review changelogs)
poetry update click pandas beautifulsoup4

# Test thoroughly
poetry run pytest tests/ -v
```

**Hold for now**:
- ❌ `numpy` (1.26 → 2.3 is major version, breaking changes)
- ❌ `cffi` (1.17 → 2.0 is major version)

**Research required**: Check if upgrading numpy 2.x breaks pandas/other deps

---

### ✅ 6. Expand Integration Test Coverage (2-3 hours)

**Current**: 37 integration tests
**Target**: 60+ integration tests

**Priority test areas**:

1. **Broker Integration** (`tests/integration/brokerages/`):
   ```python
   # test_coinbase_integration.py
   - test_place_order_end_to_end()
   - test_websocket_streaming_lifecycle()
   - test_rest_fallback_on_disconnect()
   - test_order_status_polling()
   ```

2. **Guardrails Integration** (`tests/integration/orchestration/`):
   ```python
   # test_guardrails_integration.py
   - test_daily_loss_limit_triggers_reduce_only()
   - test_circuit_breaker_halts_trading()
   - test_order_cap_blocks_large_orders()
   - test_guard_auto_reset_after_cooldown()
   ```

3. **Streaming Integration** (`tests/integration/streaming/`):
   ```python
   # test_streaming_integration.py
   - test_websocket_to_rest_fallback()
   - test_streaming_metrics_collection()
   - test_reconnect_behavior()
   ```

**Create test files**:
```bash
mkdir -p tests/integration/brokerages
mkdir -p tests/integration/streaming
touch tests/integration/brokerages/test_coinbase_integration.py
touch tests/integration/orchestration/test_guardrails_integration.py
touch tests/integration/streaming/test_streaming_integration.py
```

---

### ✅ 7. Clean Up Repository (30 min)

**Remove Python cache**:
```bash
# Add to .gitignore if not present
grep -q "__pycache__" .gitignore || echo "__pycache__/" >> .gitignore
grep -q "*.pyc" .gitignore || echo "*.pyc" >> .gitignore

# Clean existing cache (1,035 files)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Commit .gitignore update
git add .gitignore
git commit -m "chore: Ignore Python cache files"
```

**Clean coverage artifacts**:
```bash
rm -rf htmlcov/
rm -f .coverage
```

---

## 🟢 Nice-to-Have (Do Next Month)

### ✅ 8. Refactor Large Files

**Target files** (>500 LOC):

1. **`state_manager.py`** (690 LOC)
   - Split into: `state_operations.py`, `state_manager.py`

2. **`logger.py`** (638 LOC)
   - Split into: `formatters.py`, `handlers.py`, `setup.py`

3. **`metrics_server.py`** (620 LOC)
   - Split into: `collectors.py`, `server.py`, `health.py`

**Effort**: 1-2 days per file

---

### ✅ 9. Set Up Pre-commit Hooks

**File**: `.pre-commit-config.yaml` (already exists)

```bash
# Install pre-commit
poetry add --group dev pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

**Add hooks** (edit `.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.0
    hooks:
      - id: mypy
```

---

### ✅ 10. Generate API Documentation

**Tool**: Sphinx with autodoc

```bash
# Install Sphinx
poetry add --group dev sphinx sphinx-autodoc-typehints furo

# Initialize
sphinx-quickstart docs/api

# Configure autodoc in docs/api/conf.py
# Add bot_v2 to sys.path
# Enable autodoc extension

# Generate
sphinx-build -b html docs/api docs/api/_build

# Serve locally
cd docs/api/_build && python -m http.server
```

**Effort**: 4-6 hours initial setup

---

## Verification Commands

### Check Test Status
```bash
# Run all tests
poetry run pytest tests/ -v

# Check test count
poetry run pytest tests/ --co -q | wc -l
# Should show: ~6486 tests (or more after adding integration tests)

# Run only integration tests
poetry run pytest tests/integration/ -v
```

### Check Dependency Status
```bash
# Show outdated packages
poetry show --outdated

# Verify Poetry config
poetry check
# Should show: All set! (no warnings)
```

### Check Code Quality
```bash
# Count large files (>500 LOC)
find src -name "*.py" -exec wc -l {} \; | awk '$1 > 500' | sort -rn

# Count TODO/FIXME
grep -r "TODO\|FIXME\|XXX\|HACK" src --include="*.py" | wc -l
# Should show: 0 or 1

# Count type ignores
grep -r "type: ignore" src --include="*.py" | wc -l
# Should show: ~41
```

### Check Git Status
```bash
# Check for untracked files
git ls-files --others --exclude-standard

# Check modified files
git status --short

# Clean status expected after staging work
```

---

## Quick Start (30 minutes)

**Fastest path to clean codebase**:

```bash
# 1. Fix test (1 min)
sed -i.bak 's/assert len(BOT_CONFIG_ARGS) == 11/assert len(BOT_CONFIG_ARGS) == 12/' tests/unit/bot_v2/cli/test_argument_groups.py

# 2. Update security deps (5 min)
poetry update certifi cryptography coinbase-advanced-py

# 3. Fix Poetry config (5 min)
# Edit pyproject.toml: remove name/version/description/authors from [tool.poetry]

# 4. Verify Poetry (1 min)
poetry check

# 5. Run tests (2 min)
poetry run pytest tests/unit -x --tb=short

# 6. Stage work (5 min)
git add docs/testing/*.md monitoring/ scripts/ src/bot_v2/monitoring/ src/bot_v2/orchestration/guardrails.py tests/unit/bot_v2/monitoring/ tests/unit/bot_v2/orchestration/test_guardrails.py SOAK_TEST_QUICKSTART.md .env.sandbox.example

# 7. Commit (1 min)
git commit -m "feat(phase-3): Add Phase 3.2/3.3 infrastructure and fix test/config issues"

# 8. Clean cache (2 min)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 9. Update .gitignore (1 min)
echo "__pycache__/" >> .gitignore
git add .gitignore && git commit -m "chore: Ignore Python cache"

# 10. Final verification (2 min)
poetry run pytest tests/ -x
git status
```

---

## Success Criteria

**After Critical Fixes**:
- ✅ All tests passing (168/168 or more)
- ✅ No Poetry warnings (`poetry check` clean)
- ✅ Security deps updated
- ✅ All work committed (clean `git status`)

**After Important Items**:
- ✅ 60+ integration tests
- ✅ All safe deps updated
- ✅ No `__pycache__` in repo

**After Nice-to-Have**:
- ✅ No files >500 LOC
- ✅ Pre-commit hooks active
- ✅ API docs generated

---

## Need Help?

**Commands not working?**
- Check Poetry version: `poetry --version` (should be 1.8+)
- Check Python version: `python --version` (should be 3.12+)
- Reinstall dependencies: `poetry install --sync`

**Tests failing after updates?**
- Check diff: `git diff poetry.lock`
- Review package changelogs
- Revert if needed: `git checkout poetry.lock && poetry install`

**Questions about refactoring?**
- See: [DEVELOPMENT_GUIDELINES.md](docs/DEVELOPMENT_GUIDELINES.md)
- See: [Codebase Health Assessment](docs/CODEBASE_HEALTH_ASSESSMENT.md)
