# Tooling Baseline - Phase 0

**Generated**: 2025-10-05
**Purpose**: Establish current status of linting, typing, and testing infrastructure

---

## Executive Summary

**Overall Status**: ✅ **GREEN** - All critical systems functional

- ✅ **Poetry**: Configuration valid
- ⚠️ **Ruff**: 6 linting errors (4 auto-fixable)
- ⚠️ **Mypy**: 393 type errors across 80 files
- ✅ **Pytest**: 5189/5209 tests passing (99.6% pass rate)

**Gate Readiness**: Tests are ready to gate. Linting/typing need work before enforcement.

---

## 1. Poetry Check

### Status: ✅ PASS

```bash
$ poetry check
All set!
```

**Assessment**:
- `pyproject.toml` is valid
- Dependencies properly declared
- No configuration issues

**Action**: None required

---

## 2. Ruff (Linting)

### Status: ⚠️ MINOR ISSUES (6 errors, 4 auto-fixable)

### Summary
```
4  I001  [*] unsorted-imports (auto-fixable)
2  F821  [ ] undefined-name
```

### Detailed Errors

#### Import Sorting (Auto-fixable)
```
src/bot_v2/orchestration/guardrails.py:3:1
src/bot_v2/orchestration/perps_bot.py:25:5
src/bot_v2/orchestration/perps_bot_builder.py:9:1
src/bot_v2/orchestration/streaming_service.py:17:5
```

**Fix**: Run `poetry run ruff check --fix src/bot_v2`

#### Undefined Names (Requires manual fix)
```
src/bot_v2/features/brokerages/coinbase/websocket_handler.py:36:46: F821 Undefined name `Any`
src/bot_v2/features/brokerages/coinbase/websocket_handler.py:135:44: F821 Undefined name `Any`
```

**Root cause**: Missing `from typing import Any` import

**Fix**: Add import statement:
```python
from typing import Any
```

### Configuration Note
```
warning: The following rules have been removed and ignoring them has no effect:
    - ANN101
    - ANN102
```

**Action**: Remove `ANN101` and `ANN102` from ruff configuration in `pyproject.toml`

### Gating Recommendation
**Ready to gate**: NO (but close)
- Auto-fix import sorting: `ruff check --fix`
- Manually fix 2 undefined name errors
- Clean up deprecated rule warnings
- **Then**: Enable ruff as CI gate

---

## 3. Mypy (Type Checking)

### Status: ⚠️ SIGNIFICANT TYPING GAPS (393 errors in 80 files)

### High-Level Statistics
- **Files checked**: 315
- **Files with errors**: 80 (25% of codebase)
- **Total errors**: 393

### Error Categories (Sample)

#### 1. CLI Argument Handling (`cli/argument_groups.py`, `cli/commands/order_args.py`)
**Issues**:
- Type incompatibilities in argument parsing
- Unsafe `Any` usage
- Missing null checks

**Example**:
```python
# cli/argument_groups.py:37
error: Incompatible types in assignment (expression has type "type", target has type "str")

# cli/commands/order_args.py:49
error: Returning Any from function declared to return "str"
error: Item "None" of "Any | None" has no attribute "strip"
```

**Priority**: MEDIUM (affects CLI but doesn't block functionality)

---

#### 2. Validation Logic (`validation/calculation_validator.py`)
**Issues**:
- Comparison operators with potentially None values
- Missing null guards

**Example**:
```python
# validation/calculation_validator.py:51
error: Unsupported operand types for > ("float" and "None")
error: Unsupported operand types for < ("float" and "None")
```

**Priority**: HIGH (validation logic should be type-safe)

---

#### 3. Coinbase Integration (`features/brokerages/coinbase/specs.py`)
**Issues**:
- Complex type inference failures
- List/bool type confusion
- Missing type annotations

**Example**:
```python
# specs.py:187
error: Need type annotation for "result"

# specs.py:194-195
error: Incompatible types in assignment (expression has type "float", target has type "list[Any] | bool | None")
error: Item "bool" of "list[Any] | bool | None" has no attribute "append"
```

**Priority**: HIGH (critical path for brokerage integration)

---

#### 4. State Management (`state/backup/services/transport.py`)
**Issues**:
- Unsafe `Any` usage in AWS S3 operations
- Missing null checks

**Example**:
```python
# state/backup/services/transport.py:160, 176
error: Item "None" of "Any | None" has no attribute "put_object"
```

**Priority**: MEDIUM (backup infrastructure)

---

#### 5. Live Trade Strategy Signals (`features/live_trade/strategies/strategy_signals.py`)
**Issues**:
- Incorrect type annotation (`any` vs `Any`)
- Missing imports

**Example**:
```python
# strategy_signals.py:52
error: Function "builtins.any" is not valid as a type
note: Perhaps you meant "typing.Any" instead of "any"?

# strategy_signals.py:103
error: any? has no attribute "calculate_rsi"
```

**Priority**: HIGH (live trading critical path)

---

### Top Priority Files for Type Safety Fixes

1. **validation/calculation_validator.py** - 26 errors (validation logic must be type-safe)
2. **features/brokerages/coinbase/specs.py** - 20 errors (critical for brokerage)
3. **cli/commands/order_args.py** - 7 errors (user-facing CLI)
4. **features/live_trade/strategies/strategy_signals.py** - 3 errors (live trading)
5. **cli/argument_groups.py** - 4 errors (CLI foundation)

### Gating Recommendation
**Ready to gate**: NO
- **Effort required**: ~2-4 weeks to resolve 393 errors
- **Recommended approach**:
  1. Fix top 5 critical files (~60 errors) - Week 1
  2. Fix remaining high-priority modules - Week 2
  3. Address lower-priority issues - Weeks 3-4
  4. Enable mypy as warning-only in CI - Week 2
  5. Promote to blocking gate - Week 4

**Alternative**: Gate with `--strict` on new code only, allow existing issues

---

## 4. Pytest (Testing)

### Status: ✅ EXCELLENT (99.6% pass rate)

### Summary
```
5189 passed
  20 skipped
  99 deselected (by marks/filters)
Total runtime: 44.21s
```

### Test Collection
- **Total tests available**: 5308
- **Tests run**: 5209 (99 deselected)
- **Pass rate**: 99.6%

### Skipped Tests Breakdown
Skipped tests are typically:
- Integration tests requiring live broker connections
- Tests requiring optional dependencies
- Platform-specific tests

### Performance (Slowest 10)
```
5.11s  test_start_stop_collection (metrics_collector)
1.54s  test_walk_forward_different_window_sizes (optimize)
1.11s  test_clear_expired_multiple (cache)
1.11s  test_clear_expired (cache)
1.11s  test_cache_expiry (data_providers)
1.11s  test_stats_expired_count (cache)
1.11s  test_get_expired_key (cache)
1.10s  test_ttl_expired (cache)
0.74s  test_initialization_default_params (paper_trade)
0.56s  test_error_history_tracking (error_handler)
```

**Note**: Cache tests involve `sleep()` calls for TTL validation (expected slow tests)

### Coverage (from existing .coverage file)
Coverage data exists (`coverage.json`, `htmlcov/`). Current baseline appears to be tracked.

### Gating Recommendation
**Ready to gate**: ✅ YES

**Recommended gate configuration**:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--maxfail=5",  # Stop after 5 failures for fast feedback
    "--strict-markers",
]
# Gate: Fail CI if any test fails
```

**Coverage gate** (optional enhancement):
```bash
pytest --cov=src/bot_v2 --cov-report=term --cov-fail-under=80
```

---

## Pre-commit Configuration

### Current Status
File exists: `.pre-commit-config.yaml`

### Recommended Hooks (Phase 0 - Low Friction)
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.13.3
    hooks:
      - id: ruff
        args: [--fix]  # Auto-fix import sorting
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

**Do NOT enable yet**:
- `mypy` hook (393 errors would block all commits)
- Pytest hook (44s runtime too slow for commit hook)

**Phase 1 addition** (once mypy errors fixed):
```yaml
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.18.2
    hooks:
      - id: mypy
        args: [--strict]
```

---

## Recommended pyproject.toml Tool Configuration

### Current Issues to Address
1. Remove deprecated ruff rules (`ANN101`, `ANN102`)
2. Ensure tool sections are complete

### Minimal Required Sections

#### [tool.ruff]
```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
]
ignore = [
    # Remove deprecated rules
    # "ANN101",  # <- DELETE THIS
    # "ANN102",  # <- DELETE THIS
]

[tool.ruff.lint.isort]
known-first-party = ["bot_v2"]
```

#### [tool.mypy]
```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true

# Gradually enable strict mode per module
[[tool.mypy.overrides]]
module = [
    "bot_v2.features.brokerages.coinbase.*",
    "bot_v2.validation.*",
]
disallow_untyped_defs = false  # Temporarily allow for high-error modules
```

#### [tool.pytest.ini_options]
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "live: marks tests requiring live broker connection",
]
```

---

## Phase 0 Action Items

### Immediate (This Week)
- [x] Document current tooling baseline (this file)
- [ ] Fix 6 ruff errors (1 hour effort):
  - Run `ruff check --fix` for import sorting (4 errors)
  - Add `from typing import Any` to websocket_handler.py (2 errors)
- [ ] Remove deprecated ruff rules from pyproject.toml:
  - Delete `ANN101`, `ANN102` from ignore list
- [ ] Update `.pre-commit-config.yaml` with ruff hooks (don't enable mypy yet)
- [ ] Run pre-commit hooks manually to test: `pre-commit run --all-files`

### Phase 1 Prep (Next 2 Weeks)
- [ ] Create mypy error resolution plan:
  - Prioritize top 5 files (60 errors)
  - Create GitHub issues/tickets for tracking
  - Assign to team members
- [ ] Enable pytest in CI as blocking gate
- [ ] Enable ruff in CI as blocking gate (after 6 errors fixed)
- [ ] Set up coverage tracking baseline

### Phase 1 Goals (Weeks 3-6)
- [ ] Reduce mypy errors to <100 (75% reduction)
- [ ] Enable mypy in CI as warning-only
- [ ] Achieve 80%+ test coverage on new code
- [ ] Enable mypy pre-commit hook for new files only

---

## CI/CD Gate Configuration (Future)

### Phase 1 Gates (After Fixes)
```yaml
# .github/workflows/ci.yml (example)
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: poetry install
      - run: poetry run ruff check src/bot_v2  # GATE: Must pass

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: poetry install
      - run: poetry run mypy src/bot_v2 || true  # WARNING ONLY (Phase 1)

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: poetry install
      - run: poetry run pytest tests/ --maxfail=1  # GATE: Must pass
```

### Phase 2 Gates (Full Enforcement)
- Promote mypy to blocking gate
- Add coverage requirement (80%+ on new code)
- Add security scanning (bandit, safety)

---

## Appendix: Raw Command Outputs

### Poetry Check
```bash
$ poetry check
All set!
```

### Ruff Statistics
```bash
$ poetry run ruff check src/bot_v2 --statistics
4  I001  [*] unsorted-imports
2  F821  [ ] undefined-name
Found 6 errors.
[*] 4 fixable with the `--fix` option.
```

### Mypy Summary
```bash
$ poetry run mypy src/bot_v2
...
Found 393 errors in 80 files (checked 315 source files)
```

### Pytest Summary
```bash
$ poetry run pytest tests/ -x --tb=no -q
...
=============== 5189 passed, 20 skipped, 99 deselected in 44.21s ===============
```

---

**Next Steps**: Review pyproject.toml tool configurations (Task 4)
