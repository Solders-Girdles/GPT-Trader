# Tool Configuration Review - Phase 0

**Generated**: 2025-10-05
**Purpose**: Document and assess current tool configurations in `pyproject.toml` and `pytest.ini`

---

## Executive Summary

**Overall Status**: ✅ **WELL-CONFIGURED** with minor cleanup needed

### Highlights
- ✅ Comprehensive dev tooling (ruff, mypy, pytest, coverage, black)
- ✅ Pytest has extensive marker system and sensible defaults
- ✅ Coverage tracking at 85% threshold (current: 87.52%)
- ⚠️ Ruff ignoring deprecated rules (`ANN101`, `ANN102`)
- ⚠️ Ruff excluding `tests/` and `scripts/` from linting (should lint tests)

---

## 1. Project Metadata

### Package Information
```toml
[project]
name = "gpt-trader"
version = "0.1.0"
description = "Equities-first trading bot scaffold..."
requires-python = ">=3.12,<3.13"
```

**Assessment**: ✅ Clean, well-defined

### Entry Points
```toml
[project.scripts]
gpt-trader = "bot_v2.cli:main"
perps-bot = "bot_v2.cli:main"
```

**Assessment**: ✅ Two CLI aliases pointing to same entry point (good for brand evolution)

---

## 2. Dependencies

### Core Production Dependencies (12 packages)
```toml
dependencies = [
    "coinbase-advanced-py (>=1.8.2,<2.0.0)",  # Brokerage integration
    "pandas>=2.2.2,<3.0.0",                   # Data manipulation
    "numpy>=1.26.4,<2.0.0",                   # Numerical computing
    "pydantic>=2.7.4,<3.0.0",                 # Data validation
    "python-dotenv>=1.0.1,<2.0.0",            # Config management
    "requests>=2.32.3,<3.0.0",                # HTTP client
    "websockets>=12.0,<16.0",                 # WebSocket client
    "pyyaml>=6.0.1,<7.0.0",                   # YAML parsing
    "psutil>=7.0.0,<8.0.0",                   # System monitoring
    "aiohttp>=3.12.15,<4.0.0",                # Async HTTP
    "redis[hiredis]>=6.0.0,<7.0.0",           # Caching/state
    "cryptography>=46.0.0,<47.0.0",           # Security
    "pyotp>=2.9.0,<3.0.0",                    # 2FA/OTP
    "prometheus-client (>=0.23.1,<0.24.0)",   # Metrics
]
```

**Assessment**: ✅ Clean dependency list, all pinned appropriately

**Observations**:
- Wide range on `websockets` (12.0-16.0) - intentional for compatibility?
- `redis[hiredis]` - good choice for performance
- `prometheus-client` - monitoring-first architecture (good)

### Optional Dependencies
```toml
[project.optional-dependencies]
market-data = ["yfinance>=0.2.40,<0.3.0"]
```

**Assessment**: ✅ Good separation of optional features

**Recommendation**: Consider adding more optional groups:
```toml
[project.optional-dependencies]
market-data = ["yfinance>=0.2.40,<0.3.0"]
monitoring = ["prometheus-client>=0.23.1,<0.24.0"]  # Move from core if not always needed
aws = ["boto3>=1.0.0,<2.0.0"]  # If S3 backup is optional
```

### Dev Dependencies (16 packages)
```toml
[tool.poetry.group.dev.dependencies]
# Linting & Formatting
ruff = "^0.13.3"
black = "^25.9.0"

# Type Checking
mypy = "^1.18.2"
types-requests = "^2.32.0.20250913"
pandas-stubs = "^2.3.2.250926"
types-pyyaml = "^6.0.12.20250915"

# Testing
pytest = "^8.4.2"
pytest-asyncio = "^1.2.0"
pytest-cov = "^7.0.0"
pytest-mock = "^3.15.1"
pytest-xdist = "^3.0.0"
pytest-benchmark = "^5.1.0"

# Test Data Generation
faker = "^37.8.0"
freezegun = "^1.2.0"
hypothesis = "^6.140.2"
responses = "^0.25.8"

# Coverage
coverage = "^7.0.0"

# Workflow
pre-commit = "^4.3.0"

# Data (dev-only)
yfinance = "^0.2.66"
```

**Assessment**: ✅ Comprehensive testing & quality infrastructure

**Observations**:
- `black` AND `ruff` - redundant? Ruff can format too
- Excellent test tooling (async, mocking, benchmarks, property testing)
- Type stubs for major dependencies

**Recommendation**: Consider consolidating formatters:
```toml
# Option 1: Use ruff for everything (remove black)
ruff = "^0.13.3"

# Option 2: Keep both if team preference
# (no action needed)
```

---

## 3. Ruff Configuration

### Main Settings
```toml
[tool.ruff]
line-length = 100
target-version = "py312"
extend-exclude = [
    "data",
    "scripts/pickle_scanner.py",
    "scripts/pickle_to_joblib.py",
    "archived",
    "demos",
    "scripts",      # ⚠️ Excluding all scripts
    "tests",        # ⚠️ Excluding all tests
]
```

**Issues**:
1. ⚠️ **Tests excluded**: Tests should be linted! Only difference should be rule exceptions
2. ⚠️ **Scripts excluded**: Should lint scripts too (exclude specific files if needed)

**Recommendation**:
```toml
extend-exclude = [
    "data",
    "archived",
    "demos",
    "scripts/pickle_scanner.py",    # Keep specific exclusions
    "scripts/pickle_to_joblib.py",
    # Remove blanket "tests" and "scripts" exclusions
]
```

### Lint Rules
```toml
[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # Pyflakes
    "I",   # isort
    "W",   # pycodestyle warnings
]
ignore = [
    "ANN001",  # Missing type annotation for function argument
    "ANN002",  # Missing type annotation for *args
    "ANN003",  # Missing type annotation for **kwargs
    "ANN101",  # ⚠️ DEPRECATED - Remove
    "ANN102",  # ⚠️ DEPRECATED - Remove
    "ANN201",  # Missing return type annotation for public function
    "ANN202",  # Missing return type annotation for private function
    "ANN204",  # Missing return type annotation for special method
    "ANN401",  # Dynamically typed expressions (Any) are disallowed
    "S101",    # Use of assert detected
    "E501",    # Line too long (handled by line-length)
]
```

**Issues**:
1. ⚠️ `ANN101` and `ANN102` are deprecated (ruff warns about this)
2. ⚠️ Not selecting many useful rule groups (B, UP, etc.)
3. ℹ️ Ignoring all ANN rules - intentional for gradual typing adoption?

**Current Rule Coverage**: MINIMAL (only E, F, I, W)

**Recommendation**: Expand rule coverage gradually
```toml
[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # Pyflakes
    "I",   # isort
    "W",   # pycodestyle warnings
    "B",   # flake8-bugbear (common bugs)
    "UP",  # pyupgrade (modernization)
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]
ignore = [
    "ANN001", "ANN002", "ANN003",  # Function arg annotations
    # Remove: "ANN101", "ANN102",  # <- DELETE (deprecated)
    "ANN201", "ANN202", "ANN204",  # Return annotations
    "ANN401",  # Any disallowed
    "S101",    # assert allowed
    "E501",    # Line length (use line-length setting)
]
```

### Per-File Ignores
```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["ANN"]
```

**Assessment**: ✅ Reasonable - tests don't need full type annotations

**Enhancement**:
```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["ANN", "S101"]  # Allow assert in tests
"scripts/**/*.py" = ["ANN"]         # Scripts can be less strict
```

---

## 4. Black Configuration

```toml
[tool.black]
line-length = 100
target-version = ["py312"]
```

**Assessment**: ✅ Consistent with ruff (same line length)

**Note**: Ruff can replace black entirely with `ruff format`. Consider consolidation.

---

## 5. Mypy Configuration

```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true       # Strict!
no_implicit_optional = true        # Strict!
strict_optional = true             # Strict!
ignore_missing_imports = true      # Lenient (for untyped deps)
explicit_package_bases = true
mypy_path = "src"
```

**Assessment**: ⚠️ **STRICT CONFIG vs REALITY MISMATCH**

**Issue**: Configuration is very strict (`disallow_untyped_defs = true`), but we have 393 type errors across 80 files. This suggests:
1. Config was set aspirationally (target state)
2. Enforcement hasn't been enabled in CI yet
3. Legacy code predates strict enforcement

**Current Behavior**:
- Mypy runs with strict settings locally
- Errors are logged but don't block (no CI gate)

**Recommendation - Gradual Strictness**:
```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true

# Start lenient, tighten incrementally
disallow_untyped_defs = false          # Turn on per-module
no_implicit_optional = true            # Keep strict
strict_optional = true                 # Keep strict
ignore_missing_imports = true
explicit_package_bases = true
mypy_path = "src"

# Per-module strictness (opt-in approach)
[[tool.mypy.overrides]]
module = [
    "bot_v2.features.adaptive_portfolio.*",  # Well-typed modules
    "bot_v2.features.position_sizing.*",
    "bot_v2.features.data.*",
]
disallow_untyped_defs = true

# High-error modules stay lenient until fixed
[[tool.mypy.overrides]]
module = [
    "bot_v2.features.brokerages.coinbase.*",
    "bot_v2.validation.*",
    "bot_v2.cli.*",
]
disallow_untyped_defs = false
warn_return_any = false
```

---

## 6. Pytest Configuration

### pytest.ini
```ini
[pytest]
pythonpath = src
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
norecursedirs = archived scripts

addopts =
    --verbose
    --tb=short
    --strict-config
    --strict-markers
    --durations=10
    --maxfail=10
    -p no:warnings
    -m "not integration and not real_api and not uses_mock_broker"

minversion = 7.0
```

**Assessment**: ✅ **EXCELLENT** - Sensible defaults

**Highlights**:
- ✅ Skips integration/real API tests by default (fast feedback)
- ✅ Shows slowest 10 tests (`--durations=10`)
- ✅ Fails fast (`--maxfail=10`)
- ✅ Strict about markers and config
- ✅ Excludes archived code from test discovery

**Minor Enhancement**:
```ini
addopts =
    --verbose
    --tb=short
    --strict-config
    --strict-markers
    --durations=10
    --maxfail=10
    -p no:warnings
    -m "not integration and not real_api and not uses_mock_broker"
    --color=yes                    # Add: Force color output in CI
    --code-highlight=yes           # Add: Syntax highlighting in errors
```

### Pytest Markers (17 markers!)
```ini
markers =
    endpoints: Coinbase API endpoint tests
    perf: Performance benchmarks (opt-in)
    performance: Performance benchmarks (alias)
    integration: Integration tests (skipped by default)
    perps: Perpetual futures trading tests
    real_api: Real Coinbase API tests (opt-in)
    uses_mock_broker: Legacy mock broker tests (skipped)
    asyncio: Async tests
    brokerages: Brokerage adapter tests
    monitoring: Monitoring system tests
    orchestration: Orchestration layer tests
    state_management: State management tests
    security: Security and auth tests
    high_impact: High-impact tests (auto-run in CI)
    characterization: Behavior-preserving tests
    slow: Slow tests (>1s runtime, opt-in)
```

**Assessment**: ✅ **EXCELLENT** - Well-organized marker system

**Observations**:
- Good separation: unit (default) vs integration vs real API
- Domain-based markers (brokerages, monitoring, orchestration, etc.)
- Performance/speed markers (perf, slow)
- Special markers (high_impact, characterization)

**Recommendation**: Document marker usage
```bash
# Run only fast unit tests (default)
pytest

# Run integration tests
pytest -m integration

# Run brokerage tests
pytest -m brokerages

# Run high-impact tests (for CI)
pytest -m high_impact

# Run slow tests (opt-in)
pytest -m slow
```

---

## 7. Coverage Configuration

```toml
[tool.coverage.run]
source = ["src/bot_v2"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/archived/*",
]

[tool.coverage.report]
fail_under = 85                  # ✅ Current: 87.52%
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]

# Per-package coverage thresholds (commented - future phases)
# Phase 0 targets: 85%
# Phase 1 targets: 92%
# Phase 2 targets: 97%
# Phase 3 targets: 100%
```

**Assessment**: ✅ **EXCELLENT** - Well-configured with phased goals

**Highlights**:
- ✅ Current coverage (87.52%) exceeds threshold (85%)
- ✅ Reasonable exclusions (test code, repr, abstract methods)
- ✅ Phased improvement plan documented

**Recommendation**: Track per-module coverage
```toml
[tool.coverage.report]
fail_under = 85
show_missing = true
skip_covered = false

# Consider adding per-module thresholds for critical paths
# (requires pytest-cov or coverage config plugins)
```

---

## 8. Build System

```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
packages = [{ include = "bot_v2", from = "src" }]
```

**Assessment**: ✅ Standard Poetry setup

---

## Summary of Recommendations

### High Priority (Phase 0 - This Week)
1. ✅ **Remove deprecated ruff rules**:
   - Delete `ANN101`, `ANN102` from `ignore` list in `pyproject.toml`

2. ⚠️ **Fix ruff exclusions**:
   - Remove `tests` and `scripts` from `extend-exclude`
   - Keep specific file exclusions only

3. ⚠️ **Align mypy strictness with reality**:
   - Set `disallow_untyped_defs = false` globally
   - Enable strict mode per-module for well-typed code
   - Create gradual adoption plan

### Medium Priority (Phase 1 - Next 2 Weeks)
4. 📈 **Expand ruff rule coverage**:
   - Add `B` (bugbear), `UP` (pyupgrade), `C4` (comprehensions), `SIM` (simplify)

5. 🧹 **Consider black vs ruff consolidation**:
   - Evaluate replacing `black` with `ruff format`
   - Or document why both are kept

6. 📊 **Per-module coverage tracking**:
   - Investigate per-module coverage thresholds
   - Identify low-coverage modules for Phase 1 work

### Low Priority (Phase 2+)
7. 🔧 **Optional dependency reorganization**:
   - Consider splitting out `monitoring`, `aws`, etc. as optional groups

8. 📚 **Document marker usage**:
   - Add marker usage examples to `CONTRIBUTING.md` or test docs

---

## Configuration Audit Checklist

- [x] pyproject.toml is valid (poetry check passed)
- [x] All tool sections present (ruff, mypy, pytest, coverage, black)
- [x] Dependencies appropriately pinned
- [x] Dev dependencies comprehensive
- [ ] Ruff excluding too much (tests, scripts) ⚠️
- [ ] Deprecated ruff rules present (ANN101, ANN102) ⚠️
- [ ] Mypy config too strict vs reality (393 errors) ⚠️
- [x] Pytest configuration excellent
- [x] Coverage tracking active and healthy
- [x] Marker system well-organized

**Overall Grade**: B+ (Would be A after fixing the 3 warnings)

---

**Next Steps**: Review `.pre-commit-config.yaml` (Task 5)
