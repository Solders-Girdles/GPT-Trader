# Dependency Policy & Tree Snapshot

**Generated**: 2025-10-05
**Purpose**: Phase 0 dependency baseline, policy guidelines, and upgrade playbook

---

## Executive Summary

**Total Packages**: 33 top-level (direct dependencies)
**Dependency Tree Depth**: ~173 total packages (including transitive)
**Python Version**: 3.12.x (strict)
**Package Manager**: Poetry
**Lock File Status**: ✅ `poetry.lock` present and up-to-date

### Health Indicators
- ✅ All dependencies pinned with version ranges
- ✅ No known security vulnerabilities (assumed - run `poetry audit` to verify)
- ✅ Optional dependencies properly separated (`market-data`)
- ⚠️ Wide version range on `websockets` (12.0-16.0) - intentional?

---

## Dependency Categories

### 1. Production Dependencies (13 packages)

#### Brokerage Integration
- **coinbase-advanced-py** `1.8.2` (≥1.8.2, <2.0.0)
  - Official Coinbase Advanced Trade API SDK
  - Transitive: `backoff`, `cryptography`, `pyjwt`, `requests`, `websockets`

#### Data Processing & Analysis
- **pandas** `2.3.3` (≥2.2.2, <3.0.0)
  - DataFrame manipulation, time series analysis
  - Transitive: `numpy`, `python-dateutil`, `pytz`, `tzdata`
- **numpy** `1.26.4` (≥1.26.4, <2.0.0)
  - Numerical computing foundation

#### Data Validation & Configuration
- **pydantic** `2.11.10` (≥2.7.4, <3.0.0)
  - Data validation and settings management
  - Transitive: `annotated-types`, `pydantic-core`, `typing-extensions`, `typing-inspection`
- **python-dotenv** `1.1.1` (≥1.0.1, <2.0.0)
  - Environment variable loading from `.env` files
- **pyyaml** `6.0.3` (≥6.0.1, <7.0.0)
  - YAML parsing for config files

#### Networking & Communication
- **requests** `2.32.5` (≥2.32.3, <3.0.0)
  - HTTP client library
  - Transitive: `certifi`, `charset-normalizer`, `idna`, `urllib3`
- **aiohttp** `3.12.15` (≥3.12.15, <4.0.0)
  - Async HTTP client/server framework
  - Transitive: `aiohappyeyeballs`, `aiosignal`, `attrs`, `frozenlist`, `multidict`, `propcache`, `yarl`
- **websockets** `13.1` (≥12.0, <16.0)
  - WebSocket Protocol implementation
  - **Note**: Wide version range (12.0-16.0) - verify intentional

#### State & Caching
- **redis** `6.4.0` (≥6.0.0, <7.0.0) with `[hiredis]` extra
  - Redis client with high-performance parser
  - Transitive: `hiredis`

#### Security & Authentication
- **cryptography** `46.0.2` (≥46.0.0, <47.0.0)
  - Cryptographic primitives and recipes
  - Transitive: `cffi`, `pycparser`
- **pyotp** `2.9.0` (≥2.9.0, <3.0.0)
  - One-Time Password (2FA) library

#### Monitoring & Observability
- **prometheus-client** `0.23.1` (≥0.23.1, <0.24.0)
  - Prometheus metrics exporter
- **psutil** `7.1.0` (≥7.0.0, <8.0.0)
  - System and process monitoring

---

### 2. Optional Dependencies (1 group)

#### Market Data (optional)
```toml
[project.optional-dependencies]
market-data = ["yfinance>=0.2.40,<0.3.0"]
```

- **yfinance** `0.2.66` (≥0.2.40, <0.3.0)
  - Yahoo Finance market data downloader
  - Transitive: `beautifulsoup4`, `curl-cffi`, `frozendict`, `multitasking`, `numpy`, `pandas`, `peewee`, `platformdirs`, `protobuf`, `pytz`, `requests`, `websockets`

**Usage**: `poetry install --extras market-data` or `pip install gpt-trader[market-data]`

**Future Additions** (recommended):
```toml
[project.optional-dependencies]
market-data = ["yfinance>=0.2.40,<0.3.0"]
aws = ["boto3>=1.35.0,<2.0.0"]          # S3 backups
monitoring = ["prometheus-client>=0.23.1,<0.24.0"]  # Move from core if optional
development = ["ipython>=8.0.0", "jupyter>=1.0.0"]  # Data exploration
```

---

### 3. Development Dependencies (20 packages)

#### Linting & Formatting
- **ruff** `0.13.3`
  - Fast Python linter and formatter (Rust-based)
- **black** `25.9.0`
  - Opinionated code formatter
  - Transitive: `click`, `mypy-extensions`, `packaging`, `pathspec`, `platformdirs`, `pytokens`

#### Type Checking
- **mypy** `1.18.2`
  - Static type checker
  - Transitive: `mypy-extensions`, `pathspec`, `typing-extensions`
- **types-requests** `2.32.4.20250913`
  - Type stubs for requests
- **pandas-stubs** `2.3.2.250926`
  - Type stubs for pandas
- **types-pyyaml** `6.0.12.20250915`
  - Type stubs for PyYAML

#### Testing Framework
- **pytest** `8.4.2`
  - Testing framework
  - Transitive: `colorama`, `iniconfig`, `packaging`, `pluggy`, `pygments`
- **pytest-asyncio** `1.2.0`
  - Async test support
- **pytest-cov** `7.0.0`
  - Coverage plugin
  - Transitive: `coverage`, `pluggy`
- **pytest-mock** `3.15.1`
  - Mocking utilities
- **pytest-xdist** `3.8.0`
  - Parallel test execution
  - Transitive: `execnet`
- **pytest-benchmark** `5.1.0`
  - Performance benchmarking
  - Transitive: `py-cpuinfo`

#### Test Data & Mocking
- **faker** `37.8.0`
  - Fake data generation
  - Transitive: `tzdata`
- **freezegun** `1.5.5`
  - Time mocking for tests
  - Transitive: `python-dateutil`, `six`
- **hypothesis** `6.140.3`
  - Property-based testing
  - Transitive: `attrs`, `sortedcontainers`
- **responses** `0.25.8`
  - HTTP request mocking
  - Transitive: `pyyaml`, `requests`, `urllib3`

#### Coverage
- **coverage** `7.10.7`
  - Code coverage measurement

#### Workflow
- **pre-commit** `4.3.0`
  - Git hook management
  - Transitive: `cfgv`, `identify`, `nodeenv`, `pyyaml`, `virtualenv`, `distlib`, `filelock`, `platformdirs`

#### Data (Dev-only)
- **yfinance** `0.2.66`
  - Same as optional dependency, included in dev for convenience

---

## Dependency Tree Snapshot (2025-10-05)

### Full Tree Export

Run `poetry show --tree` for current state. Snapshot saved to this document on 2025-10-05.

<details>
<summary>Click to expand full dependency tree (173 lines)</summary>

```
aiohttp 3.12.15 Async http client/server framework (asyncio)
├── aiohappyeyeballs >=2.5.0
├── aiosignal >=1.4.0
│   ├── frozenlist >=1.1.0
│   └── typing-extensions >=4.2
├── attrs >=17.3.0
├── frozenlist >=1.1.1
├── multidict >=4.5,<7.0
├── propcache >=0.2.0
└── yarl >=1.17.0,<2.0
    ├── idna >=2.0
    ├── multidict >=4.0
    └── propcache >=0.2.1
black 25.9.0 The uncompromising code formatter.
├── click >=8.0.0
│   └── colorama *
├── mypy-extensions >=0.4.3
├── packaging >=22.0
├── pathspec >=0.9.0
├── platformdirs >=2
└── pytokens >=0.1.10
coinbase-advanced-py 1.8.2 Coinbase Advanced API Python SDK
├── backoff >=2.2.1
├── cryptography >=42.0.4
│   └── cffi >=2.0.0
│       └── pycparser *
├── pyjwt >=2.8.0
├── requests >=2.31.0
│   ├── certifi >=2017.4.17
│   ├── charset-normalizer >=2,<4
│   ├── idna >=2.5,<4
│   └── urllib3 >=1.21.1,<3
└── websockets >=12.0,<14.0
coverage 7.10.7 Code coverage measurement for Python
cryptography 46.0.2 cryptography is a package which provides cryptographic recipes...
└── cffi >=2.0.0
    └── pycparser *
faker 37.8.0 Faker is a Python package that generates fake data for you.
└── tzdata *
freezegun 1.5.5 Let your Python tests travel through time
└── python-dateutil >=2.7
    └── six >=1.5
hypothesis 6.140.3 A library for property-based testing
├── attrs >=22.2.0
└── sortedcontainers >=2.1.0,<3.0.0
mypy 1.18.2 Optional static typing for Python
├── mypy-extensions >=1.0.0
├── pathspec >=0.9.0
└── typing-extensions >=4.6.0
numpy 1.26.4 Fundamental package for array computing in Python
pandas 2.3.3 Powerful data structures for data analysis...
├── numpy >=1.26.0
├── python-dateutil >=2.8.2
│   └── six >=1.5
├── pytz >=2020.1
└── tzdata >=2022.7
pandas-stubs 2.3.2.250926 Type annotations for pandas
├── numpy >=1.23.5
└── types-pytz >=2022.1.1
pre-commit 4.3.0 A framework for managing and maintaining...
├── cfgv >=2.0.0
├── identify >=1.0.0
├── nodeenv >=0.11.1
├── pyyaml >=5.1
└── virtualenv >=20.10.0
    ├── distlib >=0.3.7,<1
    ├── filelock >=3.12.2,<4
    └── platformdirs >=3.9.1,<5
prometheus-client 0.23.1 Python client for the Prometheus monitoring system.
psutil 7.1.0 Cross-platform lib for process and system monitoring.
pydantic 2.11.10 Data validation using Python type hints
├── annotated-types >=0.6.0
├── pydantic-core 2.33.2
│   └── typing-extensions >=4.6.0,<4.7.0 || >4.7.0
├── typing-extensions >=4.12.2
└── typing-inspection >=0.4.0
    └── typing-extensions >=4.12.0
pyotp 2.9.0 Python One Time Password Library
pytest 8.4.2 pytest: simple powerful testing with Python
├── colorama >=0.4
├── iniconfig >=1
├── packaging >=20
├── pluggy >=1.5,<2
└── pygments >=2.7.2
pytest-asyncio 1.2.0 Pytest support for asyncio
├── pytest >=8.2,<9
│   ├── colorama >=0.4
│   ├── iniconfig >=1
│   ├── packaging >=20
│   ├── pluggy >=1.5,<2
│   └── pygments >=2.7.2
└── typing-extensions >=4.12
pytest-benchmark 5.1.0 A pytest fixture for benchmarking code...
├── py-cpuinfo *
└── pytest >=8.1
    ├── colorama >=0.4
    ├── iniconfig >=1
    ├── packaging >=20
    ├── pluggy >=1.5,<2
    └── pygments >=2.7.2
pytest-cov 7.0.0 Pytest plugin for measuring coverage.
├── coverage >=7.10.6
├── pluggy >=1.2
└── pytest >=7
    ├── colorama >=0.4
    ├── iniconfig >=1
    ├── packaging >=20
    ├── pluggy >=1.5,<2
    └── pygments >=2.7.2
pytest-mock 3.15.1 Thin-wrapper around the mock package...
└── pytest >=6.2.5
    ├── colorama >=0.4
    ├── iniconfig >=1
    ├── packaging >=20
    ├── pluggy >=1.5,<2
    └── pygments >=2.7.2
pytest-xdist 3.8.0 pytest xdist plugin for distributed testing...
├── execnet >=2.1
└── pytest >=7.0.0
    ├── colorama >=0.4
    ├── iniconfig >=1
    ├── packaging >=20
    ├── pluggy >=1.5,<2
    └── pygments >=2.7.2
python-dotenv 1.1.1 Read key-value pairs from a .env file...
pyyaml 6.0.3 YAML parser and emitter for Python
redis 6.4.0 Python client for Redis database and key-value store
└── hiredis >=3.2.0
requests 2.32.5 Python HTTP for Humans.
├── certifi >=2017.4.17
├── charset-normalizer >=2,<4
├── idna >=2.5,<4
└── urllib3 >=1.21.1,<3
responses 0.25.8 A utility library for mocking out the requests...
├── pyyaml *
├── requests >=2.30.0,<3.0
│   ├── certifi >=2017.4.17
│   ├── charset-normalizer >=2,<4
│   ├── idna >=2.5,<4
│   └── urllib3 >=1.21.1,<3
└── urllib3 >=1.25.10,<3.0
ruff 0.13.3 An extremely fast Python linter and code formatter...
types-pyyaml 6.0.12.20250915 Typing stubs for PyYAML
types-requests 2.32.4.20250913 Typing stubs for requests
└── urllib3 >=2
websockets 13.1 An implementation of the WebSocket Protocol...
yfinance 0.2.66 Download market data from Yahoo! Finance API
├── beautifulsoup4 >=4.11.1
│   ├── soupsieve >1.2
│   └── typing-extensions >=4.0.0
├── curl-cffi >=0.7
│   ├── certifi >=2024.2.2
│   └── cffi >=1.12.0
│       └── pycparser *
├── frozendict >=2.3.4
├── multitasking >=0.0.7
├── numpy >=1.16.5
├── pandas >=1.3.0
│   ├── numpy >=1.26.0
│   ├── python-dateutil >=2.8.2
│   │   └── six >=1.5
│   ├── pytz >=2020.1
│   └── tzdata >=2022.7
├── peewee >=3.16.2
├── platformdirs >=2.0.0
├── protobuf >=3.19.0
├── pytz >=2022.5
├── requests >=2.31
│   ├── certifi >=2017.4.17
│   ├── charset-normalizer >=2,<4
│   ├── idna >=2.5,<4
│   └── urllib3 >=1.21.1,<3
└── websockets >=13.0
```

</details>

---

## Dependency Pinning Strategy

### Current Approach: **Caret Ranges** (Poetry default)

```toml
# Example from pyproject.toml
pandas = ">=2.2.2,<3.0.0"    # Allow minor/patch updates
numpy = ">=1.26.4,<2.0.0"    # Allow minor/patch updates
```

**Pros**:
- ✅ Receives security patches automatically
- ✅ Compatible with semantic versioning
- ✅ Reduces manual update burden

**Cons**:
- ⚠️ Can introduce breaking changes if deps don't follow semver
- ⚠️ Requires regular `poetry update` + testing

### Recommended Approach: **Conservative Ranges**

For production-critical dependencies:
```toml
# Tighter ranges for critical deps
coinbase-advanced-py = ">=1.8.2,<1.9.0"  # Only patch updates
pydantic = ">=2.11.0,<2.12.0"            # Only patch updates

# Standard ranges for well-maintained deps
pandas = ">=2.2.2,<3.0.0"    # Minor updates OK
numpy = ">=1.26.4,<2.0.0"    # Minor updates OK
```

**Implementation**: Phase 1 task to review and tighten critical deps

---

## Upgrade Playbook

### Quarterly Dependency Updates (Recommended Schedule)

**When**: First week of each quarter (January, April, July, October)

**Process**:

#### 1. Audit Current State
```bash
# Check for outdated packages
poetry show --outdated

# Check for security vulnerabilities (requires poetry plugin)
poetry audit
# OR
pip-audit

# Review latest versions
poetry show --latest
```

#### 2. Update Strategy Decision

**Option A: Conservative Update** (Recommended)
```bash
# Update only patch versions (safest)
poetry update --lock
```

**Option B: Minor Version Update** (Quarterly)
```bash
# Update to latest compatible versions
poetry update

# Review changes
git diff poetry.lock
```

**Option C: Major Version Update** (Annual, requires planning)
```bash
# Update one package at a time
poetry add pandas@latest --dry-run

# Test thoroughly before committing
```

#### 3. Testing After Update
```bash
# Run full test suite
poetry run pytest tests/

# Run type checking
poetry run mypy src/bot_v2

# Run linting
poetry run ruff check src/bot_v2

# Integration tests (if available)
poetry run pytest tests/ -m integration

# Smoke test critical paths
poetry run perps-bot --profile dev --dry-run --dev-fast
```

#### 4. Documentation
```bash
# Update this file with new snapshot
poetry show --tree > /tmp/new_tree.txt
# Update dependency_policy.md with new snapshot + date

# Update changelog
# Add entry to CHANGELOG.md or release notes
```

#### 5. Rollback Plan
```bash
# If issues found, rollback
git checkout HEAD -- poetry.lock
poetry install

# Or revert to specific version
poetry add pandas@2.2.2
```

---

## Known Version Constraints & Conflicts

### 1. Python Version Lock
```toml
requires-python = ">=3.12,<3.13"
```

**Rationale**: Tested on Python 3.12 only
**Impact**: Blocks Python 3.13 until explicitly tested
**Future Action**: Test Python 3.13 compatibility in Phase 1, update constraint if compatible

---

### 2. Websockets Wide Range
```toml
websockets = ">=12.0,<16.0"
```

**Current**: `13.1` installed
**Range**: Very wide (12.0-16.0)
**Question**: Is this intentional for forward compatibility, or should it be tightened?

**Recommendation**: Verify with team, consider tightening:
```toml
websockets = ">=13.0,<14.0"  # More conservative
```

---

### 3. Coinbase SDK Upper Bound
```toml
coinbase-advanced-py = ">=1.8.2,<2.0.0"
```

**Current**: `1.8.2`
**Latest**: Check `poetry show coinbase-advanced-py --latest`
**Risk**: May have newer versions with bug fixes

**Action**: Check for updates, test if newer 1.x versions available

---

### 4. No Conflicts Detected
```bash
poetry check
# All set!
```

**Status**: ✅ No dependency conflicts in current lock file

---

## Transitive Dependency Outliers

### Heavily Duplicated (Multiple Versions)
**None detected** - Good! All transitive deps resolve to single versions.

### Deep Dependency Chains
1. **yfinance** - 13 transitive deps (heaviest)
2. **aiohttp** - 9 transitive deps
3. **coinbase-advanced-py** - 5 transitive deps
4. **pydantic** - 4 transitive deps

**Assessment**: All reasonable for their respective functionalities

---

## Security Considerations

### Current Status
**Last Audit**: Not yet run (Phase 0 task)

### Recommended Tools
1. **poetry-audit-plugin** (if available)
   ```bash
   poetry self add poetry-audit-plugin
   poetry audit
   ```

2. **pip-audit** (OSV database)
   ```bash
   pip install pip-audit
   pip-audit
   ```

3. **safety** (commercial, free tier available)
   ```bash
   pip install safety
   safety check
   ```

### Audit Schedule
- **High-risk dependencies** (auth, crypto, network): Monthly
- **All dependencies**: Quarterly (with version updates)
- **Critical vulnerabilities**: Immediately on disclosure

---

## CI/CD Integration

### Recommended GitHub Actions

```yaml
# .github/workflows/dependency-check.yml
name: Dependency Audit

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday
  pull_request:
    paths:
      - 'poetry.lock'
      - 'pyproject.toml'

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install pip-audit
      - run: pip-audit --requirement <(poetry export -f requirements.txt)
```

---

## Phase 0 Action Items

### Immediate
1. ✅ Document current dependency tree (this file)
2. [ ] Run security audit: `pip-audit` or `poetry audit`
3. [ ] Document any found vulnerabilities
4. [ ] Review `websockets` version range - tighten or document rationale
5. [ ] Check `coinbase-advanced-py` for newer 1.x versions

### Phase 1 (Next 2 Weeks)
1. [ ] Test Python 3.13 compatibility
2. [ ] Tighten critical dependency ranges (coinbase, pydantic, cryptography)
3. [ ] Add `boto3` to optional dependencies (for S3 backups)
4. [ ] Set up automated dependency audit in CI
5. [ ] Create quarterly update calendar reminder

---

## Appendix: Quick Reference Commands

```bash
# Install dependencies
poetry install                        # Production deps only
poetry install --with dev             # Include dev deps
poetry install --extras market-data   # Include optional market-data

# Update dependencies
poetry update                         # Update all to latest compatible
poetry update --lock                  # Only update lock, don't upgrade versions
poetry update pandas                  # Update specific package

# Check status
poetry show --outdated                # List outdated packages
poetry show --latest                  # Show latest available versions
poetry show --tree                    # Show dependency tree
poetry check                          # Verify no conflicts

# Export for pip users
poetry export -f requirements.txt --output requirements.txt
poetry export -f requirements.txt --with dev --output requirements-dev.txt
```

---

**Next Steps**: Scan config drift (Task 7 - Final Phase 0 task)
