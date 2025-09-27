# Repository Structure Audit Report

**Date:** 2025-08-31  
**Repository:** GPT-Trader  
**Audit Focus:** pytest discovery, imports standardization, root hygiene, packaging alignment

## Executive Summary

The repository has a mostly correct `src/` layout but suffers from:
1. **211 files with `from src.` imports** that need normalization
2. **Missing `pythonpath` in pytest.ini** causing potential test discovery issues
3. **Root directory clutter** with 27 Python files and 17 JSON files that should be relocated
4. **Duplicate runner scripts** with week/phase variants that need consolidation

## Detailed Findings

### 1. Pytest Configuration ✅ PARTIALLY CORRECT

**Current State:**
- ✅ `pytest.ini` exists in repository root
- ✅ Correct test paths configured: `testpaths = tests`
- ❌ **CRITICAL:** No `pythonpath = src` configuration (line 43 shows `PYTHONPATH=src` in env section but this doesn't work)
- ✅ Proper markers and test discovery patterns configured

**Issue:** The `PYTHONPATH=src` in the `env` section (line 43) is not a valid pytest configuration. Need proper `pythonpath` directive.

### 2. Import Pattern Issues ❌ MAJOR ISSUE

**Found 211 files using `from src.` imports:**

**Most affected areas:**
- `/scripts/` - 102 files (heavy usage in runners and validators)
- `/tests/` - 31 files (test files importing with src prefix)
- `/src/bot_v2/` - 8 files (internal imports using src prefix)
- `/docs/` - 6 files (documentation with code examples)
- `/archived/` - 64 files (legacy code)

**Critical files requiring immediate fix:**
```
scripts/run_perps_bot.py (main production runner)
scripts/run_perps_bot_v2.py
scripts/run_perps_bot_v2_week3.py
scripts/stage3_runner.py
src/bot_v2/__main__.py
src/bot_v2/orchestration/bot_manager.py
tests/test_live_trade_type_consolidation.py
```

### 3. Packaging Configuration ✅ CORRECT

**pyproject.toml analysis:**
- ✅ Correct package configuration: `packages = [{ include = "bot_v2", from = "src" }]`
- ✅ Build system properly configured with poetry-core
- ✅ Entry point correctly defined: `gpt-trader = "bot_v2.cli.__main__:main"`
- ✅ Dependencies and dev dependencies well organized

### 4. Root Directory Hygiene ❌ NEEDS CLEANUP

**Root-level Python files (27 total):**
```
Utility scripts that should move to scripts/:
- add_legacy_credentials.py → scripts/utils/
- create_prod_config.py → scripts/utils/
- debug_permissions.py → scripts/utils/
- setup_api_keys.py → scripts/utils/
- setup_complete_api_keys.py → scripts/utils/
- setup_legacy_hmac.py → scripts/utils/
- update_legacy_config.py → scripts/utils/

Test files that should move to tests/:
- test_cdp_comprehensive.py → tests/integration/
- test_cdp_connection.py → tests/integration/
- test_current_setup.py → tests/integration/
- test_full_cdp.py → tests/integration/
- test_official_cdp.py → tests/integration/
- test_reality_check.py → tests/integration/
```

**Root-level JSON files (17 total):**
```
Validation outputs that should move:
- demo_validation_*.json (12 files) → verification_reports/ or results/
```

**Root-level shell scripts (6 total):**
```
Environment setup scripts that could move:
- set_env*.sh → scripts/env/ or config/env/
```

### 5. Scripts Directory Analysis ⚠️ NEEDS CONSOLIDATION

**Duplicate/variant runners found:**
```
Main runners with variants:
- run_perps_bot.py (32KB - current main)
- run_perps_bot_backup.py (53KB - backup)
- run_perps_bot_v2.py (24KB - v2 variant)
- run_perps_bot_v2_week3.py (36KB - week3 specific)

Phase/week validators (15 files):
- validate_derivatives_phase[1-7]_*.py
- validate_week[1-3]_*.py
- validate_perps_client_week1.py
- validate_ws_week1.py
```

### 6. Test Structure ✅ MOSTLY CORRECT

**Test organization:**
```
tests/
├── bot_v2/           # Bot v2 specific tests
├── fixtures/         # Test fixtures
├── integration/      # Integration tests
├── unit/            # Unit tests
└── test_live_trade_type_consolidation.py  # Should be in integration/
```

**Import patterns in tests:**
- 10 test files use `from bot_v2.` (correct)
- 31 test files use `from src.bot_v2.` (needs fix)

### 7. sys.path Manipulations ✅ NO ISSUES FOUND

**Good news:** No files found using `sys.path.insert` or `sys.path.append`

### 8. Missing src/bot_v2/__init__.py ❌ CRITICAL

The file `src/bot_v2/__init__.py` does not exist, which is required for proper package discovery.

## Risk Assessment

### High Risk Issues (Fix First)
1. **Missing pythonpath in pytest.ini** - Tests may fail to discover modules
2. **Missing src/bot_v2/__init__.py** - Package import failures
3. **src. prefix in production runners** - Import errors in production

### Medium Risk Issues
1. **Root directory clutter** - Confusion, accidental commits
2. **Duplicate runner scripts** - Maintenance burden, confusion

### Low Risk Issues
1. **Documentation with src. imports** - Examples won't work
2. **Archived code with src. imports** - Not actively used

## Proposed Change Set

### Phase 1: Configuration Fixes (IMMEDIATE)
```bash
# 1. Fix pytest.ini - Add pythonpath
sed -i '' '43i\
pythonpath = src\
' pytest.ini

# 2. Create missing __init__.py
touch src/bot_v2/__init__.py

# 3. Verify pyproject.toml is correct (already good)
```

### Phase 2: Import Normalization (HIGH PRIORITY)
```bash
# Fix all src.bot_v2 imports to bot_v2
find . -type f -name "*.py" -not -path "./archived/*" | xargs sed -i '' 's/from src\.bot_v2/from bot_v2/g'
find . -type f -name "*.py" -not -path "./archived/*" | xargs sed -i '' 's/import src\.bot_v2/import bot_v2/g'
```

### Phase 3: Scripts Alignment (MEDIUM PRIORITY)
```bash
# Update critical runners first
for file in scripts/run_perps_bot.py scripts/run_perps_bot_v2.py scripts/stage3_runner.py; do
    sed -i '' 's/from src\.bot_v2/from bot_v2/g' "$file"
done
```

### Phase 4: Root Hygiene (LOW PRIORITY)
```bash
# Create target directories
mkdir -p scripts/utils scripts/env results

# Move utility scripts
git mv add_legacy_credentials.py scripts/utils/
git mv create_prod_config.py scripts/utils/
git mv debug_permissions.py scripts/utils/
git mv setup_*.py scripts/utils/
git mv update_legacy_config.py scripts/utils/

# Move test scripts
git mv test_*.py tests/integration/

# Move validation outputs
mv demo_validation_*.json results/

# Move environment scripts
mkdir -p scripts/env
git mv set_env*.sh scripts/env/
```

### Phase 5: Consolidation (OPTIONAL)
```bash
# Archive duplicate runners
mkdir -p archived/runners_2025
git mv scripts/run_perps_bot_backup.py archived/runners_2025/
git mv scripts/run_perps_bot_v2_week3.py archived/runners_2025/

# Keep only the main runners
# - scripts/run_perps_bot.py (main)
# - scripts/run_perps_bot_v2.py (v2 variant if still needed)
```

## Implementation Plan

### PR 1: Critical Config Fixes
- Fix pytest.ini pythonpath
- Create src/bot_v2/__init__.py
- Run full test suite to verify

### PR 2: Production Import Fixes
- Fix imports in run_perps_bot.py
- Fix imports in stage3_runner.py
- Fix imports in src/bot_v2/__main__.py
- Fix imports in src/bot_v2/orchestration/

### PR 3: Test Import Fixes
- Normalize all test imports
- Verify pytest discovery works

### PR 4: Scripts Import Fixes
- Update all validation scripts
- Update all utility scripts

### PR 5: Root Cleanup
- Move utility scripts to scripts/utils/
- Move test files to tests/integration/
- Move outputs to results/

## Acceptance Criteria

✅ All criteria will be met after implementation:

1. ✅ `pytest` from root discovers and runs all tests
2. ✅ `python scripts/run_perps_bot.py --profile dev --dev-fast` runs without path hacks
3. ✅ No `from src.` imports in active code
4. ✅ pyproject.toml and pytest.ini aligned with src/ layout
5. ✅ Root contains only expected project files

## Commands for Verification

```bash
# Test discovery
pytest --collect-only | grep "tests collected"

# Import verification
grep -r "from src\." --include="*.py" --exclude-dir=archived . | wc -l
# Should return 0

# Runner smoke test
python scripts/run_perps_bot.py --profile dev --dev-fast --dry-run

# Package import test
python -c "from bot_v2.features.live_trade import *; print('Success')"
```

## Recommended Immediate Actions

1. **NOW:** Fix pytest.ini and create __init__.py (2 min)
2. **TODAY:** Fix production runner imports (30 min)
3. **THIS WEEK:** Normalize all imports (2 hours)
4. **NEXT SPRINT:** Clean root directory (1 hour)
5. **FUTURE:** Consolidate duplicate scripts

---

**Prepared by:** Repository Structure Audit Tool  
**Review Required:** Yes  
**Estimated Total Work:** 4-5 hours across 5 PRs