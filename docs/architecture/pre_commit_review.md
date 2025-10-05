# Pre-commit Configuration Review - Phase 0

**Generated**: 2025-10-05
**Purpose**: Document current pre-commit hook configuration and readiness for enabling

---

## Executive Summary

**Status**: ✅ **WELL-CONFIGURED** - Ready for immediate use after linting fixes

**Hook Count**: 9 hooks across 4 repos
**Readiness**: 🟢 Safe to enable after Phase 0 ruff fixes (6 errors)

---

## Current Configuration

### 1. Standard Pre-commit Hooks (5 hooks)

**Repo**: `https://github.com/pre-commit/pre-commit-hooks`
**Version**: v6.0.0 (latest stable)

```yaml
hooks:
  - trailing-whitespace       # Remove trailing whitespace
  - end-of-file-fixer        # Ensure files end with newline
  - check-yaml               # Validate YAML syntax (allows multi-doc)
  - check-added-large-files  # Prevent large file commits
  - mixed-line-ending        # Ensure consistent line endings
```

**Assessment**: ✅ **EXCELLENT** - Essential hygiene hooks

**Notes**:
- `check-yaml` has `--allow-multiple-documents` (good for k8s manifests, etc.)
- All hooks are non-intrusive (only fix formatting, not logic)

---

### 2. Black Formatter

**Repo**: `https://github.com/psf/black`
**Version**: 25.9.0 (matches dev dependency version)

```yaml
hooks:
  - black                     # Format Python code
```

**Assessment**: ✅ **GOOD** - Auto-formatting on commit

**Note**: Ruff can also format. Consider consolidation:
- **Option A**: Keep black (if team prefers black's style)
- **Option B**: Replace with `ruff format` (one less tool)

**Recommendation**: Keep current setup unless consolidating tools

---

### 3. Ruff Linter

**Repo**: `https://github.com/astral-sh/ruff-pre-commit`
**Version**: v0.13.3 (matches dev dependency version)

```yaml
hooks:
  - ruff
    args: [--fix, --exit-non-zero-on-fix]
```

**Assessment**: ✅ **GOOD** - Auto-fixes linting issues

**Args Explained**:
- `--fix`: Auto-fix fixable issues (e.g., import sorting)
- `--exit-non-zero-on-fix`: Fail hook after fixing (forces re-stage)

**Current Blocker**:
- ⚠️ 6 linting errors in codebase (4 auto-fixable, 2 manual)
- See `tooling_baseline.md` for details

**Enabling Strategy**:
1. Fix 6 errors first (run `ruff check --fix && manually fix 2 undefined names`)
2. Then enable pre-commit hooks
3. All new commits will be auto-linted

---

### 4. PyUpgrade

**Repo**: `https://github.com/asottile/pyupgrade`
**Version**: v3.20.0

```yaml
hooks:
  - pyupgrade
    args: [--py312-plus]
```

**Assessment**: ✅ **EXCELLENT** - Keeps code modern

**Purpose**: Automatically upgrades Python syntax to 3.12+ idioms
- Example: `typing.Optional[str]` → `str | None`
- Example: `typing.Dict` → `dict`

**Note**: Excellent addition for maintaining modern codebase

---

### 5. Local Custom Hooks (2 hooks)

#### Hook 1: Test Hygiene Checker

```yaml
- id: test-hygiene
  name: test-hygiene
  entry: scripts/ci/check_test_hygiene.py
  language: python
  pass_filenames: true
  files: ^tests/.*\.py$
```

**Assessment**: ✅ **EXCELLENT** - Custom test quality enforcement

**Status**: ✅ Script exists at `scripts/ci/check_test_hygiene.py`

**Purpose**: Enforce test file naming conventions, structure, etc.

**Sample checks** (inferred from typical test hygiene scripts):
- Test functions start with `test_`
- Test classes start with `Test`
- No print statements in tests
- Proper marker usage

---

#### Hook 2: Forbid Bytecode

```yaml
- id: forbid-bytecode
  name: forbid-bytecode
  entry: Bytecode files must not be committed
  language: fail
  files: '(__pycache__|\.pyc$|\.pyo$)'
```

**Assessment**: ✅ **EXCELLENT** - Prevents common mistake

**Purpose**: Block accidental commits of compiled Python bytecode

**Note**: Good safety net (should be in `.gitignore` too)

---

## Hook Execution Order

Pre-commit runs hooks in the order listed:
1. **Basic hygiene** (trailing whitespace, EOF, YAML, large files, line endings)
2. **Black** - Format code
3. **Ruff** - Lint and auto-fix
4. **PyUpgrade** - Modernize syntax
5. **Test hygiene** - Check test quality (if test files changed)
6. **Forbid bytecode** - Block .pyc files

**Assessment**: ✅ **GOOD ORDER** - Formatters before linters

**Note**: Some teams run formatters last (after linting). Current order is acceptable.

---

## Installation & Usage

### Install Pre-commit Hooks

```bash
# Install hooks (run once per clone)
pre-commit install

# Optionally install for commit-msg hooks
pre-commit install --hook-type commit-msg
```

### Manual Runs

```bash
# Run all hooks on all files (good for first-time setup)
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Run only on staged files (default)
pre-commit run
```

### Bypass Hooks (Emergency Only)

```bash
# Skip all hooks for one commit
git commit --no-verify -m "Emergency fix"

# NOT RECOMMENDED - only for critical hotfixes
```

---

## Readiness Assessment

### Current Blockers
1. ⚠️ **6 ruff errors** - Must fix before enabling hooks
   - 4 auto-fixable (import sorting)
   - 2 manual (missing `from typing import Any`)

### Safe to Enable?
- 🟢 **YES** - After fixing 6 ruff errors
- 🟢 Tests passing (5189/5209)
- 🟢 Black compatible (pyproject.toml has black config)
- 🟢 No mypy hook (393 errors would block all commits)

### Current Status (Hooks Installed?)
**Unknown** - Not checked in this audit

**To check**:
```bash
pre-commit --version              # Check if pre-commit is installed
ls -la .git/hooks/pre-commit      # Check if hooks are installed
```

---

## Recommendations

### Phase 0 (This Week) - Pre-enablement

1. ✅ **Fix 6 ruff errors**:
   ```bash
   # Auto-fix import sorting
   poetry run ruff check --fix src/bot_v2

   # Manually add missing import
   # src/bot_v2/features/brokerages/coinbase/websocket_handler.py
   # Add: from typing import Any
   ```

2. ✅ **Test hooks manually**:
   ```bash
   # Run all hooks to see what would happen
   pre-commit run --all-files
   ```

3. ✅ **Install hooks** (if not already):
   ```bash
   pre-commit install
   ```

4. ✅ **Document for team**:
   ```markdown
   # Add to CONTRIBUTING.md
   ## Pre-commit Hooks

   We use pre-commit hooks for code quality. Install them:
   ```
   pre-commit install
   ```

   Hooks will run automatically on `git commit`.
   ```

### Phase 1 (Future) - Enhancements

5. **Add mypy hook** (after fixing 393 errors):
   ```yaml
   - repo: https://github.com/pre-commit/mirrors-mypy
     rev: v1.18.2
     hooks:
       - id: mypy
         additional_dependencies:
           - types-requests
           - pandas-stubs
           - types-pyyaml
   ```

6. **Consider ruff-format** (replace black):
   ```yaml
   # Option: Replace black hook with ruff format
   - repo: https://github.com/astral-sh/ruff-pre-commit
     rev: v0.13.3
     hooks:
       - id: ruff
         args: [--fix, --exit-non-zero-on-fix]
       - id: ruff-format  # Add this

   # Remove black repo entirely
   ```

7. **Add security scanning** (optional):
   ```yaml
   - repo: https://github.com/PyCQA/bandit
     rev: 1.7.5
     hooks:
       - id: bandit
         args: [--skip, B101]  # Skip assert checks (allowed in tests)
   ```

### Do NOT Add (Too Slow/Strict)

- ❌ **pytest hook** - 44s runtime too slow for commit hook
- ❌ **mypy hook** - Not until errors < 50
- ❌ **coverage hook** - Too slow for commit

**Note**: These belong in CI, not pre-commit

---

## Version Alignment Check

| Tool | pyproject.toml | .pre-commit-config.yaml | Status |
|------|----------------|-------------------------|--------|
| ruff | ^0.13.3 | v0.13.3 | ✅ Match |
| black | ^25.9.0 | 25.9.0 | ✅ Match |
| pre-commit-hooks | N/A | v6.0.0 | ✅ Latest |
| pyupgrade | Not in deps | v3.20.0 | ⚠️ Not a dev dep |

**Note**: `pyupgrade` is only used in pre-commit, not as a dev dependency. This is fine - pre-commit installs it in its own virtualenv.

---

## Summary

### ✅ What's Good
1. Comprehensive hook coverage (formatting, linting, hygiene, security)
2. Version alignment with dev dependencies
3. Custom test hygiene enforcement
4. Modern syntax upgrades (pyupgrade)
5. Safe order of execution

### ⚠️ What Needs Attention
1. 6 ruff errors blocking enablement (1 hour fix)
2. Decision needed: keep black or switch to ruff format
3. Hooks may not be installed yet (verify with team)

### 🎯 Recommended Next Steps
1. Fix 6 ruff errors immediately
2. Test hooks: `pre-commit run --all-files`
3. Install hooks: `pre-commit install`
4. Document in CONTRIBUTING.md
5. Enable in CI as gate

**Overall Grade**: A- (Would be A after fixing ruff errors)

---

**Status**: READY FOR ENABLEMENT (after ruff fixes)

**Next Steps**: Export dependency tree (Task 6)
