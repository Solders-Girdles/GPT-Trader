# Task Templates for V2 Agent Delegation

## Purpose
These templates ensure agents receive complete context for common V2 tasks.
Copy, fill in the bracketed values, and delegate.

**V2 Architecture Context**: GPT-Trader now uses vertical slice architecture in `src/bot_v2/` with complete isolation between features.

---

## 🐛 Fix Test Failure

```
Task: Fix the failing test [TEST_NAME]

1. Read the test file: [FULL_PATH_TO_TEST]
2. Run the test to see the error: poetry run pytest [TEST_PATH]::[TEST_NAME] -xvs
3. The error is: [PASTE_ERROR_MESSAGE]
4. Read the implementation file: [FULL_PATH_TO_IMPLEMENTATION]
5. Fix the issue by: [SPECIFIC_FIX_NEEDED]
6. Verify the fix: poetry run pytest [TEST_PATH]::[TEST_NAME] -xvs
7. Return: "FIXED: [what you changed]" or "FAILED: [error message]"
```

---

## 🔍 Analyze V2 Slice Quality

```
Task: Analyze [SLICE_NAME] vertical slice for code quality issues

1. Read all files in: /Users/rj/PycharmProjects/GPT-Trader/src/bot_v2/features/[SLICE_NAME]/
2. Check slice isolation principles:
   - No imports from other slices (only local implementations)
   - Complete self-containment (~500 tokens to load)
   - Local types.py with all needed types
   - README.md with usage examples
3. Check for these quality issues:
   - Unused imports (run: poetry run ruff check [PATH])
   - Missing type hints on public functions
   - Functions longer than 50 lines
   - Missing docstrings on classes
4. For each issue found, provide:
   - File path and line number
   - Issue description
   - Suggested fix
5. Return results as JSON: {"issues": [{"file": "", "line": 0, "issue": "", "fix": ""}]}
```

---

## ✅ Implement New V2 Feature Slice

```
Task: Implement new feature slice [FEATURE_NAME]

CONTEXT:
- V2 uses complete vertical slice isolation 
- Each slice is self-contained in src/bot_v2/features/[feature]/
- No dependencies between slices allowed
- Current system state: Read .knowledge/PROJECT_STATE.json

IMPLEMENTATION:
1. Create the slice directory structure:
   - mkdir src/bot_v2/features/[feature]/
   - Create: [feature].py (main implementation)
   - Create: types.py (all needed types)
   - Create: README.md (usage examples)
   - Create: __init__.py (clean exports)

2. First write a failing test:
   - Create: src/bot_v2/test_[feature].py 
   - Test should verify: [EXPECTED_BEHAVIOR]
   - Run it to confirm it fails: poetry run python src/bot_v2/test_[feature].py

3. Implement the feature:
   - Follow complete isolation principle
   - Duplicate common code rather than sharing
   - Include all needed types in local types.py
   - No imports from other slices

4. Make the test pass:
   - Run: poetry run python src/bot_v2/test_[feature].py
   - Must see successful output

5. Update .knowledge/PROJECT_STATE.json:
   - Add feature to v2_slices.[feature] entry
   - Set verified: false (until full verification)

6. Return: Test output showing it passes, or error details
```

---

## 🔧 Fix V2 Slice Isolation Violations

```
Task: Fix isolation violations in [SLICE_NAME]

1. Scan for isolation violations:
   - Run: grep -r "from bot_v2.features" src/bot_v2/features/[slice]/
   - Look for cross-slice imports (should be none)
   - Check for shared dependencies outside local scope

2. For each violation found:
   - Read the file with the violation: [FILE_PATH]
   - Identify what shared code it needs
   - Fix by duplicating the needed code locally
   - Update local types.py with any needed types

3. Verify isolation is restored:
   - Run: poetry run python src/bot_v2/test_[slice].py
   - Should work without any external slice dependencies

4. Check slice can be loaded independently:
   - Run: poetry run python -c "from src.bot_v2.features.[slice] import *"
   - Should complete without errors

5. Return: "FIXED: [list of changes]" or "BLOCKED: [what's still broken]"
```

---

## 📊 Verify V2 Slice Status

```
Task: Verify [SLICE_NAME] status and update .knowledge/PROJECT_STATE.json

1. Read current status: 
   - cat .knowledge/PROJECT_STATE.json | grep -A 10 '"[slice]"'

2. Run slice test:
   - Command: poetry run python src/bot_v2/test_[slice].py
   - Timeout: 30 seconds
   - Capture output

3. Check slice isolation:
   - Verify no cross-slice imports
   - Confirm self-contained operation
   - Test independent loading

4. Analyze results:
   - If test passes and isolated: Status = "working"
   - If test fails: Status = "failed" 
   - If isolation violated: Status = "broken_isolation"

5. Update .knowledge/PROJECT_STATE.json:
   - Set v2_slices.[slice].status = [new_status]
   - Set v2_slices.[slice].verified = true
   - Set v2_slices.[slice].last_verified = [current_timestamp]
   - Add any isolation_violations found

6. Return: "[SLICE]: [old_status] → [new_status]"
```

---

## 🧹 Clean Up V2 Repository Structure

```
Task: Clean up [AREA] following V2 ultraclean principles

AREAS TO CLEAN:
- [X] Remove old V1 references in [directory]
- [X] Archive unused demo/test files
- [X] Consolidate duplicate configurations
- [X] Remove empty directories
- [X] Clean up backup files

PROCESS:
1. Scan for cleanup opportunities:
   - Find backup files: find . -name "*.bak" -o -name "*backup*"
   - Find empty dirs: find . -type d -empty -not -path "./.git*"
   - Find old V1 references: grep -r "src/bot_v2/features/" . --exclude-dir=archived

2. For each cleanup area:
   - Archive rather than delete (maintain history)
   - Move to archived/[category]_[date]/
   - Update any references in documentation

3. Verify no functionality broken:
   - Run: poetry run python src/bot_v2/test_all_slices.py
   - All V2 slices should still work

4. Return summary:
   - Files archived: [count]
   - Space freed: [amount]
   - V2 functionality verified: [yes/no]
```

---

## 🚀 V2 Performance Optimization

```
Task: Optimize performance of [SLICE/FUNCTION] in V2 system

PERFORMANCE ISSUE:
- Current behavior: [WHAT'S_SLOW]
- Expected: [PERFORMANCE_TARGET]
- Measured with: [HOW_TO_MEASURE]

OPTIMIZATION:
1. Profile current performance:
   - Run: poetry run python -c "import time; start=time.time(); from src.bot_v2.features.[slice] import [function]; [function]([args]); print(f'Time: {time.time()-start:.3f}s')"
   - Identify bottlenecks in the slice

2. Read slice implementation: src/bot_v2/features/[slice]/[file].py

3. Apply V2-specific optimizations:
   - Leverage slice isolation for caching
   - Optimize local data structures
   - Use efficient pandas/numpy operations
   - Cache expensive calculations within slice

4. Verify improvements:
   - Run same performance test
   - Ensure slice isolation maintained
   - Run: poetry run python src/bot_v2/test_[slice].py

5. Return:
   - Before: [time]
   - After: [time]
   - Speedup: [X]x
   - Isolation maintained: [yes/no]
```

---

## 📝 Write V2 Slice Tests

```
Task: Write comprehensive tests for [SLICE_NAME]

1. Read the slice implementation:
   - Directory: src/bot_v2/features/[slice]/
   - Main file: [slice].py
   - Types: types.py

2. Create test file:
   - Path: src/bot_v2/test_[slice].py
   - Import: from src.bot_v2.features.[slice] import [main_functions]

3. Write test cases for:
   - Happy path (normal slice operation)
   - Edge cases (empty data, None inputs)
   - Error cases (invalid inputs)
   - Isolation verification (no external dependencies)

4. Run tests:
   - poetry run python src/bot_v2/test_[slice].py
   - Should see all tests pass

5. Verify slice can work independently:
   - Test loading slice in isolation
   - Verify no cross-slice dependencies
   - Check all types are locally defined

6. Return: Test results and isolation verification
```

---

## Usage Instructions for V2

1. **Choose the right template** for your V2 task
2. **Fill in ALL bracketed placeholders** with specific values
3. **Use V2 paths** (src/bot_v2/features/[slice]/)
4. **Respect slice isolation** - no cross-slice dependencies
5. **Test slice independence** after changes

## Example: Using the V2 Fix Test Template

```
Task: Fix the failing test test_backtest_execution

1. Read the test file: /Users/rj/PycharmProjects/GPT-Trader/src/bot_v2/test_backtest.py
2. Run the test to see the error: poetry run python src/bot_v2/test_backtest.py
3. The error is: "KeyError: 'returns' in calculate_metrics"
4. Read the implementation file: /Users/rj/PycharmProjects/GPT-Trader/src/bot_v2/features/backtest/backtest.py
5. Fix the issue by: Adding returns calculation before metrics calculation
6. Verify the fix: poetry run python src/bot_v2/test_backtest.py
7. Return: "FIXED: Added returns calculation in backtest metrics"
```

This provides complete V2 context for the agent to succeed while maintaining slice isolation.