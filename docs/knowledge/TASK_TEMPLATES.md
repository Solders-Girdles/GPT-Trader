# Task Templates for Agent Delegation

## Purpose
These templates ensure agents receive complete context for common tasks.
Copy, fill in the bracketed values, and delegate.

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

## 🔍 Analyze Code Quality

```
Task: Analyze [MODULE_NAME] for code quality issues

1. Read all files in: /Users/rj/PycharmProjects/GPT-Trader/src/bot/[MODULE_PATH]/
2. Check for these specific issues:
   - Unused imports (run: poetry run ruff check [PATH])
   - Missing type hints on public functions
   - Functions longer than 50 lines
   - Duplicate code patterns
   - Missing docstrings on classes
3. For each issue found, provide:
   - File path and line number
   - Issue description
   - Suggested fix
4. Return results as JSON: {"issues": [{"file": "", "line": 0, "issue": "", "fix": ""}]}
```

---

## ✅ Implement New Feature

```
Task: Implement [FEATURE_NAME]

CONTEXT:
- Current system state: Read .knowledge/PROJECT_STATE.json
- This feature should: [CLEAR_DESCRIPTION]
- It will be used by: [WHO_USES_IT]

IMPLEMENTATION:
1. First write a failing test:
   - Create: tests/unit/[module]/test_[feature].py
   - Test should verify: [EXPECTED_BEHAVIOR]
   - Run it to confirm it fails: poetry run pytest [TEST_PATH] -xvs

2. Implement the feature:
   - Add to: src/bot/[module]/[file].py
   - Follow patterns in: [SIMILAR_EXISTING_FILE]
   - Use these dependencies only: [ALLOWED_IMPORTS]

3. Make the test pass:
   - Run: poetry run pytest [TEST_PATH] -xvs
   - Must see: "1 passed"

4. Update .knowledge/PROJECT_STATE.json:
   - Add feature to components.[module].features list
   - Set verified: false (until full verification)

5. Return: Test output showing it passes, or error details
```

---

## 🔧 Fix Import Errors

```
Task: Fix import errors in [MODULE_NAME]

1. Identify all import errors:
   - Run: python -c "import bot.[module]"
   - Capture the full traceback

2. For each import error:
   - Read the file with the error: [FILE_PATH]
   - Check if the imported module exists: ls src/bot/[expected_path]/
   - Fix by either:
     a) Correcting the import path
     b) Creating a missing __init__.py
     c) Installing missing dependency: poetry add [package]

3. Verify all imports work:
   - Run: python -c "from bot.[module] import *"
   - Should complete without errors

4. Run module tests to ensure nothing broke:
   - poetry run pytest tests/unit/[module]/ -x

5. Return: "FIXED: [list of changes]" or "BLOCKED: [what's still broken]"
```

---

## 📊 Verify Component Status

```
Task: Verify [COMPONENT_NAME] status and update .knowledge/PROJECT_STATE.json

1. Read current status: 
   - cat .knowledge/PROJECT_STATE.json | grep -A 10 '"[component]"'

2. Run component test:
   - Command: [TEST_COMMAND_FROM_PROJECT_STATE]
   - Timeout: 30 seconds
   - Capture output

3. Analyze results:
   - If test passes: Status = "working"
   - If test fails: Status = "failed"
   - If no test exists: Status = "unknown"

4. Update .knowledge/PROJECT_STATE.json:
   - Set components.[component].status = [new_status]
   - Set components.[component].verified = true
   - Set components.[component].last_verified = [current_timestamp]
   - Add any new known_issues found

5. Return: "[COMPONENT]: [old_status] → [new_status]"
```

---

## 🧹 Clean Up Technical Debt

```
Task: Clean up [SPECIFIC_DEBT_TYPE] in [MODULE]

DEBT TO CLEAN:
- [X] Remove commented code
- [X] Delete unused imports  
- [X] Consolidate duplicate functions
- [X] Fix TODO comments
- [X] Remove debug print statements

PROCESS:
1. Scan for issues:
   - poetry run ruff check src/bot/[module]/ --select E,W,F
   - grep -r "TODO\|FIXME\|XXX" src/bot/[module]/
   - grep -r "print(" src/bot/[module]/

2. For each file with issues:
   - Read the file: [FILE_PATH]
   - Make specific fixes (don't change functionality)
   - Verify syntax: python -m py_compile [FILE_PATH]

3. Run tests to ensure nothing broke:
   - poetry run pytest tests/unit/[module]/ -x
   - Must maintain same pass/fail count

4. Return summary:
   - Files cleaned: [count]
   - Issues fixed: [list]
   - Tests still passing: [yes/no]
```

---

## 🚀 Performance Optimization

```
Task: Optimize performance of [FUNCTION/MODULE]

PERFORMANCE ISSUE:
- Current behavior: [WHAT'S SLOW]
- Expected: [PERFORMANCE_TARGET]
- Measured with: [HOW_TO_MEASURE]

OPTIMIZATION:
1. Profile current performance:
   - Run: python -m cProfile -s cumtime [SCRIPT_THAT_USES_IT]
   - Identify top 3 bottlenecks

2. Read implementation: src/bot/[module]/[file].py

3. Apply optimizations:
   - Replace .iterrows() with vectorized operations
   - Cache expensive calculations
   - Use numpy/pandas built-ins instead of loops
   - Avoid repeated file I/O

4. Verify improvements:
   - Run same profile command
   - Compare timings
   - Run tests: poetry run pytest tests/unit/[module]/ -x

5. Return:
   - Before: [time]
   - After: [time]  
   - Speedup: [X]x
   - Changes made: [list]
```

---

## 📝 Write Missing Tests

```
Task: Write tests for [MODULE/FUNCTION]

1. Read the implementation:
   - File: src/bot/[module]/[file].py
   - Focus on: [FUNCTION_NAME] function

2. Create test file:
   - Path: tests/unit/[module]/test_[function].py
   - Import: from bot.[module].[file] import [function]

3. Write test cases for:
   - Happy path (normal inputs)
   - Edge cases (empty, None, boundaries)
   - Error cases (invalid inputs)
   - Use existing fixtures from tests/fixtures/

4. Run tests:
   - poetry run pytest [TEST_FILE] -xvs
   - All must pass

5. Check coverage:
   - poetry run pytest [TEST_FILE] --cov=bot.[module].[file]
   - Should cover > 80% of the function

6. Return: Test file path and coverage percentage
```

---

## Usage Instructions

1. **Choose the right template** for your task
2. **Fill in ALL bracketed placeholders** with specific values
3. **Include full file paths** (/Users/rj/PycharmProjects/GPT-Trader/...)
4. **Specify exact commands** to run
5. **Define clear success criteria**

## Example: Using the Fix Test Template

```
Task: Fix the failing test test_calculate_signals

1. Read the test file: /Users/rj/PycharmProjects/GPT-Trader/tests/unit/strategy/test_demo_ma.py
2. Run the test to see the error: poetry run pytest tests/unit/strategy/test_demo_ma.py::test_calculate_signals -xvs
3. The error is: "AttributeError: 'NoneType' object has no attribute 'empty'"
4. Read the implementation file: /Users/rj/PycharmProjects/GPT-Trader/src/bot/strategy/demo_ma.py
5. Fix the issue by: Returning an empty DataFrame instead of None when no data
6. Verify the fix: poetry run pytest tests/unit/strategy/test_demo_ma.py::test_calculate_signals -xvs
7. Return: "FIXED: [what you changed]" or "FAILED: [error message]"
```

This provides complete context for the agent to succeed.