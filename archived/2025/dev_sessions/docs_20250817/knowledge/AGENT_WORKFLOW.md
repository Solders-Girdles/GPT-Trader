# Optimal Claude Code Agent Workflow (V2 Edition)

## Core Reality
- **Agents are stateless**: Each invocation starts fresh
- **Agents are isolated**: No shared context or communication
- **Main agent coordinates**: You manage all context passing
- **V2 uses vertical slices**: Each feature is completely self-contained

## The RIGHT Way to Use Agents in V2

### 1. Self-Contained V2 Slice Tasks
```python
# ✅ GOOD: Everything the agent needs for V2 slice work
"""
Task: Fix the backtest slice test failure
1. Read src/bot_v2/features/backtest/backtest.py
2. The error is 'KeyError: returns' on line 89
3. Fix by adding: returns = (data['close'] / data['close'].shift(1) - 1).fillna(0)
4. Test with: poetry run python src/bot_v2/test_backtest.py
5. Return: "FIXED" if test passes, or the error message
"""

# ❌ BAD: Assuming context
"Continue fixing the backtest issues from before"
```

### 2. Explicit V2 File References
```python
# ✅ GOOD: Tell agent exactly what to read/write in V2 structure
"""
Read these files:
- /Users/rj/PycharmProjects/GPT-Trader/.knowledge/PROJECT_STATE.json
- /Users/rj/PycharmProjects/GPT-Trader/src/bot_v2/features/ml_strategy/ml_strategy.py

Update .knowledge/PROJECT_STATE.json v2_slices.ml_strategy.status to "working" if tests pass.
"""

# ❌ BAD: Vague references
"Check the state files and update as needed"
```

### 3. V2 Slice Isolation Verification
```python
# ✅ GOOD: Include V2 isolation verification
"""
After making changes to the analyze slice:
1. Run: poetry run python src/bot_v2/test_analyze.py
2. Verify no cross-slice imports: grep -r "from bot_v2.features" src/bot_v2/features/analyze/
3. Check independent loading: python -c "from src.bot_v2.features.analyze import *"
4. Return: Test output and isolation verification
"""

# ❌ BAD: Assuming agent will verify isolation
"Fix the analyze slice and make sure it works"
```

## V2-Specific Task Patterns

### Pattern 1: V2 Slice Investigation → Action
```python
# First agent: Investigate slice dependencies
"Find all imports in src/bot_v2/features/market_regime/ that violate isolation.
Look for imports from other slices. Return file paths and violating imports."

# Main agent processes response, then:
# Second agent: Fix isolation violations
"In files [list from first agent], replace cross-slice imports with local implementations.
Duplicate needed code in the local slice rather than importing.
Test with: poetry run python src/bot_v2/test_market_regime.py"
```

### Pattern 2: V2 Test → Fix → Verify
```python
# Single agent, complete V2 workflow
"1. Run: poetry run python src/bot_v2/test_position_sizing.py
2. If it fails, fix the issue in src/bot_v2/features/position_sizing/
3. Ensure slice isolation maintained (no external dependencies)
4. Run the test again
5. Return: test output and isolation status"
```

### Pattern 3: V2 Parallel Slice Analysis
```python
# Agent 1 (parallel)
"Analyze src/bot_v2/features/backtest/ for code quality and isolation compliance"

# Agent 2 (parallel) 
"Analyze src/bot_v2/features/paper_trade/ for code quality and isolation compliance"

# Main agent synthesizes both responses for overall V2 health
```

### Pattern 4: V2 New Slice Creation
```python
# Complete V2 slice creation task
"Create new feature slice 'risk_monitor' following V2 principles:
1. Create src/bot_v2/features/risk_monitor/ directory
2. Implement risk_monitor.py with complete self-containment
3. Create types.py with all needed types (no external imports)
4. Write src/bot_v2/test_risk_monitor.py
5. Verify isolation: no imports from other slices
6. Return: test results and file structure created"
```

## What NOT to Do in V2

### ❌ Don't Create V1-Style Shared Components
```python
# These break V2 isolation principles:
- src/bot_v2/shared/
- src/bot_v2/common/
- Cross-slice utility imports
- Shared configuration objects
```

### ❌ Don't Expect V1 Paths to Work
```python
# Agent won't find these (archived/deprecated):
- src/bot_v2/features/ (old V1 system)
- examples/ (archived to archived/old_v1_examples_20250817/)
- data/optimization/ (archived to archived/old_optimization_data_20250817/)
- logs/metrics/ (archived to archived/metrics/)
```

### ❌ Don't Violate Slice Isolation
```python
# Bad: Breaking V2 isolation
"Make the backtest slice use the risk calculations from the paper_trade slice"

# Good: Maintaining V2 isolation  
"Duplicate the needed risk calculations locally in the backtest slice"
```

## The V2 Ultraclean Workflow

1. **Main agent** reads .knowledge/PROJECT_STATE.json for V2 slice status
2. **Main agent** identifies which V2 slice needs work
3. **Main agent** delegates slice-specific task with ALL context
4. **Sub-agent** executes following V2 isolation principles
5. **Main agent** verifies slice independence maintained
6. **Main agent** updates .knowledge/PROJECT_STATE.json v2_slices section
7. **Main agent** runs slice-specific verification tests

## V2 Repository Structure Awareness

Current ultraclean structure (post-230M cleanup):
```
src/bot_v2/                 # ONLY active system
├── features/               # 9 vertical slices
│   ├── backtest/          # ~500 tokens each
│   ├── paper_trade/       # Complete isolation
│   ├── analyze/           # No cross-dependencies
│   ├── optimize/          # Self-contained
│   ├── live_trade/        # Local implementations
│   ├── monitor/           # Local types
│   ├── data/              # Local everything
│   ├── ml_strategy/       # Week 1-2 complete
│   └── market_regime/     # Week 3 complete
└── test_*.py              # Integration tests

archived/                   # 7.5M historical reference
tests/                     # 1.6M V2-focused tests  
scripts/                   # 1.1M organized by category
docs/                      # 1.1M consolidated structure
```

## Quick V2 Verification Commands

For agents to verify V2 compliance:
```bash
# Test specific slice
poetry run python src/bot_v2/test_[slice].py

# Check slice isolation  
grep -r "from bot_v2.features" src/bot_v2/features/[slice]/

# Verify independent loading
python -c "from src.bot_v2.features.[slice] import *"

# Test all slices
poetry run python src/bot_v2/test_all_slices.py
```

No orchestration. No shared state. No V1 legacy.
Just clear, self-contained V2 slice tasks with explicit isolation verification.