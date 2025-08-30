# Optimal Claude Code Agent Workflow

## Core Reality
- **Agents are stateless**: Each invocation starts fresh
- **Agents are isolated**: No shared context or communication
- **Main agent coordinates**: You manage all context passing

## The RIGHT Way to Use Agents

### 1. Self-Contained Tasks
```python
# ✅ GOOD: Everything the agent needs in one prompt
"""
Task: Fix the data pipeline test failure
1. Read src/bot/dataflow/pipeline.py
2. The error is 'KeyError: symbol' on line 145
3. Fix by adding: if 'symbol' not in params: params['symbol'] = 'AAPL'
4. Test with: poetry run pytest tests/unit/dataflow/test_pipeline.py::test_get_data
5. Return: "FIXED" if test passes, or the error message
"""

# ❌ BAD: Assuming context
"Continue fixing the pipeline issues from before"
```

### 2. Explicit File References
```python
# ✅ GOOD: Tell agent exactly what to read/write
"""
Read these files:
- /Users/rj/PycharmProjects/GPT-Trader/.knowledge/PROJECT_STATE.json
- /Users/rj/PycharmProjects/GPT-Trader/src/bot/strategy/demo_ma.py

Update .knowledge/PROJECT_STATE.json components.strategies.status to "working" if tests pass.
"""

# ❌ BAD: Vague references
"Check the state files and update as needed"
```

### 3. Verification in Delegation
```python
# ✅ GOOD: Include verification steps
"""
After making changes:
1. Run: poetry run pytest tests/unit/strategy/ -v
2. If tests pass, run: poetry run python scripts/verify_capabilities.py
3. Return the test output
"""

# ❌ BAD: Assuming agent will verify
"Fix the strategy and make sure it works"
```

## Task Patterns That Work

### Pattern 1: Investigation → Action
```python
# First agent: Investigate
"Find all files importing 'bot.ml.models'. 
Return a list of file paths and line numbers."

# Main agent processes response, then:
# Second agent: Fix
"In files [list from first agent], change 'bot.ml.models' to 'bot.ml.models.strategy_selector'.
Test each change with: python -c 'import [module]'"
```

### Pattern 2: Test → Fix → Verify
```python
# Single agent, complete workflow
"1. Run: poetry run pytest tests/unit/risk/test_integration.py
2. If it fails with import error, fix the import in src/bot/risk/integration.py
3. Run the test again
4. Return: test output and whether it passes"
```

### Pattern 3: Parallel Analysis
```python
# Agent 1 (parallel)
"Analyze src/bot/strategy/ for duplicate code patterns"

# Agent 2 (parallel) 
"Count test coverage in tests/unit/strategy/"

# Main agent synthesizes both responses
```

## What NOT to Do

### ❌ Don't Create Orchestration Files
```python
# These don't help agents:
- orchestration_context.json
- .claude_state/*
- agent_handoff_report.md
```

### ❌ Don't Expect Memory
```python
# Agent won't remember:
- Previous tasks
- Other agents' work  
- Patterns from AGENT_TRIGGERS.md
- Trust scores or history
```

### ❌ Don't Chain Dependencies
```python
# Bad: Agent 2 depends on Agent 1's internal state
"Agent 1: Set up the context"
"Agent 2: Use the context from Agent 1"  # Won't work
```

## The Simplified Workflow

1. **Main agent** reads .knowledge/PROJECT_STATE.json
2. **Main agent** decides what needs fixing
3. **Main agent** delegates specific task with ALL context
4. **Sub-agent** executes and returns result
5. **Main agent** updates .knowledge/PROJECT_STATE.json
6. **Main agent** verifies with scripts/verify_capabilities.py

No orchestration. No state files. No complex patterns.
Just clear, self-contained tasks with explicit verification.