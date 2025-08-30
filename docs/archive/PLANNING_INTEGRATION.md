# Planning Integration Guide

## The Complete Planning Stack

### 1. Strategic Level: ROADMAP.json
- **Scope**: Weeks/Months
- **Owns**: Phases, milestones, success criteria
- **Updated**: When milestones complete
- **Example**: "Phase 1: Get system to 80% functional"

### 2. Tactical Level: Main Agent
- **Scope**: Current session
- **Owns**: Interpreting roadmap, delegating tasks
- **Updated**: During session
- **Example**: "Fix data pipeline by addressing import errors"

### 3. Operational Level: TodoWrite
- **Scope**: Individual tasks
- **Owns**: Task tracking, completion status
- **Updated**: As work progresses
- **Example**: "Fix import in test_pipeline.py"

## Workflow Example

```python
# Start of session - Agent loads strategic context
from scripts.update_roadmap import get_current_tasks

# 1. Get strategic direction
current = get_current_tasks()
print(f"Working on: {current['milestone']}")
# Output: "Working on: Fix data pipeline"

# 2. Create operational todos
todos = [
    {"id": "1", "content": task, "status": "pending"}
    for task in current['tasks']
]
TodoWrite(todos)

# 3. Work through todos
for todo in todos:
    # Do work...
    todo["status"] = "completed"
    TodoWrite(todos)

# 4. Update strategic plan when done
from scripts.update_roadmap import update_milestone_status
update_milestone_status("m1", "completed")
```

## Benefits of This Approach

### ✅ Persistent Memory
```
Without ROADMAP.json:
- Agent 1: "I'll fix the ML pipeline"
- Agent 2: "What ML pipeline? Let me build a dashboard"
- Result: Chaos

With ROADMAP.json:
- Agent 1: "Roadmap says fix data pipeline first"
- Agent 2: "Roadmap says data pipeline still needs work"
- Result: Coordinated progress
```

### ✅ Clear Priorities
```json
{
  "current_focus": {
    "milestone_id": "m1",
    "milestone_title": "Fix data pipeline",
    "next_tasks": [
      "Run diagnostics on data_pipeline component",
      "Fix import errors in tests/unit/dataflow/",
      "Update PROJECT_STATE.json when tests pass"
    ]
  }
}
```

### ✅ Success Tracking
- Not just "is it working?" but "does it meet our goals?"
- Links component fixes to business objectives
- Provides context for WHY we're doing something

## Quick Commands

```bash
# Check strategic status
poetry run python scripts/update_roadmap.py status

# Get current tasks for todos
poetry run python scripts/update_roadmap.py tasks

# Mark milestone complete
poetry run python scripts/update_roadmap.py complete m1

# Add a blocker
poetry run python scripts/update_roadmap.py block "Missing API keys"

# Sync with PROJECT_STATE.json
poetry run python scripts/update_roadmap.py sync
```

## Integration Rules

1. **Every session starts with roadmap check**
   ```python
   roadmap = json.load(open('ROADMAP.json'))
   current_milestone = roadmap['current_focus']['milestone_title']
   ```

2. **Todos come from roadmap tasks**
   - Don't create random todos
   - Link todos to current milestone

3. **Update both directions**
   - Milestone complete → Update PROJECT_STATE.json
   - Component fixed → Update ROADMAP.json

4. **Document blockers**
   - Can't complete milestone? Add blocker
   - Helps next agent understand why work stopped

## The Planning Hierarchy in Action

```
STRATEGIC (ROADMAP.json)
"Build ML-driven trading system"
    ↓
"Phase 1: Foundation repair"
    ↓
"Milestone: Fix data pipeline"
    ↓
TACTICAL (Main Agent)
"I need to fix YFinance integration"
    ↓
"Delegate: Fix import errors in dataflow"
    ↓
OPERATIONAL (TodoWrite)
[✓] Run diagnostic on imports
[✓] Fix missing __init__.py
[✓] Update test fixtures
[✓] Run tests to verify
```

## Why This Works

1. **Agents always know what to work on** (current_focus)
2. **Work contributes to larger goals** (phases)
3. **Progress persists between sessions** (JSON files)
4. **Clear handoff between agents** (documented plan)
5. **No duplicate or wasted effort** (single source of truth)

The roadmap provides the "why" and "what next" that agents desperately need to work effectively across sessions.