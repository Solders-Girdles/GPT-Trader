# Complete Agent Delegation Guide

## 🔧 Agent Tool Access Reference

### Read-Only Agents (Analysis/Planning)
**Tools**: `Read, Grep, Glob, LS`
- `planner` - Creates implementation plans
- `code-archaeologist` - Analyzes legacy code
- `code-reviewer` - Reviews code quality
- `trading-strategy-consultant` - Validates trading logic
- `repo-structure-guardian` - Enforces standards

### Write-Capable Agents (Implementation)
**Tools**: `Read, Write, Edit, MultiEdit, Glob, LS`
- `frontend-developer` - UI implementation
- `backend-developer` - Server-side code
- `tailwind-frontend-expert` - CSS styling
- `documentation-specialist` - Creates docs

### Full-Access Agents (Testing/Debugging)
**Tools**: `Read, Write, Edit, Bash, Grep, Glob, LS`
- `tech-lead-orchestrator` - Complex analysis & implementation
- `performance-optimizer` - System optimization
- `test-runner` - Runs and analyzes tests
- `debugger` - Fixes failing tests

### Special Agents
- `gemini-gpt-hybrid` - Uses external AI for analysis (Tools: Read, Edit, Bash)
- `gemini-gpt-hybrid-hard` - Aggressive automation (Tools: Bash only)

---

## ✅ Pre-Delegation Checklist

Before delegating ANY task, ensure you have:

### 1. Context Preparation
- [ ] **Current state known**: Read .knowledge/PROJECT_STATE.json
- [ ] **Files identified**: Know exact paths to relevant files
- [ ] **Error captured**: Have specific error messages if fixing bugs
- [ ] **Dependencies clear**: Know what packages/modules are available

### 2. Task Definition
- [ ] **Single responsibility**: One clear goal, not multiple
- [ ] **Specific files**: Full paths provided (/Users/rj/PycharmProjects/GPT-Trader/...)
- [ ] **Clear success criteria**: How to know when done
- [ ] **Verification command**: Exact command to test success

### 3. Agent Selection
- [ ] **Right tools**: Agent has necessary tool access
- [ ] **Right expertise**: Agent description matches task
- [ ] **Not overkill**: Don't use tech-lead for simple tasks
- [ ] **Not underpowered**: Don't use read-only agent for fixes

### 4. Instruction Completeness
- [ ] **No assumed context**: Agent won't know previous conversation
- [ ] **No external references**: Include all needed info in prompt
- [ ] **Specific return format**: Tell agent exactly what to return
- [ ] **Fallback instructions**: What to do if blocked

---

## 📝 Delegation Template

```markdown
@agent-[type]: [One-line task summary]

CONTEXT:
- System state: [From .knowledge/PROJECT_STATE.json]
- Working directory: /Users/rj/PycharmProjects/GPT-Trader
- Relevant files: [List with full paths]

TASK:
1. [Specific step 1 with exact command/file]
2. [Specific step 2 with exact command/file]
3. [Verification step with command]

SUCCESS CRITERIA:
- [ ] [Measurable outcome 1]
- [ ] [Measurable outcome 2]

RETURN FORMAT:
- If successful: "SUCCESS: [what was done]"
- If failed: "FAILED: [specific error and what was tried]"

VERIFICATION:
Run: [exact command to verify success]
Expected output: [what should appear]
```

---

## 🚫 Common Delegation Mistakes

### ❌ Too Vague
```
"Fix the broken tests"
```
**Why it fails**: Which tests? What's broken? Where are they?

### ❌ Missing Context
```
"Continue from before"
```
**Why it fails**: Agent has no memory of "before"

### ❌ Wrong Agent Type
```
"@agent-planner: Fix the import error"
```
**Why it fails**: Planner can only read, not edit

### ❌ Multiple Responsibilities
```
"Fix tests, update docs, and optimize performance"
```
**Why it fails**: Too broad, unclear priorities

### ❌ No Verification
```
"Make it work"
```
**Why it fails**: No way to confirm success

---

## ✅ Good Delegation Examples

### Example 1: Fix Specific Test
```
@agent-debugger: Fix failing test_calculate_signals test

CONTEXT:
- Test file: /Users/rj/PycharmProjects/GPT-Trader/tests/unit/strategy/test_demo_ma.py
- Implementation: /Users/rj/PycharmProjects/GPT-Trader/src/bot/strategy/demo_ma.py
- Error: "AttributeError: 'NoneType' object has no attribute 'empty'"

TASK:
1. Run test: poetry run pytest tests/unit/strategy/test_demo_ma.py::test_calculate_signals -xvs
2. Read implementation at line causing error
3. Fix: Return empty DataFrame instead of None when no data
4. Verify: poetry run pytest tests/unit/strategy/test_demo_ma.py::test_calculate_signals -xvs

RETURN:
- "FIXED: Changed line X to return pd.DataFrame()" or error details
```

### Example 2: Code Analysis
```
@agent-code-archaeologist: Analyze risk management module complexity

CONTEXT:
- Module path: /Users/rj/PycharmProjects/GPT-Trader/src/bot/risk/
- Focus on: integration.py, config.py, dashboard.py
- Looking for: Duplicate code, overly complex functions, dead code

TASK:
1. Read all files in src/bot/risk/
2. Identify functions > 50 lines
3. Find duplicate logic patterns
4. Check for unused imports/functions

RETURN:
JSON list of issues with file, line, description, suggested fix
```

### Example 3: Implementation Task
```
@agent-backend-developer: Add timeout parameter to DataPipeline

CONTEXT:
- File: /Users/rj/PycharmProjects/GPT-Trader/src/bot/dataflow/pipeline.py
- Current: get_data() method has no timeout
- Need: Add configurable timeout with 30s default

TASK:
1. Read pipeline.py to understand current structure
2. Add timeout parameter to get_data(timeout: int = 30)
3. Implement timeout logic using concurrent.futures
4. Test: python -c "from bot.dataflow.pipeline import DataPipeline; p = DataPipeline(); p.get_data('AAPL', '2024-01-01', '2024-01-31', timeout=5)"

RETURN:
"IMPLEMENTED: Added timeout to lines X-Y" or blocker details
```

---

## 📊 Quick Reference: Task → Agent Mapping

| Task Type | Best Agent | Tools Needed |
|-----------|-----------|--------------|
| Fix failing test | `debugger` | Read, Edit, Bash |
| Analyze code quality | `code-reviewer` | Read, Grep |
| Implement feature | `backend-developer` | Read, Write, Edit |
| Optimize performance | `performance-optimizer` | Read, Edit, Bash |
| Create documentation | `documentation-specialist` | Read, Write |
| Plan implementation | `planner` | Read, Grep |
| Fix UI/styling | `frontend-developer` | Read, Edit, Write |
| Security review | `code-reviewer` | Read, Grep |
| Complex orchestration | `tech-lead-orchestrator` | All tools |

---

## 🎯 Key Principles

1. **Agents are goldfish**: No memory between calls
2. **Context is everything**: Include ALL needed information
3. **Verification is mandatory**: Always include test commands
4. **One task per agent**: Don't overload with multiple goals
5. **Full paths always**: Never use relative paths
6. **Specific returns**: Tell agent exact format needed

---

## 🔄 Post-Delegation Workflow

After agent returns:

1. **Parse Response**: Extract success/failure status
2. **Verify Claims**: Run verification commands yourself
3. **Update State**: Modify .knowledge/PROJECT_STATE.json if needed
4. **Follow Up**: Create new specific task if needed

Never trust agent claims without verification!