# Complete V2 Agent Delegation Guide

## 🔧 Agent Tool Access Reference

### Read-Only Agents (Analysis/Planning)
**Tools**: `Read, Grep, Glob, LS`
- `planner` - Creates V2 slice implementation plans
- `code-archaeologist` - Analyzes V2 slice complexity
- `code-reviewer` - Reviews V2 slice quality
- `trading-strategy-consultant` - Validates trading logic
- `repo-structure-guardian` - Enforces V2 isolation

### Write-Capable Agents (Implementation)
**Tools**: `Read, Write, Edit, MultiEdit, Glob, LS`
- `frontend-developer` - UI implementation (if needed)
- `backend-developer` - V2 slice implementation
- `tailwind-frontend-expert` - CSS styling (if needed)
- `documentation-specialist` - Creates slice docs

### Full-Access Agents (Testing/Debugging)
**Tools**: `Read, Write, Edit, Bash, Grep, Glob, LS`
- `tech-lead-orchestrator` - Complex V2 analysis & implementation
- `performance-optimizer` - V2 slice optimization
- `test-runner` - Runs and analyzes V2 tests
- `debugger` - Fixes failing V2 tests

### Special Agents
- `gemini-gpt-hybrid` - Uses external AI for analysis (Tools: Read, Edit, Bash)
- `gemini-gpt-hybrid-hard` - Aggressive automation (Tools: Bash only)

---

## ✅ V2 Pre-Delegation Checklist

Before delegating ANY V2 task, ensure you have:

### 1. V2 Context Preparation
- [ ] **Current state known**: Read .knowledge/PROJECT_STATE.json v2_slices section
- [ ] **Slice identified**: Know which feature slice needs work
- [ ] **Files identified**: Know exact V2 paths (src/bot_v2/features/[slice]/)
- [ ] **Error captured**: Have specific error messages if fixing bugs
- [ ] **Isolation verified**: Understand current slice dependencies

### 2. V2 Task Definition
- [ ] **Single slice focus**: One slice at a time
- [ ] **Specific V2 files**: Full paths to src/bot_v2/features/[slice]/
- [ ] **Clear success criteria**: How to know when slice works
- [ ] **Isolation verification**: How to verify slice independence
- [ ] **V2 test command**: Exact command to test slice

### 3. Agent Selection for V2
- [ ] **Right tools**: Agent has necessary tool access
- [ ] **V2 expertise**: Agent understands vertical slice principles
- [ ] **Not overkill**: Don't use tech-lead for simple slice tasks
- [ ] **Not underpowered**: Don't use read-only agent for slice fixes

### 4. V2 Instruction Completeness
- [ ] **No assumed context**: Agent won't know previous V2 work
- [ ] **Slice isolation noted**: Emphasize no cross-slice imports
- [ ] **Specific return format**: Tell agent exactly what to return
- [ ] **V2 fallback instructions**: What to do if slice isolation breaks

---

## 📝 V2 Delegation Template

```markdown
@agent-[type]: [One-line V2 slice task summary]

V2 CONTEXT:
- System: Vertical slice architecture with complete isolation
- Working directory: /Users/rj/PycharmProjects/GPT-Trader
- Target slice: src/bot_v2/features/[slice]/
- Slice status: [From .knowledge/PROJECT_STATE.json v2_slices section]

TASK:
1. [Specific step 1 with exact V2 path/command]
2. [Specific step 2 maintaining slice isolation]
3. [V2 verification step with slice test command]

V2 ISOLATION REQUIREMENTS:
- No imports from other slices
- Local implementations only
- All types in local types.py
- Independent operation verified

SUCCESS CRITERIA:
- [ ] [Slice-specific measurable outcome]
- [ ] [Isolation verification passed]

RETURN FORMAT:
- If successful: "SUCCESS: [what was done] - Isolation: [verified/broken]"
- If failed: "FAILED: [specific error and what was tried]"

VERIFICATION:
Run: poetry run python src/bot_v2/test_[slice].py
Expected: All tests pass and slice loads independently
```

---

## 🚫 Common V2 Delegation Mistakes

### ❌ Breaking Slice Isolation
```
"Make the backtest slice use the risk utils from paper_trade slice"
```
**Why it fails**: Violates V2 isolation principle

### ❌ Using Old V1 Paths
```
"Fix the test in src/bot_v2/features/strategy/test_demo_ma.py"
```
**Why it fails**: V1 paths are archived, only V2 exists

### ❌ Cross-Slice Dependencies
```
"Import the shared config from common utilities"
```
**Why it fails**: V2 has no shared components

### ❌ Multiple Slice Responsibilities
```
"Fix backtest and paper_trade slices together"
```
**Why it fails**: Each slice must be handled independently

### ❌ No Isolation Verification
```
"Make the analyze slice work"
```
**Why it fails**: No verification that isolation is maintained

---

## ✅ Good V2 Delegation Examples

### Example 1: Fix V2 Slice Test
```
@agent-debugger: Fix failing backtest slice test

V2 CONTEXT:
- Target slice: src/bot_v2/features/backtest/
- Test file: src/bot_v2/test_backtest.py
- Error: "KeyError: 'returns' in calculate_metrics function"
- Slice should be completely isolated

TASK:
1. Run test: poetry run python src/bot_v2/test_backtest.py
2. Read implementation in src/bot_v2/features/backtest/backtest.py
3. Fix: Add returns calculation before metrics
4. Verify isolation: grep -r "from bot_v2.features" src/bot_v2/features/backtest/
5. Test: poetry run python src/bot_v2/test_backtest.py

V2 ISOLATION REQUIREMENTS:
- Keep all imports within the backtest slice
- Use local types from src/bot_v2/features/backtest/types.py
- No dependencies on other slices

RETURN:
"FIXED: Added returns calculation on line X - Isolation: verified" or error details
```

### Example 2: V2 Slice Analysis
```
@agent-code-archaeologist: Analyze ml_strategy slice for isolation compliance

V2 CONTEXT:
- Target slice: src/bot_v2/features/ml_strategy/
- Focus: Verify complete self-containment (~500 tokens to load)
- Check: No cross-slice imports, local types, independent operation

TASK:
1. Read all files in src/bot_v2/features/ml_strategy/
2. Check imports: grep -r "from bot_v2.features" src/bot_v2/features/ml_strategy/
3. Verify local types: check types.py completeness
4. Test independence: python -c "from src.bot_v2.features.ml_strategy import *"
5. Measure: Count tokens needed to load slice

V2 ISOLATION REQUIREMENTS:
- Zero cross-slice imports
- All needed types locally defined
- README.md with clear usage examples
- Independent loading verified

RETURN:
JSON: {"isolation_score": "pass/fail", "issues": [], "token_count": N, "dependencies": []}
```

### Example 3: New V2 Slice Implementation
```
@agent-backend-developer: Create new performance_monitor slice

V2 CONTEXT:
- New slice: src/bot_v2/features/performance_monitor/
- Purpose: Real-time performance tracking for trading
- Must follow: Complete isolation principles
- Integration: Via test file src/bot_v2/test_performance_monitor.py

TASK:
1. Create directory: src/bot_v2/features/performance_monitor/
2. Implement performance_monitor.py with local functions only
3. Create types.py with all needed types (no external imports)
4. Create README.md with usage examples
5. Write test: src/bot_v2/test_performance_monitor.py
6. Verify isolation: no imports from other slices

V2 ISOLATION REQUIREMENTS:
- Complete self-containment
- Duplicate any needed utilities locally
- Local type definitions
- Independent operation

RETURN:
"IMPLEMENTED: Created performance_monitor slice - Files: [list] - Isolation: verified"
```

---

## 📊 V2 Task → Agent Mapping

| V2 Task Type | Best Agent | Key Requirement |
|--------------|-----------|-----------------|
| Fix slice test | `debugger` | Maintain isolation |
| Analyze slice quality | `code-reviewer` | Check isolation compliance |
| Implement new slice | `backend-developer` | Follow isolation principles |
| Optimize slice performance | `performance-optimizer` | Local optimizations only |
| Create slice docs | `documentation-specialist` | Document isolation |
| Plan slice implementation | `planner` | Design for isolation |
| Review slice architecture | `tech-lead-orchestrator` | Assess V2 compliance |

---

## 🎯 V2-Specific Principles

1. **Slices are islands**: No communication between features
2. **Isolation is sacred**: Never break for convenience  
3. **Duplication over sharing**: Copy code rather than import
4. **Local everything**: Types, utilities, constants - all local
5. **Test independence**: Each slice tests in isolation
6. **V2 paths only**: src/bot_v2/features/[slice]/ only

## 📁 V2 Repository Structure (Post-Cleanup)

```
src/bot_v2/                    # ONLY active system (8K lines)
├── features/                  # 9 isolated slices
│   ├── backtest/             # ~500 tokens each
│   ├── paper_trade/          # Complete isolation
│   ├── analyze/              # No cross-dependencies  
│   ├── optimize/             # Self-contained
│   ├── live_trade/           # Local implementations
│   ├── monitor/              # Local types
│   ├── data/                 # Local everything
│   ├── ml_strategy/          # Week 1-2 complete
│   └── market_regime/        # Week 3 complete
└── test_*.py                 # Integration tests

archived/                     # 7.5M historical (don't reference)
tests/                       # 1.6M V2-focused tests
scripts/                     # 1.1M organized by function
docs/                        # 1.1M consolidated docs
data/                        # 344K essential data only
logs/                        # 660K recent logs only
```

## 🔄 V2 Post-Delegation Workflow

After V2 agent returns:

1. **Parse Response**: Extract success/failure and isolation status
2. **Verify Slice**: Run slice-specific test
3. **Check Isolation**: Verify no cross-slice imports added
4. **Update State**: Modify .knowledge/PROJECT_STATE.json v2_slices section
5. **Test Independence**: Verify slice loads independently

Never trust agent claims about isolation without verification!

## 🚀 Quick V2 Verification Commands

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

The V2 system is all about **complete isolation** - never compromise this principle!