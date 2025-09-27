# V2 Knowledge Layer Maintenance Guide

## The Problem We're Solving

**Agents create reports because they have no persistent memory.**
Our V2 knowledge layer provides that memory for vertical slice architecture, but only if it stays current.

## V2-Focused Automated Maintenance System

### 1. **V2 Validation Script** (`scripts/validation/validate_knowledge.py`)
Detects staleness and V2 isolation issues:
- Flags stale V2 slice verifications (>24 hours)
- Finds forbidden report files
- Checks V2 slice isolation compliance
- Validates test references are valid for V2 structure
- Auto-updates timestamps

**Run after EVERY V2 change:**
```bash
poetry run python scripts/validation/validate_knowledge.py
```

### 2. **V2 Update Triggers**

| V2 Event | Required Update | Command |
|----------|----------------|---------|
| Slice fixed | .knowledge/PROJECT_STATE.json v2_slices section | `python -c "from scripts.validation.validate_knowledge import update_v2_slice; update_v2_slice('backtest', 'working')"` |
| New V2 error | .knowledge/KNOWN_FAILURES.md | Add under V2 Slice Issues section |
| Slice test added | docs/knowledge/TEST_MAP.json | Update V2 test location |
| Isolation violation | .knowledge/PROJECT_STATE.json | Add isolation_violations |
| Report file created | Delete it | `rm *_REPORT.md *_COMPLETE.md` |
| 24 hours passed | Re-verify all slices | `poetry run python src/bot_v2/test_all_slices.py` |

### 3. **V2 Enforcement Rules**

#### Pre-Commit Hook (add to `.git/hooks/pre-commit`)
```bash
#!/bin/bash
# Prevent committing report files
if git diff --cached --name-only | grep -E "_REPORT\.md|_COMPLETE\.md|_STATUS\.md"; then
    echo "❌ Cannot commit report files. Move info to .knowledge/PROJECT_STATE.json"
    exit 1
fi

# Validate V2 knowledge layer
poetry run python scripts/validation/validate_knowledge.py || exit 1

# Check V2 slice isolation
if git diff --cached --name-only | grep "src/bot_v2/features/"; then
    echo "🔍 Checking V2 slice isolation..."
    for slice in backtest paper_trade analyze optimize live_trade monitor data ml_strategy market_regime; do
        if grep -r "from bot_v2.features" "src/bot_v2/features/$slice/" 2>/dev/null; then
            echo "❌ Isolation violation in $slice slice"
            exit 1
        fi
    done
fi
```

#### V2 Agent Task Template Addition
Add to EVERY V2 agent task:
```
AFTER completing V2 task:
1. Run: poetry run python scripts/validation/validate_knowledge.py
2. If slice status changed, update .knowledge/PROJECT_STATE.json v2_slices section
3. Verify slice isolation: grep -r "from bot_v2.features" src/bot_v2/features/[slice]/
4. If isolation violated, add to .knowledge/PROJECT_STATE.json isolation_violations
5. DO NOT create any *_REPORT.md files
```

## V2-Specific Maintenance Features

### ✅ Prevents Report Sprawl (V2 Edition)
- **NO_NEW_DOCS_POLICY.md** - Explicit ban on reports in V2 system
- **V2 validation script** - Detects violations in slice context  
- **Git hooks** - Prevent committing reports and isolation violations

### ✅ Stays Current with V2 Reality
- **Auto-update mechanisms** - V2 slice timestamps, verification flags
- **Isolation validation** - Checks slice independence
- **V2 structure validation** - Ensures proper slice organization

### ✅ V2 Single Source of Truth

| Information Type | Single Location | V2 Context |
|-----------------|-----------------|------------|
| V2 slice status | .knowledge/PROJECT_STATE.json v2_slices | All 9 slice states |
| V2 error solutions | .knowledge/KNOWN_FAILURES.md | Slice-specific solutions |
| V2 slice dependencies | docs/knowledge/DEPENDENCIES.json | Isolation violations |
| V2 test locations | docs/knowledge/TEST_MAP.json | src/bot_v2/test_*.py |
| V2 import patterns | docs/knowledge/IMPORTS.md | Local-only imports |

### ✅ V2 Self-Healing
- Stale slice verifications auto-marked as unverified
- Isolation violations auto-detected
- Missing V2 tests auto-flagged
- Cross-slice imports auto-detected and flagged

## V2 Maintenance Workflow

### Daily (Automated V2 Checks)
```bash
# Morning V2 validation
poetry run python scripts/validation/validate_knowledge.py

# V2 slice isolation check
for slice in backtest paper_trade analyze optimize live_trade monitor data ml_strategy market_regime; do
    echo "Checking $slice isolation..."
    if grep -r "from bot_v2.features" "src/bot_v2/features/$slice/" 2>/dev/null; then
        echo "⚠️ Isolation violation in $slice"
    fi
done

# If issues found, run V2 tests
poetry run python src/bot_v2/test_all_slices.py

# Update V2 state
poetry run python scripts/validation/validate_knowledge.py
```

### After Each V2 Slice Fix
```python
# In agent task or script
from scripts.validation.validate_knowledge import update_v2_slice

# After fixing slice
update_v2_slice('backtest', 'working')

# After encountering new slice error  
update_v2_slice('paper_trade', 'failed', 'KeyError: symbol not found')

# After isolation violation
update_v2_slice('analyze', 'broken_isolation', 'imports from backtest slice')
```

### Weekly V2 Review
1. Check .knowledge/KNOWN_FAILURES.md for outdated V2 solutions
2. Verify docs/knowledge/DEPENDENCIES.json reflects V2 isolation
3. Update docs/knowledge/TEST_MAP.json with new V2 tests
4. Audit V2 slice isolation compliance
5. Delete any accumulated report files

## V2 Repository Structure Post-Cleanup

Current ultraclean structure (230M+ freed):
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

.knowledge/                    # V2 state tracking
├── PROJECT_STATE.json        # v2_slices section
├── KNOWN_FAILURES.md         # V2-specific solutions
├── SYSTEM_REALITY.md         # V2 architecture notes
└── ROADMAP.json              # V2 development path

docs/knowledge/               # V2 workflow guides
├── TASK_TEMPLATES.md         # V2 slice templates
├── AGENT_WORKFLOW.md         # V2 best practices
├── AGENT_DELEGATION_GUIDE.md # V2 isolation focus
└── TEST_MAP.json             # V2 test locations

archived/                     # 7.5M historical (don't maintain)
tests/                       # 1.6M V2-focused tests
scripts/                     # 1.1M organized by function
```

## Signs V2 Knowledge Layer is Working

### ✅ Good Signs
- No new `*_REPORT.md` files for 7+ days
- .knowledge/PROJECT_STATE.json v2_slices updated daily
- All V2 slices show isolation: "verified"/"broken"
- No cross-slice imports detected
- V2 tests passing independently
- .knowledge/KNOWN_FAILURES.md growing with V2 solutions

### ⚠️ Warning Signs  
- Finding `*_REPORT.md` files
- V2 slice "last_verified" older than 7 days
- Cross-slice imports detected
- "isolation": null in v2_slices
- V2 validation script showing violations
- Agents creating new V2 documentation

## V2 Isolation Monitoring

### Critical V2 Commands
```bash
# Check specific slice isolation
grep -r "from bot_v2.features" src/bot_v2/features/[slice]/

# Test slice independence  
python -c "from src.bot_v2.features.[slice] import *"

# Verify all V2 slices
poetry run python src/bot_v2/test_all_slices.py

# Check V2 knowledge state
cat .knowledge/PROJECT_STATE.json | jq '.v2_slices'
```

### V2 Quick Fixes
```bash
# Update V2 slice status
python -c "from scripts.validation.validate_knowledge import update_v2_slice; update_v2_slice('backtest', 'working')"

# Mark isolation violation
python -c "from scripts.validation.validate_knowledge import update_v2_slice; update_v2_slice('analyze', 'broken_isolation')"

# Clean up reports
rm -f *_REPORT.md *_COMPLETE.md *_STATUS.md

# Check failed V2 slices
python -c "import json; s=json.load(open('.knowledge/PROJECT_STATE.json')); print([k for k,v in s['v2_slices'].items() if v['status']=='failed'])"
```

## The V2 Key Insight

**The V2 knowledge layer only works if it's the ONLY place for V2 slice information.**

By enforcing:
1. V2 slice isolation principles
2. No cross-slice dependencies
3. Local implementations only  
4. Single source of truth for slice status
5. Automated violation detection

We maintain clean V2 architecture while keeping knowledge current.

## V2 Quick Reference Card

```bash
# Validate V2 knowledge (run often!)
poetry run python scripts/validation/validate_knowledge.py

# Update V2 slice after fixing
python -c "from scripts.validation.validate_knowledge import update_v2_slice; update_v2_slice('slice_name', 'working')"

# Test all V2 slices
poetry run python src/bot_v2/test_all_slices.py

# Check V2 slice isolation
grep -r "from bot_v2.features" src/bot_v2/features/

# Clean up reports  
rm -f *_REPORT.md *_COMPLETE.md *_STATUS.md

# Check V2 slice health
cat .knowledge/PROJECT_STATE.json | jq '.v2_slices | to_entries[] | select(.value.status != "working")'
```

This V2 maintenance system ensures the knowledge layer remains the single, current, comprehensive source of truth for our vertical slice architecture.