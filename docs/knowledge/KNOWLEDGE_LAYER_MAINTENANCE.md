# Knowledge Layer Maintenance Guide

## The Problem We're Solving

**Agents create reports because they have no persistent memory.**
Our knowledge layer provides that memory, but only if it stays current.

## Automated Maintenance System

### 1. **Validation Script** (`scripts/validate_knowledge.py`)
Detects staleness and inconsistencies:
- Flags stale verifications (>24 hours)
- Finds forbidden report files
- Checks test references are valid
- Auto-updates timestamps

**Run after EVERY change:**
```bash
poetry run python scripts/validate_knowledge.py
```

### 2. **Update Triggers**

| Event | Required Update | Command |
|-------|----------------|---------|
| Component fixed | .knowledge/PROJECT_STATE.json status | `python -c "from scripts.validate_knowledge import update_after_change; update_after_change('component', 'working')"` |
| New error found | .knowledge/KNOWN_FAILURES.md | Add under appropriate section |
| Test added/moved | docs/knowledge/TEST_MAP.json | Update test location |
| Report file created | Delete it | `rm *_REPORT.md` |
| 24 hours passed | Re-verify all | `poetry run python scripts/verify_capabilities.py` |

### 3. **Enforcement Rules**

#### Pre-Commit Hook (add to `.git/hooks/pre-commit`)
```bash
#!/bin/bash
# Prevent committing report files
if git diff --cached --name-only | grep -E "_REPORT\.md|_COMPLETE\.md|_STATUS\.md"; then
    echo "❌ Cannot commit report files. Move info to .knowledge/PROJECT_STATE.json"
    exit 1
fi

# Validate knowledge layer
poetry run python scripts/validate_knowledge.py || exit 1
```

#### Agent Task Template Addition
Add to EVERY agent task:
```
AFTER completing task:
1. Run: poetry run python scripts/validate_knowledge.py
2. If component status changed, update .knowledge/PROJECT_STATE.json
3. If new error encountered, add to .knowledge/KNOWN_FAILURES.md
4. DO NOT create any *_REPORT.md files
```

## What Makes This Comprehensive

### ✅ Prevents Report Sprawl
- **NO_NEW_DOCS_POLICY.md** - Explicit ban on reports
- **Validation script** - Detects and flags violations
- **Git hooks** - Prevent committing reports

### ✅ Stays Current
- **Auto-update mechanisms** - Timestamps, verification flags
- **Validation script** - Runs checks automatically
- **Update helpers** - Simple functions to update state

### ✅ Single Source of Truth
| Information Type | Single Location |
|-----------------|-----------------|
| System status | .knowledge/PROJECT_STATE.json |
| Error solutions | .knowledge/KNOWN_FAILURES.md |
| Component dependencies | docs/knowledge/DEPENDENCIES.json |
| Test locations | docs/knowledge/TEST_MAP.json |
| Import patterns | docs/knowledge/IMPORTS.md |

### ✅ Self-Healing
- Stale verifications auto-marked as unverified
- Missing tests auto-detected
- Forbidden files auto-flagged for deletion

## Maintenance Workflow

### Daily (Automated)
```bash
# Morning validation
poetry run python scripts/validate_knowledge.py

# If issues found
poetry run python scripts/verify_capabilities.py

# Update state
poetry run python scripts/validate_knowledge.py
```

### After Each Fix
```python
# In agent task or script
from scripts.validate_knowledge import update_after_change

# After fixing component
update_after_change('data_pipeline', 'working')

# After encountering new error
update_after_change('risk_management', 'failed', 'ImportError: No module named risk')
```

### Weekly Review
1. Check .knowledge/KNOWN_FAILURES.md for outdated solutions
2. Verify docs/knowledge/DEPENDENCIES.json matches architecture
3. Update docs/knowledge/TEST_MAP.json with new tests
4. Delete any accumulated report files

## Signs Knowledge Layer is Working

### ✅ Good Signs
- No new `*_REPORT.md` files for 7+ days
- .knowledge/PROJECT_STATE.json updated daily
- All components show "verified": true/false (not null)
- .knowledge/KNOWN_FAILURES.md growing with solutions
- Validation script passes without issues

### ⚠️ Warning Signs
- Finding `*_REPORT.md` files
- "last_verified" older than 7 days
- "verified": null in components
- Validation script showing many issues
- Agents creating new documentation

## The Key Insight

**The knowledge layer only works if it's the ONLY place for this information.**

If agents can create reports elsewhere, they will. By:
1. Banning new reports (NO_NEW_DOCS_POLICY)
2. Validating constantly (validate_knowledge.py)
3. Auto-updating (update helpers)
4. Enforcing via git hooks

We force all information through the knowledge layer, keeping it current and comprehensive.

## Quick Reference Card

```bash
# Validate knowledge (run often!)
poetry run python scripts/validate_knowledge.py

# Update after fixing something
python -c "from scripts.validate_knowledge import update_after_change; update_after_change('component_name', 'working')"

# Full system verification
poetry run python scripts/verify_capabilities.py

# Clean up reports
rm -f *_REPORT.md *_COMPLETE.md *_STATUS.md

# Check what needs fixing
python -c "import json; s=json.load(open('PROJECT_STATE.json')); print([k for k,v in s['components'].items() if v['status']=='failed'])"
```

This maintenance system ensures the knowledge layer remains the single, current, comprehensive source of truth.