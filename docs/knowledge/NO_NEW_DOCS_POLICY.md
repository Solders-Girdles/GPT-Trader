# 🚫 NO NEW DOCUMENTATION POLICY

## STRICT RULES - NO EXCEPTIONS

### ❌ NEVER CREATE THESE FILES
```
*_REPORT.md
*_COMPLETE.md
*_STATUS.md
*_ANALYSIS.md
*_IMPLEMENTATION.md
*_SUMMARY.md
*_PROGRESS.md
*_UPDATE.md
```

### ✅ ONLY UPDATE THESE EXISTING FILES

| File | Update When | Update What |
|------|-------------|-------------|
| **.knowledge/PROJECT_STATE.json** | After ANY component change | Status, verification, timestamp |
| **.knowledge/KNOWN_FAILURES.md** | After NEW error discovered | Add solution under appropriate section |
| **docs/knowledge/TEST_MAP.json** | After adding/moving tests | Test locations only |
| **docs/knowledge/DEPENDENCIES.json** | After architectural changes | Rarely needs updates |
| **.gitignore** | NEVER | Don't hide problems |

### 📝 Where Information Goes Instead

| If You Want To Document... | Put It In... |
|----------------------------|--------------|
| Task completion | Update .knowledge/PROJECT_STATE.json status |
| Error and fix | Add to .knowledge/KNOWN_FAILURES.md |
| Test results | Update .knowledge/PROJECT_STATE.json verified flag |
| Implementation details | Code comments (minimal) |
| Architecture decisions | Update existing README.md |
| Performance metrics | Add to .knowledge/PROJECT_STATE.json |
| Bug discovery | Create GitHub issue or fix immediately |
| Feature completion | Update .knowledge/PROJECT_STATE.json components |

### 🔒 Enforcement

```python
# This check MUST pass before committing
def no_new_reports_check():
    """Agents must run this before claiming completion."""
    import glob
    
    bad_patterns = [
        "*_REPORT.md",
        "*_COMPLETE.md", 
        "*_STATUS.md",
        "*_IMPLEMENTATION.md"
    ]
    
    for pattern in bad_patterns:
        if glob.glob(pattern):
            raise ValueError(f"❌ Found forbidden file matching {pattern}")
            
    print("✅ No report files created")
    return True
```

### ⚠️ If You're About to Create a Report

**STOP** and ask:
1. Does this info belong in .knowledge/PROJECT_STATE.json? (usually yes)
2. Is this a recurring error for .knowledge/KNOWN_FAILURES.md?
3. Is this already documented elsewhere?
4. Will anyone ever read this file again? (usually no)

### 🎯 The Only Acceptable New Files

1. **Source code** in `src/bot/`
2. **Tests** in `tests/`
3. **Scripts** in `scripts/` (if they DO something)
4. **Config files** (if required by tools)

### 🔴 Violation Consequences

If you create a report file:
1. It will be immediately deleted
2. The task is considered INCOMPLETE
3. You must redo without creating reports

## Remember

**Code is documentation. Tests are proof. Reports are waste.**

Update existing knowledge files. Don't create new ones.