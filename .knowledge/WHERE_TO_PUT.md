# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# 📍 Where to Put Things

Quick reference for where different types of content belong in the GPT-Trader V2 repository.

## Decision Tree

| What | Where | Example |
|------|-------|---------|
| **Code** | `src/bot_v2/features/[slice]/` | New trading logic → create new slice |
| **Tests** | `src/bot_v2/test_[slice].py` | Test for backtest → `test_backtest.py` |
| **Agent Knowledge** | `.knowledge/` | Navigation guides, patterns, state |
| **Config Files** | `config/` | Poetry files, dependencies |
| **Environment** | Root (`.env.template`) | API keys template only |
| **Archives** | `archived/` | Old code, deprecated systems |

## Detailed Rules

### ✅ New Feature Code
```
src/bot_v2/features/[new_slice]/
├── __init__.py      # Public API
├── core.py          # Main logic
├── models.py        # Data models (if needed)
└── utils.py         # Local utilities
```

### ✅ Test Files
```
src/bot_v2/test_[slice].py  # Integration test for slice
```

### ✅ Knowledge Updates
```
.knowledge/
├── STATE.json          # Update for system changes
├── HOW_TO/            # Update task guides
└── REFERENCE/         # Update technical docs
```

### ❌ NEVER Create
- Root Python files (archive them)
- Documentation in `docs/` (use `.knowledge/`)
- Shared code directories (`common/`, `utils/`)
- Reports or analysis files (update STATE.json)
- `.env` file (security risk)
- `.venv/` in repository (local only)

## Common Scenarios

### "I need to add ML prediction"
→ Create new slice: `src/bot_v2/features/ml_prediction/`

### "I need to update documentation"
→ Update existing files in `.knowledge/`

### "I found a bug in backtest"
→ Fix in `src/bot_v2/features/backtest/`

### "I need a utility function"
→ Add to the slice's `utils.py` (duplicate if needed)

### "I want to share code between slices"
→ DON'T! Duplicate the code in each slice

### "I need to track progress"
→ Update `.knowledge/STATE.json`

### "I have analysis results"
→ Add to STATE.json metrics section

## File Size Guidelines

- **Slice**: ~500 tokens total
- **Test file**: < 200 lines
- **Knowledge files**: < 150 lines
- **STATE.json**: < 200 lines

## Quick Check

Before creating any file, ask:
1. Is it code? → Goes in a slice
2. Is it a test? → Goes in test_[slice].py
3. Is it knowledge? → Update existing .knowledge/ files
4. Is it config? → Goes in config/
5. None of above? → Probably shouldn't create it

## Remember

**When in doubt, don't create new files.**
Update existing ones instead.