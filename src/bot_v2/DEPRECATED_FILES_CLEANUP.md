# 🧹 Deprecated Files Cleanup Plan

## 📊 Current Issue

Our repository has accumulated many deprecated files from previous iterations that could cause confusion and outdated knowledge. These need to be cleaned up to maintain system clarity.

## 📁 Deprecated Files Identified

### Root Directory Reports (Safe to Remove)
```
❌ CRITICAL_FIXES_COMPLETE.md
❌ DEPRECATION_AUDIT_REPORT.md  
❌ DEPRECATION_CLEANUP_COMPLETE.md
❌ DEVELOPER_EXPERIENCE_IMPROVEMENTS.md
❌ INTEGRATION_COMPLETE.md
❌ ONLINE_LEARNING_IMPLEMENTATION_REPORT.md
❌ PERFORMANCE_REPORT_AUTO_RETRAINING.md
❌ PERFORMANCE_REPORT_ITERROWS_OPTIMIZATION.md
❌ PICKLE_REMOVAL_REPORT.md
❌ RISK_INTEGRATION_IMPLEMENTATION_COMPLETE.md
❌ SECURITY_AUDIT_COMPLETE.md
❌ USER_EXPERIENCE_REPORT.md
❌ WEEK_7_IMPLEMENTATION_REPORT.md
```

### Archived Content (Already Moved)
```
✅ archived/old_plans/ - Outdated roadmaps and plans
✅ archived/old_reports/ - Historical status reports  
✅ archived/benchmarks/ - Old benchmark data
✅ archived/*_20250812_* - Timestamped archives from August
```

### Current Active Files (Keep)
```
✅ CLAUDE.md - Main project control file
✅ README.md - Project documentation
✅ CONTRIBUTING.md - Development guidelines
✅ src/bot_v2/ - Current active system
✅ SLICES.md - Architecture navigation
```

## 🎯 Cleanup Strategy

### Phase 1: Root Directory Cleanup
- Remove all deprecated report files from root
- Keep only essential files:
  - CLAUDE.md (control center)
  - README.md (main docs)
  - CONTRIBUTING.md (development)

### Phase 2: Archive Validation
- Verify archived/ contains only historical content
- No current code should reference archived files
- Ensure all active development is in src/bot_v2/

### Phase 3: Knowledge Layer Update
- Update SLICES.md with all 8 current slices
- Update documentation to reflect current reality
- Remove references to deprecated components

## 📝 Action Items

### Immediate (Now)
- [x] Create this cleanup plan
- [x] Update SLICES.md with market_regime slice
- [ ] Remove deprecated report files from root
- [ ] Update CLAUDE.md if needed

### Next Session
- [ ] Validate all slice references are current
- [ ] Check for any remaining deprecated code
- [ ] Update any documentation pointing to old files

## 🛡️ Safety Measures

1. **Never remove without review** - Always check what references a file first
2. **Archive before delete** - Important files should be moved to archived/ first  
3. **Update references** - Fix any broken links after moving files
4. **Test after cleanup** - Ensure system still works after changes

## ✅ Current Clean State

After cleanup, repository should contain only:

```
GPT-Trader/
├── CLAUDE.md                    # Control center
├── README.md                    # Main documentation  
├── CONTRIBUTING.md              # Development guide
├── src/bot_v2/                  # Active system (8 slices)
│   ├── features/               # Feature slices
│   │   ├── backtest/          ✅ Working
│   │   ├── paper_trade/       ✅ Working
│   │   ├── analyze/           ✅ Working
│   │   ├── optimize/          ✅ Working
│   │   ├── live_trade/        ✅ Working
│   │   ├── monitor/           ✅ Working
│   │   ├── data/              ✅ Working
│   │   └── market_regime/     ✅ Working (NEW!)
│   └── ml_strategy/           ✅ Working
├── archived/                   # Historical content only
├── tests/                      # Current tests
├── docs/                       # Current documentation
└── [other active directories]
```

## 🎯 Benefits

1. **Reduced Confusion** - No outdated files to accidentally reference
2. **Cleaner Navigation** - Easier to find current components
3. **Updated Knowledge** - All references point to current reality
4. **Faster Development** - Less time spent on deprecated content

---

**Status**: In Progress  
**Next**: Remove deprecated root files and update references