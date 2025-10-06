# Phase 1: Documentation Hygiene - Changelog

**Date**: 2025-10-06
**Status**: ✅ Complete
**Duration**: ~2 hours

---

## Summary

Phase 1 cleaned up configuration drift by removing 16 orphaned/broken config files and updating documentation to reflect the actual configuration system. The bot's config system is now accurately documented: hardcoded profiles + env vars, not YAML-based.

## Changes Made

### 🗑️ Deletions (16 files)

#### Orphaned Configs from Deleted Features (14 files)
```
config/acceptance_tuning.yaml
config/adaptive_portfolio_aggressive.yaml
config/adaptive_portfolio_config.yaml
config/adaptive_portfolio_conservative.yaml
config/backtest_config.yaml
config/brokers/coinbase_perp_specs.yaml
config/database.yaml
config/live_trade_config.yaml
config/ml_strategy_config.yaml
config/position_sizing_config.yaml
config/stage1_scaleup.yaml
config/stage2_scaleup.yaml
config/stage3_scaleup.yaml
config/system_config.yaml
```

**Reason**: Features deleted/archived but configs remained, creating misleading documentation.

#### Broken Risk Configs (2 files)
```
config/risk/coinbase_perps.prod.yaml
config/risk/spot_top10.yaml
```

**Reason**:
- `coinbase_perps.prod.yaml`: YAML format, but `RiskConfig.from_json()` expects JSON (would crash)
- `spot_top10.yaml`: Code references `spot_top10.json` which doesn't exist (fallback never fired)

**Decision**: Option 2 - Retire broken overrides, keep working `dev_dynamic.json`

### 🔧 Code Changes

#### `src/bot_v2/orchestration/configuration.py:29`
```python
# Before
DEFAULT_SPOT_RISK_PATH = Path(...) / "config" / "risk" / "spot_top10.json"  # File doesn't exist

# After
DEFAULT_SPOT_RISK_PATH = Path(...) / "config" / "risk" / "dev_dynamic.json"  # Working file
```

**Impact**: Spot profile fallback now actually works

### 📝 Documentation Updates

#### `config/risk/README.md`
- **Before**: Referenced deleted `coinbase_perps.prod.yaml`, outdated usage
- **After**: Complete rewrite:
  - Documents env vars as primary config method
  - Lists 30+ actual `RISK_*` environment variables (verified against `live_trade_config.py:56-113`)
  - Explains RISK_CONFIG_PATH override (JSON only)
  - Migration notes explaining why YAMLs were removed
  - Examples for dev vs prod usage
- **Corrections Applied**: Fixed initial documentation errors:
  - Corrected env var names (e.g., `RISK_DAY_LEVERAGE_MAX_PER_SYMBOL` not `RISK_DAY_MAX_LEVERAGE`)
  - Fixed `RISK_DAILY_LOSS_LIMIT` examples in all locations (USD amount, not percentage):
    - Env var examples: 100 (USD) not 0.02
    - JSON template: 100 (USD) not 0.02, added explicit note
  - Added missing vars (circuit breakers, market impact, dynamic sizing, etc.)

#### `README.md`
**Changes**:
1. Fixed risk config reference (line 45):
   - Before: "`config/risk/spot_top10.json`" (doesn't exist)
   - After: Explains env vars + optional JSON override
   - Corrected: `RISK_DAILY_LOSS_LIMIT=100` (USD) not `0.02` (looks like percentage)

2. Removed deleted features from architecture tree:
   - Removed: `adaptive_portfolio/`, `state/`
   - Added note pointing to `archived/` directory

3. Added **"What Actually Works"** section:
   - Configuration system reality (hardcoded profiles, NOT YAML)
   - Risk config (env vars + optional JSON)
   - List of what was removed and why

#### `docs/ARCHITECTURE.md`
**Changes**:
1. Updated refactoring history (lines 29-35):
   - Added adaptive_portfolio and state platform retirement
   - Added config cleanup (16 files removed)

2. Removed `adaptive_portfolio/` from feature tree (line 54)
   - Added note about archival

3. Added **Configuration System** section (lines 331-379):
   - Profile-based configuration (hardcoded + 3 YAMLs)
   - Risk configuration (env vars + optional JSON)
   - Configuration migration notes
   - Clear distinction: NOT YAML-based

### 📚 New Documentation

#### `docs/ops/RISK_CONFIG_DECISION.md`
Detailed options analysis for risk config cleanup with rationale for chosen Option 2.

#### `docs/ops/PHASE1_CHANGELOG.md` (this file)
Complete record of Phase 1 changes.

---

## Verification

### Grep Checks (All Pass ✅)
```bash
# Check: No references to deleted configs
rg "adaptive_portfolio.*\.yaml" src/ docs/ README.md
# Result: 0 matches ✅

# Check: No references to deleted features in main docs
rg "adaptive_portfolio" README.md docs/ARCHITECTURE.md
# Result: Only archive references ✅

# Check: Risk config accuracy
rg "spot_top10\.json" src/
# Result: 1 match in configuration.py (fixed) ✅
```

### Config Files Remaining
```bash
find config -type f \( -name "*.yaml" -o -name "*.json" \)
# Result:
config/.pre-commit-config.yaml     # Git hooks (kept)
config/profiles/canary.yaml        # Active ✅
config/profiles/dev_entry.yaml     # Active ✅
config/profiles/spot.yaml          # Active ✅
config/risk/dev_dynamic.json       # Active ✅
```

### Documentation Accuracy
- ✅ README.md "What Actually Works" section explains real config system
- ✅ ARCHITECTURE.md Configuration System section documents implementation
- ✅ config/risk/README.md lists 30+ actual RISK_* env vars (verified against live_trade_config.py)
- ✅ No references to deleted features in active docs
- ✅ Risk config examples corrected (USD amounts, not percentages)
- ✅ Env var names match actual code (RISK_DAY_LEVERAGE_MAX_PER_SYMBOL, etc.)

---

## Impact Assessment

### User-Facing Changes
1. **Risk Config**: Users must use env vars or `RISK_CONFIG_PATH=config/risk/dev_dynamic.json` (JSON only)
2. **Documentation**: Clear explanation of what actually works vs what's in archived/
3. **No Breaking Changes**: Existing env-var-based configs continue to work

### AI Agent Impact
1. **Before**: Agents would try to use non-existent configs, leading to confusion
2. **After**: Clear documentation of actual config system, agents can work correctly

### Maintenance Impact
1. **Before**: 16 misleading config files, broken fallbacks
2. **After**: Only working configs remain, documented accurately

---

## Lessons Learned

### Root Causes
1. **Config files outlived features**: Deleting code but not config creates drift
2. **Format mismatches**: YAML configs for JSON-only loader (never tested)
3. **Path bugs**: Code references `.json`, file is `.yaml` (never worked)
4. **Documentation lag**: Docs not updated when features removed

### Prevention
1. **Delete configs with features**: When archiving/removing code, delete associated configs
2. **Verify config loading**: Test that config files can actually be loaded by the code
3. **Document reality**: Update docs immediately when removing features
4. **Regular audits**: Periodic config file audits to catch orphans

---

## Next Steps

### Phase 2: Remove Unused Over-Engineering (Deferred)
The initial audit claimed registries/factories were "unused" but Phase 0 proved this wrong:
- ServiceRegistry: 9+ files, core DI
- CapabilityRegistry: Used in order_policy.py
- All abstractions actively referenced

**Decision**: Skip Phase 2 removal, abstractions are in use. May simplify PerpsBotBuilder later.

### Phase 3: Execution Layer Consolidation (Future)
- Decide: LiveExecutionEngine vs AdvancedExecutionEngine
- Requires deeper analysis of trade-offs
- Document coordinator/orchestrator boundaries

### Phase 4: Stabilization (Future)
- Fix remaining xfail tests
- Establish coding standards
- Create AI agent guidelines
- Feature freeze to prevent churn

---

## Files Changed Summary

### Deleted (16)
- 14 orphaned config files
- 2 broken risk configs

### Modified (4)
- `src/bot_v2/orchestration/configuration.py` - Fix DEFAULT_SPOT_RISK_PATH
- `config/risk/README.md` - Complete rewrite
- `README.md` - Remove deleted refs, add "What Actually Works"
- `docs/ARCHITECTURE.md` - Add config system section

### Created (3)
- `docs/ops/RISK_CONFIG_DECISION.md`
- `docs/ops/PHASE1_CHANGELOG.md` (this file)
- Updated `docs/ops/phase0_inventory.md` with execution summary

---

## Sign-Off

**Checkpoint 1 Criteria**: ✅ All Met
- [x] All asset categories inventoried
- [x] Safe-to-delete classification complete
- [x] User approved deletions (Option 2 for risk configs)
- [x] 16 files removed
- [x] Documentation updated
- [x] Grep verification passed

**Phase 1 Complete**: Ready for review and commit.
