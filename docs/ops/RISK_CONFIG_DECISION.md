# Risk Config Decision Matrix

**Issue**: Phase 0 discovered broken risk config override system
**Status**: Requires decision before Phase 1 cleanup

---

## Problem Summary

The risk config override system has **multiple broken pieces**:

### 1. Format Mismatch (CRITICAL)
```python
# runtime_coordinator.py:215
risk_config = RiskConfig.from_json(resolved_risk_path)
```

**RiskConfig.from_json() only accepts JSON**, but 2 of 3 config files are YAML:

| File | Format | Can Load? | Issue |
|------|--------|-----------|-------|
| `config/risk/dev_dynamic.json` | JSON | ✅ Yes | Works correctly |
| `config/risk/coinbase_perps.prod.yaml` | YAML | ❌ No | Would fail with JSON parse error |
| `config/risk/spot_top10.yaml` | YAML | ❌ No | Would fail with JSON parse error |

### 2. Path Mismatch (CRITICAL)
```python
# configuration.py:29
DEFAULT_SPOT_RISK_PATH = Path(__file__).resolve().parents[3] / "config" / "risk" / "spot_top10.json"
```

**Code references `spot_top10.json`** (doesn't exist), repo has `spot_top10.yaml` (can't be loaded)

Result: DEFAULT_SPOT_RISK_PATH.exists() always returns False, fallback never fires.

### 3. Runtime Behavior

**Current flow** (runtime_coordinator.py:202-217):
```
1. Check RISK_CONFIG_PATH env var
   ├─ If set → try RiskConfig.from_json(path)
   └─ If unset → check profile
       ├─ If SPOT/DEV/DEMO → check DEFAULT_SPOT_RISK_PATH.exists()
       │   ├─ If exists → RiskConfig.from_json(path)  [Never happens - file doesn't exist]
       │   └─ If not exists → RiskConfig.from_env()   [Always happens]
       └─ Else → RiskConfig.from_env()

Fallback: RiskConfig() with defaults
```

**Reality**:
- Only `dev_dynamic.json` can actually be loaded (via `RISK_CONFIG_PATH` env var)
- YAML files would crash if someone tried to use them
- DEFAULT_SPOT_RISK_PATH never works (wrong file extension)

---

## Decision Options

### Option 1: Fix & Document (Recommended)
**Action**: Make the override system work as intended

**Steps**:
1. Convert YAML to JSON:
   ```bash
   # Convert existing YAMLs
   python -c "import yaml, json; yaml_to_json('config/risk/coinbase_perps.prod.yaml')"
   python -c "import yaml, json; yaml_to_json('config/risk/spot_top10.yaml')"
   ```

2. Fix DEFAULT_SPOT_RISK_PATH:
   ```python
   # configuration.py:29
   DEFAULT_SPOT_RISK_PATH = Path(...) / "config" / "risk" / "spot_top10.json"
   # Now file will exist after conversion
   ```

3. Update docs:
   - Document `RISK_CONFIG_PATH` env var in README
   - List available risk profiles with paths
   - Example: `RISK_CONFIG_PATH=config/risk/dev_dynamic.json`

**Pros**:
- Preserves opt-in override capability
- Makes existing configs usable
- Clear documentation for future use

**Cons**:
- Requires conversion work
- Adds maintenance burden

---

### Option 2: Retire Broken Overrides
**Action**: Remove unusable YAML files, keep only dev_dynamic.json

**Steps**:
1. Delete broken files:
   ```bash
   rm config/risk/coinbase_perps.prod.yaml
   rm config/risk/spot_top10.yaml
   ```

2. Fix DEFAULT_SPOT_RISK_PATH:
   ```python
   # configuration.py:29
   DEFAULT_SPOT_RISK_PATH = Path(...) / "config" / "risk" / "dev_dynamic.json"
   # OR remove entirely and always use from_env()
   ```

3. Update docs:
   - Document single working override: dev_dynamic.json
   - Recommend env vars for customization (via from_env())
   - Mark RISK_CONFIG_PATH as advanced/optional

**Pros**:
- Removes broken/misleading configs
- Simpler system (env vars + optional dev_dynamic.json)
- Less maintenance

**Cons**:
- Loses potential prod/spot profiles (though they never worked)
- Users relying on RISK_CONFIG_PATH lose flexibility

---

### Option 3: Full Retirement (Simplest)
**Action**: Remove all risk config overrides, use env vars only

**Steps**:
1. Delete all risk configs:
   ```bash
   rm -r config/risk/
   ```

2. Remove RISK_CONFIG_PATH logic from runtime_coordinator.py

3. Update docs:
   - Document RiskConfig.from_env() as THE way to configure risk
   - List all supported env vars
   - Example: `RISK_MAX_LEVERAGE=3 RISK_DAILY_LOSS_LIMIT=0.02`

**Pros**:
- Simplest system
- No config file confusion
- Pure env-var driven (12-factor compliant)

**Cons**:
- Loses file-based config option
- More env vars to manage

---

## Current State Inventory

### What Exists
```
config/risk/
├── coinbase_perps.prod.yaml  (979 bytes, YAML, BROKEN)
├── dev_dynamic.json           (627 bytes, JSON, WORKS)
├── spot_top10.yaml            (860 bytes, YAML, BROKEN)
└── README.md                  (1820 bytes, may be outdated)
```

### What Code Expects
```python
# Only JSON files via RiskConfig.from_json()
# DEFAULT_SPOT_RISK_PATH points to non-existent spot_top10.json
# RISK_CONFIG_PATH env var for opt-in override
```

### What Actually Works
- `RISK_CONFIG_PATH=config/risk/dev_dynamic.json` ✅
- `RISK_CONFIG_PATH=config/risk/coinbase_perps.prod.yaml` ❌ (JSON parse error)
- `RISK_CONFIG_PATH=config/risk/spot_top10.yaml` ❌ (JSON parse error)
- DEFAULT_SPOT_RISK_PATH fallback ❌ (file doesn't exist)

---

## Recommendation

**I recommend Option 2: Retire Broken Overrides**

**Rationale**:
1. The YAML files have **never worked** (wrong format)
2. DEFAULT_SPOT_RISK_PATH has **never worked** (wrong filename)
3. Only dev_dynamic.json is proven working
4. Env vars provide all needed flexibility (from_env())
5. Simpler is better after cleanup

**Implementation**:
1. Delete: coinbase_perps.prod.yaml, spot_top10.yaml
2. Keep: dev_dynamic.json (proven working)
3. Fix: DEFAULT_SPOT_RISK_PATH → dev_dynamic.json OR remove fallback logic
4. Document: RISK_CONFIG_PATH as optional override, env vars as primary config

**Phase 1 Integration**:
- Bundle this with config cleanup
- Update "What Actually Works" section to explain risk config system
- Add env var reference for all RISK_* variables

---

## User Decision Required

**Please choose**:
- [ ] **Option 1**: Fix & document (convert YAML→JSON, make system work)
- [ ] **Option 2**: Retire broken overrides (delete YAMLs, keep dev_dynamic.json)
- [ ] **Option 3**: Full retirement (remove all, env vars only)
- [ ] **Custom**: Different approach?

Once decided, I'll integrate into Phase 1 execution.
