# Configuration Drift Scan - Phase 0

**Generated**: 2025-10-05
**Purpose**: Identify mismatches between templates, production configs, and runtime expectations

---

## Executive Summary

**Status**: ⚠️ **MODERATE DRIFT** - Multiple template versions with different purposes

**Key Findings**:
- ✅ Well-organized config structure with clear separation (environments, risk, profiles, brokers)
- ⚠️ Three different `.env` templates for different purposes (dev, prod, sandbox)
- ⚠️ Template vs production env has significant drift (different defaults and structure)
- ⚠️ Risk configs use mixed formats (YAML and JSON)
- ℹ️ No actual `.env` file in root (good for security, users must create from template)

---

## Configuration Inventory

### Directory Structure
```
config/
├── environments/
│   ├── .env.template          # Development template (114 lines)
│   ├── .env.production        # Production template (111 lines)
│   └── README.md
├── risk/
│   ├── coinbase_perps.prod.yaml   # Production risk config (YAML)
│   ├── dev_dynamic.json           # Dev risk config (JSON)
│   ├── spot_top10.json            # Spot trading risk (JSON)
│   ├── spot_top10.yaml            # Spot trading risk (YAML)
│   └── README.md
├── profiles/
│   ├── canary.yaml            # Canary deployment profile
│   ├── dev_entry.yaml         # Dev entry profile
│   └── spot.yaml              # Spot trading profile
├── brokers/
│   └── coinbase_perp_specs.yaml   # Coinbase perpetuals specs
├── archive/
│   └── README.md              # Archived configs
├── adaptive_portfolio_*.yaml  # 3 risk profile variants (aggressive, default, conservative)
├── backtest_config.yaml
├── database.yaml
├── live_trade_config.yaml
├── ml_strategy_config.yaml
├── position_sizing_config.yaml
├── system_config.yaml
└── stage{1,2,3}_scaleup.yaml

Root directory:
├── .env.sandbox.example       # Sandbox-specific template (18 lines)
└── .env                       # NOT present (users create from template)
```

**Total Config Files**: 23+ YAML/JSON files

---

## Environment Variable Templates

### 1. Development Template
**Path**: `config/environments/.env.template`
**Lines**: 114
**Purpose**: Local development with paper trading

**Key Characteristics**:
```bash
PERPS_PAPER=1                # Paper trading enabled
COINBASE_SANDBOX=1           # Use sandbox
COINBASE_ENABLE_DERIVATIVES=0  # Derivatives disabled
ENV=development
RISK_CONFIG_PATH=            # Empty (use defaults)
```

**Target Audience**: Developers testing locally

---

### 2. Production Template
**Path**: `config/environments/.env.production`
**Lines**: 111
**Purpose**: Live production deployment

**Key Characteristics**:
```bash
PERPS_PAPER=0                # Live trading
COINBASE_SANDBOX=0           # Production API
COINBASE_ENABLE_DERIVATIVES=1  # Derivatives enabled
RISK_CONFIG_PATH=config/risk/coinbase_perps.prod.yaml
# Requires CDP credentials
COINBASE_PROD_CDP_API_KEY=
COINBASE_PROD_CDP_PRIVATE_KEY=
```

**Target Audience**: Production deployment

**Additional Features**:
- Production preflight script instructions
- Structured deployment workflow (dry-run → canary → prod)
- Stricter risk defaults

---

### 3. Sandbox Soak Test Template
**Path**: `.env.sandbox.example` (root directory)
**Lines**: 18
**Purpose**: Sandbox soak testing with monitoring

**Key Characteristics**:
```bash
COINBASE_API_KEY=your_sandbox_api_key_here
COINBASE_API_SECRET=your_sandbox_api_secret_here
PERPS_ENABLE_STREAMING=1
PERPS_PROFILE=canary
ADMIN_PASSWORD=admin123      # For Grafana
DATABASE_PASSWORD=trader     # For PostgreSQL exporter
```

**Target Audience**: CI/CD soak tests, monitoring validation

**Scope**: Minimal (only 18 lines) - focused on soak testing

---

## Drift Analysis: Dev vs Production Templates

### Major Differences

| Setting | Dev Template | Prod Template | Impact |
|---------|-------------|---------------|--------|
| `PERPS_PAPER` | `1` (paper) | `0` (live) | Critical - trading mode |
| `COINBASE_SANDBOX` | `1` | `0` | Critical - API endpoint |
| `COINBASE_ENABLE_DERIVATIVES` | `0` | `1` | Major - product access |
| `RISK_CONFIG_PATH` | Empty | `config/risk/coinbase_perps.prod.yaml` | Major - risk limits |
| `ENV` | `development` | Not set | Minor - logging labels |
| CDP Credentials | Commented | Uncommented | Critical - auth method |

### Structural Differences

**Production template adds**:
1. **Deployment workflow instructions** (preflight, dry-run, canary)
2. **CDP/JWT credential fields** (for derivatives)
3. **Explicit risk config path**

**Development template includes**:
1. **More inline comments** (beginner-friendly)
2. **Optional runtime flags section** (examples commented)
3. **ENV label** (for local dev identification)

### Shared Settings (No Drift)

Both templates agree on:
- Order execution safety defaults (`ORDER_PREVIEW_ENABLED=1`, `DEFAULT_ORDER_TYPE=limit`)
- Risk management structure (though values differ)
- Logging configuration
- Monitoring settings
- Symbol/fee configuration format

---

## Risk Configuration Drift

### Format Inconsistency: YAML vs JSON

**YAML Configs** (3 files):
- `config/risk/coinbase_perps.prod.yaml` - Production perps risk
- `config/risk/spot_top10.yaml` - Spot trading risk

**JSON Configs** (2 files):
- `config/risk/dev_dynamic.json` - Development risk
- `config/risk/spot_top10.json` - Spot trading risk (duplicate format!)

**Issue**: `spot_top10` exists in BOTH YAML and JSON formats

**Recommendation**: Standardize on YAML (more comments, better for config)
```bash
# Future: Remove JSON versions or consolidate
rm config/risk/spot_top10.json  # After verifying YAML is canonical
```

---

### Risk Config Variations

#### Production Perps (`coinbase_perps.prod.yaml`)
```yaml
max_leverage: 5
daily_loss_limit: '50'
max_exposure_pct: 0.5
max_position_pct_per_symbol: 0.2
slippage_guard_bps: 30
# Day/night leverage scheduling
daytime_start_utc: '13:00'
daytime_end_utc: '20:00'
day_leverage_max_per_symbol:
  BTC-PERP: 10
  ETH-PERP: 8
```

**Characteristics**: Conservative base, higher leverage during day hours, per-symbol caps

---

#### Dev Dynamic (`dev_dynamic.json`)
```json
{
  "max_leverage": 3,
  "daily_loss_limit": "1000",
  "max_exposure_pct": 1.00,
  "max_position_pct_per_symbol": 1.00,
  "slippage_guard_bps": 80,
  "enable_dynamic_position_sizing": true,
  "position_sizing_method": "intelligent"
}
```

**Characteristics**: Much higher loss limits (dev testing), looser guards, enables experimental features

---

### Drift: Dev vs Prod Risk Defaults

| Setting | Dev | Prod | Ratio |
|---------|-----|------|-------|
| `max_leverage` | 3x | 5x | 1.7x |
| `daily_loss_limit` | $1000 | $50 | 20x ⚠️ |
| `max_exposure_pct` | 100% | 50% | 2x |
| `max_position_pct_per_symbol` | 100% | 20% | 5x |
| `slippage_guard_bps` | 80 | 30 | 2.7x |

**Analysis**:
- ⚠️ Dev loss limit 20x higher - intentional for testing with fake money
- ⚠️ Dev allows 100% exposure (risky even for dev - could hide bugs)
- ✅ Prod has stricter slippage protection
- ✅ Prod includes time-based leverage adjustments (day/night)

**Recommendation**:
- Dev loss limit is fine (paper trading), but consider lowering exposure/position limits to mirror production constraints (catch bugs earlier)

---

## Profile Configuration

### Available Profiles (3 total)

1. **`canary.yaml`** (7294 bytes) - Canary deployment profile
2. **`dev_entry.yaml`** (725 bytes) - Dev entry profile
3. **`spot.yaml`** (2419 bytes) - Spot trading profile

**Usage**: `PERPS_PROFILE=canary` or `--profile canary`

**Coverage**:
- ✅ Canary deployment covered
- ✅ Dev entry covered
- ✅ Spot trading covered
- ⚠️ No explicit "production" profile (uses canary?)
- ⚠️ No "staging" profile

**Recommendation**: Consider adding:
```yaml
# config/profiles/production.yaml
# config/profiles/staging.yaml
```

---

## Configuration Validation Findings

### Missing Schema Validation

**Current State**: No Pydantic schemas found for config file validation

**Risk**:
- ⚠️ Typos in YAML/JSON won't be caught until runtime
- ⚠️ Invalid values may be silently ignored

**Recommendation**: Create validation schemas (Phase 0 deliverable mentioned in work plan)
```python
# Future: scripts/tools/config_doctor.py
from pydantic import BaseModel, Field

class RiskConfig(BaseModel):
    max_leverage: int = Field(ge=1, le=20)
    daily_loss_limit: str  # Could be Decimal
    max_exposure_pct: float = Field(ge=0, le=1)
    # ... etc

def validate_risk_config(path: str) -> RiskConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return RiskConfig(**data)  # Raises ValidationError if invalid
```

---

### Undocumented Settings

**Observed in configs but not in templates**:

From `coinbase_perps.prod.yaml`:
- `daytime_start_utc` / `daytime_end_utc` - Not in any .env template
- `day_leverage_max_per_symbol` - Not in .env template
- `night_leverage_max_per_symbol` - Not in .env template
- `default_maintenance_margin_rate` - Not in .env template

From `dev_dynamic.json`:
- `enable_dynamic_position_sizing` - In .env as commented optional
- `position_sizing_method` - In .env as commented optional
- `enable_market_impact_guard` - In .env as commented optional

**Issue**: Risk configs have features not documented in .env templates

**Recommendation**:
1. Update `.env.template` to include commented examples of ALL risk config options
2. Or clarify that risk YAML/JSON fully overrides .env risk settings

---

## Configuration Hierarchy

### Current Loading Priority (Inferred)

1. **Environment variables** (`.env` file)
2. **Risk config file** (if `RISK_CONFIG_PATH` set)
3. **Profile config** (if `PERPS_PROFILE` set or `--profile` arg)
4. **Hardcoded defaults** (in code)

**Overlap**: Risk settings exist in:
- `.env` files (e.g., `RISK_MAX_LEVERAGE=3`)
- Risk YAML/JSON files (e.g., `max_leverage: 5`)
- Profile YAML files (unknown - not inspected)

**Question**: Which takes precedence?

**Recommendation**: Document config hierarchy in `config/README.md`
```markdown
# Configuration Precedence (highest to lowest)
1. CLI arguments (e.g., --symbols BTC-PERP)
2. Environment variables (.env)
3. Profile config (--profile canary → config/profiles/canary.yaml)
4. Risk config (RISK_CONFIG_PATH → config/risk/*.yaml)
5. Default config files (e.g., live_trade_config.yaml)
6. Hardcoded defaults
```

---

## Detected Mismatches & Inconsistencies

### 1. Duplicate Risk Configs
- ⚠️ `spot_top10.yaml` AND `spot_top10.json` - Which is canonical?

**Action**:
```bash
# Verify they're identical
diff <(yq -o=json config/risk/spot_top10.yaml) config/risk/spot_top10.json

# If identical, delete JSON and standardize on YAML
# If different, understand why and consolidate
```

---

### 2. Three .env Templates Without Clear Usage Docs

**Issue**: New developers may not know which template to use

**Solution**: Update `config/environments/README.md` with decision tree:
```markdown
# Which .env template should I use?

- **Local development / testing**: Copy `.env.template` to `.env`
- **Production deployment**: Copy `.env.production` to `.env` (or `.env.production.local`)
- **CI sandbox soak tests**: Copy `../../.env.sandbox.example` to `.env`

Never commit the actual `.env` file (it's in .gitignore).
```

---

### 3. Commented vs Uncommented Settings

**Dev template**:
```bash
# Uncomment when testing INTX/CDP credentials:
# COINBASE_PROD_CDP_API_KEY=
# COINBASE_PROD_CDP_PRIVATE_KEY=
```

**Prod template**:
```bash
COINBASE_PROD_CDP_API_KEY=
COINBASE_PROD_CDP_PRIVATE_KEY=
```

**Assessment**: ✅ Intentional - dev doesn't need CDP, prod requires it

---

### 4. ENV Variable Missing in Prod

**Dev template**:
```bash
ENV=development
```

**Prod template**: (not set)

**Impact**: Minor - only affects log labels

**Recommendation**: Add to prod template:
```bash
ENV=production
```

---

## Configuration Security Audit

### Secrets in Version Control ✅

**Checked**:
- ✅ `.env.template` - No secrets (placeholders only)
- ✅ `.env.production` - No secrets (placeholders only)
- ✅ `.env.sandbox.example` - Example passwords only (safe)

**Git ignore status**:
```bash
# .gitignore should contain:
.env
.env.local
.env.*.local
.env.production
```

**Verification needed**: Check if `.gitignore` properly excludes runtime .env files

---

### Hardcoded Test Passwords

In `.env.sandbox.example`:
```bash
ADMIN_PASSWORD=admin123
DATABASE_PASSWORD=trader
```

**Risk**: ⚠️ Low (sandbox only, but could be better)

**Recommendation**: Change to:
```bash
ADMIN_PASSWORD=change_me_please
DATABASE_PASSWORD=change_me_please
```

---

## Missing Configurations (Gaps)

### 1. No Staging Environment Template
- Dev template ✅
- Prod template ✅
- Staging template ❌

**Recommendation**: Create `config/environments/.env.staging`

---

### 2. No Backup/Recovery Configuration
**Searched for**: `backup`, `recovery`, `s3`, `archive`

**Found**: State management code exists (`src/bot_v2/state/backup/`) but no config template

**Recommendation**: Document backup config in `.env.template`:
```bash
# -----------------------------------------------------------------------------
# BACKUP & RECOVERY (OPTIONAL)
# -----------------------------------------------------------------------------
BACKUP_ENABLED=0
BACKUP_PROVIDER=s3              # s3, gcs, local
BACKUP_S3_BUCKET=
BACKUP_S3_REGION=us-east-1
BACKUP_INTERVAL_HOURS=24
```

---

### 3. No Database Configuration in .env Templates
**Found**: `config/database.yaml` exists

**Missing**: No `DATABASE_*` variables in .env templates (except sandbox password for exporter)

**Question**: Is database config only loaded from YAML? Or also from env vars?

**Recommendation**: Document database config loading in `config/README.md`

---

## Recommendations Summary

### High Priority (Phase 0 Completion)

1. **Create `config/README.md`** documenting:
   - Which template to use when
   - Configuration hierarchy/precedence
   - How to validate configs

2. **Create `scripts/tools/config_doctor.py`** (mentioned in work plan):
   - Validate YAML/JSON syntax
   - Check for required fields
   - Detect value out-of-range errors
   - Compare actual .env against template (flag missing/extra vars)

3. **Standardize risk config format**:
   - Choose YAML or JSON (recommend YAML for comments)
   - Consolidate `spot_top10.{yaml,json}` → pick one
   - Document format choice

4. **Add ENV to production template**:
   ```bash
   ENV=production
   ```

5. **Improve sandbox example passwords**:
   ```bash
   ADMIN_PASSWORD=change_me_please
   ```

---

### Medium Priority (Phase 1)

6. **Add missing .env template sections**:
   - Backup/recovery config
   - Database config (if applicable)
   - Staging environment template

7. **Create Pydantic validation schemas**:
   - `RiskConfigSchema`
   - `ProfileConfigSchema`
   - `EnvConfigSchema`
   - Integrate into `config_doctor.py`

8. **Document all risk config options in .env templates**:
   - Add commented examples for day/night leverage
   - Include all optional flags from risk YAML files

9. **Align dev risk limits closer to prod** (for bug detection):
   ```json
   // dev_dynamic.json - suggested changes
   "max_exposure_pct": 0.75,  // Down from 1.00
   "max_position_pct_per_symbol": 0.50,  // Down from 1.00
   ```

---

### Low Priority (Phase 2)

10. **Create configuration versioning**:
    - Add `config_version: "1.0"` to all config files
    - Track breaking changes to config schema

11. **Add config migration scripts**:
    - `scripts/config_migrate_v1_to_v2.py` (when schema changes)

12. **Centralize config defaults**:
    - `config/defaults.yaml` as single source of truth
    - Templates reference defaults + override specific values

---

## Config Doctor Script Design (Phase 0 Deliverable)

### Proposed Script: `scripts/tools/config_doctor.py`

**Features**:

```python
#!/usr/bin/env python3
"""
Config Doctor - Validate GPT-Trader configuration files

Usage:
  python scripts/tools/config_doctor.py --check all
  python scripts/tools/config_doctor.py --check env
  python scripts/tools/config_doctor.py --check risk
  python scripts/tools/config_doctor.py --compare .env config/environments/.env.template
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List

def check_env_file(env_path: Path, template_path: Path) -> List[str]:
    """Compare actual .env against template, flag missing/extra vars."""
    issues = []

    # Load both files
    env_vars = parse_env(env_path)
    template_vars = parse_env(template_path)

    # Find missing required vars
    missing = set(template_vars.keys()) - set(env_vars.keys())
    if missing:
        issues.append(f"Missing variables: {', '.join(missing)}")

    # Find extra unexpected vars
    extra = set(env_vars.keys()) - set(template_vars.keys())
    if extra:
        issues.append(f"Extra variables (not in template): {', '.join(extra)}")

    return issues

def validate_risk_config(risk_path: Path) -> List[str]:
    """Validate risk config against schema."""
    issues = []

    # Load config
    if risk_path.suffix == '.yaml':
        with open(risk_path) as f:
            config = yaml.safe_load(f)
    elif risk_path.suffix == '.json':
        with open(risk_path) as f:
            config = json.load(f)
    else:
        return [f"Unknown format: {risk_path.suffix}"]

    # Validate required fields
    required = ['max_leverage', 'daily_loss_limit', 'max_exposure_pct']
    for field in required:
        if field not in config:
            issues.append(f"Missing required field: {field}")

    # Validate ranges
    if 'max_leverage' in config:
        if not (1 <= config['max_leverage'] <= 20):
            issues.append(f"max_leverage out of range: {config['max_leverage']} (expected 1-20)")

    if 'max_exposure_pct' in config:
        if not (0 <= config['max_exposure_pct'] <= 1):
            issues.append(f"max_exposure_pct out of range: {config['max_exposure_pct']} (expected 0-1)")

    return issues

def main():
    parser = argparse.ArgumentParser(description='Validate GPT-Trader configs')
    parser.add_argument('--check', choices=['all', 'env', 'risk', 'profiles'], default='all')
    parser.add_argument('--compare', nargs=2, metavar=('ENV', 'TEMPLATE'))

    args = parser.parse_args()

    issues_found = False

    if args.compare:
        env_path, template_path = args.compare
        issues = check_env_file(Path(env_path), Path(template_path))
        if issues:
            print(f"Issues in {env_path}:")
            for issue in issues:
                print(f"  - {issue}")
            issues_found = True

    elif args.check in ['all', 'risk']:
        risk_dir = Path('config/risk')
        for risk_file in risk_dir.glob('*.{yaml,json}'):
            issues = validate_risk_config(risk_file)
            if issues:
                print(f"Issues in {risk_file}:")
                for issue in issues:
                    print(f"  - {issue}")
                issues_found = True

    # ... more checks

    if not issues_found:
        print("✅ All configurations valid!")
        return 0
    else:
        print("\n⚠️  Configuration issues detected. Please fix before deploying.")
        return 1

if __name__ == '__main__':
    exit(main())
```

---

## Appendix: Configuration File Listing

```bash
$ find config/ -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name ".env*" \) | sort
config/.pre-commit-config.yaml
config/acceptance_tuning.yaml
config/adaptive_portfolio_aggressive.yaml
config/adaptive_portfolio_config.yaml
config/adaptive_portfolio_conservative.yaml
config/backtest_config.yaml
config/brokers/coinbase_perp_specs.yaml
config/database.yaml
config/environments/.env.production
config/environments/.env.template
config/live_trade_config.yaml
config/ml_strategy_config.yaml
config/position_sizing_config.yaml
config/profiles/canary.yaml
config/profiles/dev_entry.yaml
config/profiles/spot.yaml
config/risk/coinbase_perps.prod.yaml
config/risk/dev_dynamic.json
config/risk/spot_top10.json
config/risk/spot_top10.yaml
config/stage1_scaleup.yaml
config/stage2_scaleup.yaml
config/stage3_scaleup.yaml
config/system_config.yaml

$ ls -1 .env*
.env.sandbox.example
```

**Total**: 24 config files + 1 root template

---

**Next Steps**: Implement `config_doctor.py` script (Phase 0 deliverable from work plan)
