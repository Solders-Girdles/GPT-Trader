# Risk Configuration

## Overview

Risk configuration for the trading bot uses **environment variables as the primary method**, with an optional JSON file override via `RISK_CONFIG_PATH`.

**Current Config**: `dev_dynamic.json` - Working JSON risk config for development/testing.

## Configuration Methods (Priority Order)

1. **RISK_CONFIG_PATH Override** (Highest priority)
   ```bash
   export RISK_CONFIG_PATH=config/risk/dev_dynamic.json
   ```
   - Must be valid JSON (YAML not supported - `RiskConfig.from_json()` uses `json.load()`)
   - If file exists and loads successfully, overrides all env vars

2. **Environment Variables** (Default method)
   ```bash
   export RISK_MAX_LEVERAGE=3
   export RISK_DAILY_LOSS_LIMIT=100        # USD amount
   export RISK_MAX_EXPOSURE_PCT=0.80       # 80% exposure
   # ... (see full list below)
   ```

3. **Fallback**: Hardcoded defaults in `RiskConfig` class

## Available Environment Variables

### Core Leverage Controls
- `RISK_MAX_LEVERAGE` - Global leverage cap (default: 5)
- `RISK_LEVERAGE_MAX_PER_SYMBOL` - Per-symbol leverage caps (e.g., "BTC-PERP:10,ETH-PERP:8")

### Time-Based Leverage (UTC)
- `RISK_DAYTIME_START_UTC` - Start of day hours (e.g., "09:00")
- `RISK_DAYTIME_END_UTC` - End of day hours (e.g., "17:00")
- `RISK_DAY_LEVERAGE_MAX_PER_SYMBOL` - Daytime per-symbol leverage caps
- `RISK_NIGHT_LEVERAGE_MAX_PER_SYMBOL` - Nighttime per-symbol leverage caps
- `RISK_DAY_MMR_PER_SYMBOL` - Daytime maintenance margin rates per symbol
- `RISK_NIGHT_MMR_PER_SYMBOL` - Nighttime maintenance margin rates per symbol

### Loss Limits
- `RISK_DAILY_LOSS_LIMIT` - Max daily loss in **USD** (default: 100) - NOT a percentage!

### Exposure Controls
- `RISK_MAX_EXPOSURE_PCT` - Max portfolio exposure as decimal (e.g., 0.80 = 80%)
- `RISK_MAX_POSITION_PCT_PER_SYMBOL` - Max per-symbol position as decimal (e.g., 0.20 = 20%)
- `RISK_MAX_NOTIONAL_PER_SYMBOL` - Hard per-symbol USD ceilings (e.g., "BTC-PERP:50000,ETH-PERP:30000")

### Liquidation Safety
- `RISK_MIN_LIQUIDATION_BUFFER_PCT` - Min buffer before liquidation as decimal (default: 0.15 = 15%)
- `RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION` - Enable pre-trade liquidation projection (true/false)
- `RISK_DEFAULT_MMR` - Default maintenance margin rate when exchange doesn't provide (default: 0.005)

### Market Impact
- `RISK_ENABLE_MARKET_IMPACT_GUARD` - Enable market impact checks (true/false)
- `RISK_MAX_MARKET_IMPACT_BPS` - Max market impact in basis points

### Slippage Protection
- `RISK_SLIPPAGE_GUARD_BPS` - Slippage guard in basis points (default: 50 = 0.50%)

### Dynamic Position Sizing
- `RISK_ENABLE_DYNAMIC_POSITION_SIZING` - Enable dynamic sizing (true/false)
- `RISK_POSITION_SIZING_METHOD` - Method to use (e.g., "kelly", "fixed")
- `RISK_POSITION_SIZING_MULTIPLIER` - Multiplier for position size

### Emergency Controls
- `RISK_KILL_SWITCH_ENABLED` - Global trading halt (true/false, default: false)
- `RISK_REDUCE_ONLY_MODE` - Only allow reducing positions (true/false, default: false)

### Mark Price Staleness
- `RISK_MAX_MARK_STALENESS_SECONDS` - Max age of mark price in seconds

### Circuit Breakers
- `RISK_ENABLE_VOLATILITY_CB` - Enable volatility circuit breaker (true/false)
- `RISK_MAX_INTRADAY_VOL` - Max intraday volatility threshold
- `RISK_VOL_WINDOW_PERIODS` - Volatility calculation window periods
- `RISK_CB_COOLDOWN_MIN` - Circuit breaker cooldown in minutes
- `RISK_VOL_WARNING_THRESH` - Volatility warning threshold
- `RISK_VOL_REDUCE_ONLY_THRESH` - Volatility threshold for reduce-only mode
- `RISK_VOL_KILL_SWITCH_THRESH` - Volatility threshold for kill switch

### Fee Configuration (separate)
- `FEE_BPS_BY_SYMBOL` - Per-symbol fee bps (e.g., "BTC-PERP:6,ETH-PERP:8")

## How It Works

### Runtime Flow
```python
# runtime_coordinator.py
1. Check RISK_CONFIG_PATH env var
   └─ If set and file exists → RiskConfig.from_json(path)

2. Fallback to profile-based path (SPOT/DEV/DEMO only)
   └─ DEFAULT_SPOT_RISK_PATH → config/risk/dev_dynamic.json

3. Ultimate fallback
   └─ RiskConfig.from_env() → reads RISK_* env vars

4. Emergency fallback
   └─ RiskConfig() with hardcoded defaults
```

### Config Override
```python
# Example: Override max leverage from bot config
if bot.config.max_leverage:
    risk_config.max_leverage = int(bot.config.max_leverage)
```

## Migration Notes (2025-10-06)

**Removed**:
- `coinbase_perps.prod.yaml` - YAML format not supported by `RiskConfig.from_json()`
- `spot_top10.yaml` - Wrong file extension (code expected .json)

**Why**: `RiskConfig.from_json()` uses `json.load()` which fails on YAML. Files were never functional.

**Retained**:
- `dev_dynamic.json` - Proven working JSON config for dev/test environments

## Creating New Risk Configs

If you need a custom risk config file:

1. **Copy the template**:
   ```bash
   cp config/risk/dev_dynamic.json config/risk/my_config.json
   ```

2. **Edit as JSON** (not YAML):
   ```json
   {
     "max_leverage": 3,
     "daily_loss_limit": 100,
     "max_exposure_pct": 0.80
   }
   ```
   **Note**: `daily_loss_limit` is USD amount (100 = $100), not a percentage!

3. **Use via env var**:
   ```bash
   export RISK_CONFIG_PATH=config/risk/my_config.json
   ```

## Recommendations

### For Development
```bash
# Use dev_dynamic.json with conservative limits
export RISK_CONFIG_PATH=config/risk/dev_dynamic.json
```

### For Production
```bash
# Use env vars for explicit control
export RISK_MAX_LEVERAGE=3
export RISK_DAILY_LOSS_LIMIT=100        # USD amount, NOT percentage
export RISK_MAX_EXPOSURE_PCT=0.80       # 80% portfolio exposure
export RISK_SLIPPAGE_GUARD_BPS=30       # 30 bps = 0.30%
```

### First-Run Values
- `daily_loss_limit`: 50-100 (USD) while validating stability
- `slippage_guard_bps`: 30 (0.30%) - tighten if seeing rejections
- `max_leverage`: Start at 2-3x, increase gradually
- `max_exposure_pct`: 0.60-0.80 (60-80% of portfolio)

## Notes

- **JSON only**: YAML files will fail with JSON parse errors
- **Env vars override file**: If `RISK_CONFIG_PATH` fails to load, falls back to env vars
- **Exchange rules change**: These are YOUR upper bounds, exchange may have stricter limits
- **Document changes**: Add inline comments to JSON files with rationale
