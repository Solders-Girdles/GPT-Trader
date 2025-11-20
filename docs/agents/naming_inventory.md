# Naming Inventory Summary

**Last audit: 2025-11-20**

Total findings: **264 hits** across the codebase.

- **qty**: 129
- **utils**: 81
- **svc**: 30
- **cfg**: 7
- **calc**: 13

## Regenerate

```bash
python scripts/agents/naming_inventory.py \
    --summary docs/agents/naming_inventory.md \
    --json var/agents/naming_inventory.json
```

## Priority Areas

### 1. Quantity Naming (qty → quantity) - 129 hits

**Risk & Execution Layer:**
- `src/bot_v2/features/live_trade/risk/manager/validation.py:17-220` - qty vs quantity inconsistency across validation logic
- `src/bot_v2/features/live_trade/risk/pre_trade/guards.py` - Pre-trade guard parameters use qty
- `src/bot_v2/backtesting/simulation/simulated_broker/portfolio.py:135-193` - Simulated portfolio qty fields

**Impact:** High - Affects core trading logic, risk calculations, and testing parity

### 2. Service/Config Abbreviations (svc/cfg) - 37 hits

**Test Infrastructure:**
- `tests/unit/bot_v2/features/brokerages/coinbase/rest/test_intx_portfolio.py:30-118` - svc parameters in fixtures
- Multiple test files using cfg for configuration objects

**Runtime Code:**
- `src/bot_v2/monitoring/guards/manager.py:56-116` - svc/cfg in guard manager

**Impact:** Medium - Primarily test code, but affects readability and consistency

### 3. Utils File Naming - 81 hits

**Core Utilities:**
- `src/bot_v2/utilities/import_utils.py` - Import helpers
- `src/bot_v2/utilities/async_utils.py` - Async utilities (candidate: `asyncio_helpers.py`)
- `tests/unit/bot_v2/utilities/test_async_utils_core.py` - Test files

**Domain Utilities:**
- `src/bot_v2/orchestration/shared_utils/logging_utils.py` - Logging helpers (candidate: `logging.py`)
- `src/bot_v2/features/live_trade/liquidity/utils.py` - Liquidity time helpers (candidate: `liquidity_time.py`)

**Impact:** Medium - File-level refactor with broad import impact

### 4. Domain Terminology (perps → spot-first) - Multiple surfaces

**Orchestration Layer:**
- `src/bot_v2/orchestration/perps_bot_state.py` - State management (misnomer: handles spot)
- `src/bot_v2/monitoring/domain/perps/*` - Monitoring domain package

**Configuration:**
- `config/risk/coinbase_perps.prod.yaml` - Risk config (misnomer: spot-focused)

**CLI:**
- Legacy `perps-bot` alias - Should migrate users to `coinbase-trader`

**Impact:** Low urgency, High clarity - Derivatives gated behind feature flag; spot is primary

### 5. Calculation Helpers (calc) - 13 hits

**Impact:** Low - Limited scope, straightforward rename

## Cleanup Waves

See detailed execution plan in main task description. High-level sequencing:

1. **Guardrails** - Finalize naming standards, add CI enforcement
2. **Wave 1** - Quantity naming (qty → quantity)
3. **Wave 2** - Service/config abbreviations (svc/cfg → service/config)
4. **Wave 3** - Utils file renaming (domain-specific names)
5. **Wave 4** - Domain terminology (perps → spot terminology)
6. **Wave 5** - Verification & lock-in (rerun inventory, document exceptions)

## Policy

- New code must follow `docs/naming.md` standards
- Exceptions require `# naming: allow` inline comment with justification
- Third-party API field names are exempt (document in code comments)
- Maintain backward compatibility aliases during migration periods
