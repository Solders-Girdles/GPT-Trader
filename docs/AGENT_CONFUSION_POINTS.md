# Agent Confusion Points for GPT-Trader Repository

This document highlights key areas of confusion for AI agents working with the GPT-Trader repository. The codebase has undergone significant architectural changes and contains mixed messaging that can mislead agents.

## 🚨 Critical Confusion Points

### 1. Spot vs Perpetuals Confusion
**Primary Issue**: The bot is now **spot-first** but extensive documentation and code still references perpetuals as if they're actively enabled.

**Confusing Elements**:
- README mentions "perpetual futures logic remains in the tree" but requires INTX gate
- Architecture doc describes perps components as "future-ready" without clear distinction
- CLI is called `coinbase-trader` but primarily does spot trading
- Multiple references to "perps" throughout codebase that are actually dormant

**Reality Check**:
- Spot trading: ✅ Active (BTC-USD, ETH-USD, etc.)
- Perpetuals: ⚠️ Code exists but requires `COINBASE_ENABLE_DERIVATIVES=1` + INTX access
- Default behavior: Spot-only with mock broker for dev profile

### 2. Legacy vs Active Code Confusion
**Major Issue**: Extensive legacy code and documentation mixed with active components.

**Confusing Elements**:
- Legacy bundles exist but the in-tree code is gone, so outdated docs can still point to removed paths.
- Multiple "legacy recovery" instructions were scattered across docs.
- Older test counts included legacy suites.

**Reality Check**:
- Active stack: `src/bot_v2/**` (all runtime code)
- Legacy artifacts: packaged under `var/legacy/legacy_bundle_*.tar.gz` (see `docs/archive/legacy_recovery.md`)
- Tests: `poetry run pytest --collect-only` currently reports 1484 collected / 1483 selected tests (1 deselected legacy placeholder)

### 3. Documentation Inconsistencies
**Critical Issue**: Documentation contains contradictory information.

**Confusing Elements**:
- `docs/ARCHITECTURE.md` describes archived slices as part of current system
- `docs/agents/Agents.md` references experimental slices that were removed
- Multiple guides contain restoration steps for legacy monitoring
- `docs/reference/system_capabilities.md` now points to an archived December 2024 snapshot

**Reality Check**:
- Always verify file modification dates
- Cross-reference with actual code structure
- Prefer recent documentation over older references

### 4. Configuration and Environment Confusion
**Issue**: Complex configuration with mixed legacy and modern settings.

**Confusing Elements**:
- `config/system_config.yaml` contains outdated settings (yfinance, mock data providers)
- Multiple authentication methods (HMAC vs JWT) with unclear usage
- Environment variables like `COINBASE_ENABLE_DERIVATIVES` that default to disabled
- Debug flags scattered throughout (`COINBASE_TRADER_DEBUG`, legacy `PERPS_DEBUG`, `LOG_LEVEL`)

**Reality Check**:
- Spot trading uses HMAC auth (API key/secret)
- Perps requires CDP auth (JWT) + INTX access
- Most config files are profile-specific under `config/`

### 5. CLI and Entry Point Confusion
**Issue**: Multiple CLI tools with unclear purposes.

**Confusing Elements**:
- References to retired PoC CLIs (`gpt-trader-next`) still appear in older docs.
- Command aliases and shims that may not work.
- Multiple entry points mentioned historically.

**Reality Check**:
- Primary CLI: `poetry run coinbase-trader` (legacy alias `poetry run perps-bot` remains available for older scripts)
- Legacy CLI prototypes live only in the legacy bundle/tag; do not expect them in the workspace.

### 6. Testing and Validation Confusion
**Issue**: Complex test structure with mixed active/legacy tests.

**Confusing Elements**:
- Tests deselected/skipped without clear reasons
- Legacy test coverage kept alive for PoC code
- Multiple test commands with different purposes
- Integration tests archived but referenced in docs

**Reality Check**:
- Active tests: `poetry run pytest -q`
- Test discovery: `poetry run pytest --collect-only`
- Legacy tests: Skipped by markers

### 7. Naming and Terminology Confusion
**Issue**: Inconsistent naming conventions and terminology.

**Confusing Elements**:
- `qty` vs `quantity` (legacy aliases removed but may still appear)
- "perps" naming throughout spot-first system
- Mixed abbreviations and full terms
- Legacy naming patterns in some areas

**Reality Check**:
- Use `quantity` not `qty`
- `coinbase-trader` is the canonical CLI/runtime name (legacy `perps-bot` remains as an alias during migration)
- Check naming standards in `docs/agents/naming_standards_outline.md`

### 8. Operational Confusion
**Issue**: Unclear operational procedures and status.

**Confusing Elements:**
- Multiple monitoring approaches (active vs legacy)
- Unclear production vs sandbox boundaries
- Mixed deployment instructions
- Emergency procedures referencing retired components

**Reality Check:**
- Production: Use `canary` or `prod` profiles
- Development: Use `dev` profile with `--dev-fast`
- Sandbox: Not recommended (API diverges)

### 9. Document Trustworthiness Confusion
**Issue**: Inability to trust documents due to mixed currency and inconsistent authority levels.

**Specific Confusing Elements:**
- **Dated References**: Documents like `docs/archive/2024/system_capabilities.md` appear legitimate until reading fine-print warnings
- **Authority Hierarchy**: Unclear which documents are authoritative vs historical
- **Partial Updates**: Some docs mix current facts with obsolete information
- **Missing Dates**: No standardized "last-updated" headers across all documentation

**Common Misreadings:**
```markdown
# ❌ CONFUSING: This looks current but is December 2024
# Source: docs/archive/2024/system_capabilities.md
- **Active Tests**: 220 tests - 100% pass rate ✅
- Perps: BTC-PERP, ETH-PERP (perpetual futures)

# ✅ REALITY: Current state is different
- **Active Tests**: 1484 collected, 1483 selected (1 deselected legacy placeholder)
- Spot: BTC-USD, ETH-USD via Coinbase Advanced Trade
```

**Authority Hierarchy for Agents:**
1. **✅ PRIMARY**: `README.md`, `docs/ARCHITECTURE.md` (verify last-updated status)
2. **⚠️ SECONDARY**: Agent-specific guides (`docs/agents/`)
3. **❌ AVOID**: Documents with "archived", "historical", or before 2025 dates
4. **🔍 VERIFY**: Always cross-reference before acting

**Reality Check:**
- **Verification Required**: Check file modification dates before trusting any document
- **Cross-Reference**: Compare 3+ sources when uncertain
- **Current Standard**: Spot-first with INTX-gated perps (not perps-first)

## 🔍 Agent Verification Checklist

Before making changes, agents should:

### Pre-Task Verification
- [ ] Check if feature is spot-only or perps-gated
- [ ] Verify which codebase section is active vs legacy
- [ ] Confirm documentation recency (check file dates)
- [ ] Test with dev profile first: `poetry run coinbase-trader run --profile dev --dev-fast`
- [ ] Run test discovery: `poetry run pytest --collect-only` (1484 collected / 1483 selected / 1 deselected)

### Code Navigation
- [ ] Use `src/bot_v2/` imports for active code
- [ ] Avoid `archived/` unless specifically needed
- [ ] Check coordinator pattern usage (new architecture)
- [ ] Verify configuration actually exists and is used

### Testing Approach
- [ ] Add tests to `tests/unit/bot_v2/` for new features
- [ ] Run `poetry run pytest -q` for regression testing
- [ ] Check for skipped/legacy tests in output
- [ ] Verify test counts match expectations (1484 collected / 1483 selected / 1 deselected)

### Documentation Updates
- [ ] Update relevant docs if behavior changes
- [ ] Sync agent guides after architectural changes
- [ ] Note INTX gating for perps-related changes
- [ ] Remove references to archived components

## 🚨 Common Pitfalls

1. **Assuming perps are enabled** - They're gated behind INTX access
2. **Using legacy imports** - Stick to `src/bot_v2/` paths
3. **Trusting outdated docs** - Always verify with current code
4. **Missing profile context** - Dev uses mock broker, prod uses live
5. **Ignoring test selection** - Many tests are deselected/skipped
6. **Configuration drift** - Some configs contain legacy settings

## 📚 Quick Reference Commands

```bash
# Verify current state
poetry run pytest --collect-only
poetry run coinbase-trader run --profile dev --dev-fast

# Testing
poetry run pytest -q
poetry run pytest tests/unit/bot_v2/ -q

# Account verification
poetry run coinbase-trader account snapshot

# Metrics
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/coinbase_trader/prod/metrics.json
```

## 🔄 Architecture Evolution

The codebase is transitioning from:
- Legacy: Monolithic `src/bot` + experimental slices
- Current: Vertical slices `src/bot_v2/` + coordinators
- Future: Cleaner separation of concerns

Always check the modification date and context before trusting any documentation or code patterns.
