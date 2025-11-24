# AI Agent Documentation Hub

This directory contains documentation specifically for AI agents working with the GPT-Trader repository.

## 📋 Essential Reading Order

### 1. Start Here
- **[AGENT_QUICK_REFERENCE.md](../AGENT_QUICK_REFERENCE.md)** - Condensed reference for daily work
- **[Agents.md](Agents.md)** - Shared playbook for all assistants

### 2. Deep Dive
- **[AGENT_CONFUSION_POINTS.md](../AGENT_CONFUSION_POINTS.md)** - Known issues and pitfalls
- **[CLAUDE.md](CLAUDE.md)** - Claude-specific guidance
- **[Gemini.md](Gemini.md)** - Gemini-specific guidance

### 3. Reference Materials
- **[naming_inventory.md](naming_inventory.md)** - Current naming conventions
- **[naming_standards_outline.md](naming_standards_outline.md)** - Naming rules and guidelines

## 🎯 Current System State (2025-10)

**Active Architecture**: Spot-first trading bot
- **Primary CLI**: `poetry run coinbase-trader`
- **Codebase**: `src/gpt_trader/` (vertical slices + coordinators)
- **Tests**: 1483 active tests (1484 collected; 1 deselected legacy placeholder)
- **Perpetuals**: Code exists but requires INTX access

**Legacy Components**:
- `var/legacy/legacy_bundle_latest.tar.gz` – archived experimental slices and the retired PoC CLI (`docs/archive/legacy_recovery.md`)
- Legacy documentation snapshots in `docs/archive/`

## 🚨 Critical Gotchas

1. **Don't assume perps are enabled** - They're gated behind `COINBASE_ENABLE_DERIVATIVES=1` + INTX access
2. **Use `src/gpt_trader/` imports only** - Legacy paths will cause import errors
3. **Verify documentation recency** - Many docs contain outdated information
4. **Test with dev profile first** - Use `--dev-fast` for safe mock trading
5. **Check test selection** - Many tests are deselected/skipped

## 🔄 Agent Workflow

### Before Starting
```bash
poetry install                                    # Fresh dependencies
poetry run pytest --collect-only                 # Verify test state
poetry run coinbase-trader run --profile dev --dev-fast # Quick smoke test
```

### During Development
- Use `src/gpt_trader/` imports exclusively
- Test with dev profile before production profiles
- Add tests to `tests/unit/gpt_trader/`
- Run `poetry run pytest -q` regularly

### Before Finishing
- Update relevant documentation
- Note INTX gating for perps work
- Remove references to archived components
- Verify test counts remain stable

## 📚 Quick Commands

```bash
# Development
poetry run coinbase-trader run --profile dev --dev-fast
poetry run pytest -q

# Operations
poetry run coinbase-trader account snapshot
poetry run coinbase-trader treasury convert --from USD --to USDC --amount 1000

# Monitoring
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/coinbase_trader/prod/metrics.json
```

## 🔍 Debugging

```bash
# Enable debug logging
export PERPS_DEBUG=1
export LOG_LEVEL=DEBUG

# Check system state
poetry run coinbase-trader account snapshot
tail -f var/logs/coinbase_trader.log
```

## 📞 Getting Help

1. Check [AGENT_CONFUSION_POINTS.md](../AGENT_CONFUSION_POINTS.md) for known issues
2. Run test discovery to verify system state
3. Use dev profile for safe testing
4. Check file modification dates for recency
5. Refer to agent-specific guides

---

*Last updated: 2025-10-18 | See [AGENT_CONFUSION_POINTS.md](../AGENT_CONFUSION_POINTS.md) for latest issues*
