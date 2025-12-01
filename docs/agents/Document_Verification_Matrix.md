# Document Verification Matrix

This matrix helps developers and AI agents determine which documentation sources to trust and how to verify information.

## Trust Levels

| Level | Description | Action Required |
|-------|-------------|-----------------|
| **High** | Core architecture docs, actively maintained | Trust but verify against code |
| **Medium** | Feature guides, may have minor drift | Cross-reference with source code |
| **Low** | Older docs, potentially stale | Verify all claims against current code |

## Document Trust Ratings

### High Trust (Actively Maintained)

| Document | Last Verified | Notes |
|----------|---------------|-------|
| `docs/ARCHITECTURE.md` | 2025-12-01 | Core system architecture |
| `docs/TUI_ROADMAP.md` | 2025-12-01 | TUI development roadmap |
| `docs/guides/backtesting.md` | 2025-12-01 | Backtesting framework guide |
| `src/gpt_trader/features/brokerages/coinbase/README.md` | 2025-12-01 | Coinbase integration |
| `.claude/CLAUDE.md` | Current | Project instructions for AI agents |

### Medium Trust (Generally Accurate)

| Document | Notes |
|----------|-------|
| `docs/guides/production.md` | 2025-12-01 | Production deployment guide |
| `docs/guides/testing.md` | 2025-12-01 | Testing conventions |
| `docs/RISK_INTEGRATION_GUIDE.md` | 2025-12-01 | Risk management integration |
| `docs/reference/trading_logic_perps.md` | 2025-12-01 | Perpetuals trading logic |
| `docs/TUI_GUIDE.md` | 2025-12-01 | TUI User Guide |

### Low Trust (May Be Stale)

| Document | Notes |
|----------|-------|
| `docs/TRAINING_GUIDE.md` | 2025-12-01 | Updated links and TUI reference |
| `docs/reference/coinbase_complete.md` | 2025-12-01 | Consolidated reference |

## Verification Workflow

1. **Before relying on documentation:**
   - Check the `last-updated` field if present
   - Cross-reference key claims against source code
   - For API details, verify against actual implementation

2. **When in doubt:**
   - Source code is the ultimate truth
   - Use `grep` or IDE search to verify module locations
   - Check test files for current usage patterns

3. **Reporting stale docs:**
   - Note discrepancies in PR descriptions
   - Update docs as part of feature work

## Key Source of Truth Files

| Topic | Source File |
|-------|-------------|
| Broker interface | `src/gpt_trader/features/brokerages/core/interfaces.py` |
| REST service | `src/gpt_trader/features/brokerages/coinbase/rest_service.py` |
| Risk management | `src/gpt_trader/features/live_trade/risk/manager.py` |
| Configuration | `src/gpt_trader/orchestration/configuration.py` |
| Execution engine | `src/gpt_trader/orchestration/live_execution.py` |

---

*Last updated: 2025-11-25*
