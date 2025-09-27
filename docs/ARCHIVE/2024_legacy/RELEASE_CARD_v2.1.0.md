---
status: deprecated
archived: 2024-12-31
reason: Pre-perpetuals documentation from Alpaca/equities era
---

# ⚠️ DEPRECATED DOCUMENT

This document is from the legacy Alpaca/Equities version of GPT-Trader and is no longer current.
The project has migrated to Coinbase Perpetual Futures.

For current documentation, see: [docs/README.md](/docs/README.md)

---


# 🚀 v2.1.0 Release Card - Quick Reference

## Ship It! 🎯

### 1️⃣ Merge & Tag (5 min)
```bash
git checkout main
git merge feat/qol-progress-logging
git tag -a v2.1.0 -m "v2.1.0: type consolidation + performance"
git push origin main --tags
```

### 2️⃣ Verify (2 min)
```bash
./scripts/merge_checklist_v2.1.0.sh      # Interactive checklist
python scripts/post_merge_monitor.py -o   # Health check
```

### 3️⃣ Smoke Test (3 min)
```bash
python -m src.bot_v2.simple_cli broker --broker coinbase --sandbox
python scripts/validate_critical_fixes.py
```

## 🔄 If Issues Arise

### Quick Disable (10 sec)
```bash
export COINBASE_ENABLE_KEEP_ALIVE=0  # Disable keep-alive
export COINBASE_JITTER_FACTOR=0      # Disable jitter
```

### Full Rollback (30 sec)
```bash
git revert HEAD~2  # Revert both PRs
git push origin main
```

## 📊 Watch For (72 hours)

| Metric | Good | Alert |
|--------|------|-------|
| Rate Limits | <80% | ≥80% |
| API Latency | -20ms | +100ms |
| WS Reconnects | <10/hr | >10/hr |
| CI Status | Green | Any Red |

## 📢 Team Message

```
v2.1.0 SHIPPED! 🎉

What's New:
• Type consolidation → use brokerages.core.interfaces
• 20-40ms faster API calls with keep-alive
• Smart retry jitter

Breaking: order.order_id → order.id, order.quantity → order.qty

Rollback: Set COINBASE_ENABLE_KEEP_ALIVE=0 if issues

Docs: RELEASE_NOTES_v2.1.0.md
```

---
**Status:** READY TO SHIP ✅
**Confidence:** HIGH 🟢
**Rollback Time:** <1 minute ⚡