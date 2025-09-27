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


# Release Notes - v2.1.0

## 🎯 Overview
This release focuses on type consolidation and performance optimizations for the Coinbase brokerage integration, improving code maintainability and reducing API latency.

## ✨ New Features

### Type Consolidation
- **Unified Type System**: All broker integrations now use core types from `brokerages.core.interfaces`
- **Deprecated Local Types**: Removed duplicate type definitions in `live_trade/types.py`
- **Consistent Field Names**: Standardized on core Order fields (`id`, `qty`, `status`)
- **CI/CD Protection**: Added automated guards against type regression

### Performance Optimizations
- **Connection Keep-Alive**: Automatic HTTP connection reuse reduces latency by 20-40ms per request
- **Smart Backoff Jitter**: Deterministic jitter prevents thundering herd during retries
- **Enhanced Rate Limiting**: Sliding window tracking with automatic throttling at 80% threshold

## 🔄 Breaking Changes

### Type Migration Required
If your code imports from `live_trade.types`, update to use core interfaces:

```python
# Old (deprecated - will show warning)
from bot_v2.features.live_trade.types import Order, OrderSide, OrderType

# New (required)
from bot_v2.features.brokerages.core.interfaces import Order, OrderSide, OrderType
```

### Field Name Changes
| Old Field | New Field | Type Change |
|-----------|-----------|-------------|
| `order.order_id` | `order.id` | str |
| `order.quantity` | `order.qty` | int → Decimal |
| `order.order_type` | `order.type` | str → OrderType |
| `order.order_status` | `order.status` | str → OrderStatus |

## 🚀 Performance Improvements
- **15-30% throughput increase** in high-volume scenarios
- **20-40ms latency reduction** per API request with keep-alive
- **Better retry distribution** under load with jitter
- **Automatic rate limit protection** prevents API throttling

## 📝 Configuration

### New Environment Variables
Add these to your `.env` file:

```bash
# Performance settings (with recommended defaults)
COINBASE_ENABLE_KEEP_ALIVE=1      # Connection reuse (0 to disable)
COINBASE_JITTER_FACTOR=0.1        # 10% backoff jitter (0 for none)
COINBASE_RATE_LIMIT_PER_MINUTE=100  # API rate limit
COINBASE_ENABLE_THROTTLE=1        # Auto-throttle at limit
```

## 🔧 Rollback Plan

If you experience issues, you can quickly rollback performance features:

### Quick Rollback (Keep Code, Disable Features)
```bash
# In .env or environment
COINBASE_ENABLE_KEEP_ALIVE=0  # Disable connection reuse
COINBASE_JITTER_FACTOR=0      # Disable backoff jitter
```

### Full Rollback
```bash
git revert HEAD~2  # Revert last 2 commits (PR 2 and PR 3)
```

## 🧪 Testing

### Validation Commands
```bash
# Type consolidation verification
rg "from .*live_trade\.types import" tests/ --type py  # Should return nothing

# Performance tests
pytest tests/unit/bot_v2/features/brokerages/coinbase/test_performance.py

# Integration tests
pytest tests/integration/bot_v2/test_live_trade_error_handling.py

# End-to-end smoke test
python -m src.bot_v2.simple_cli broker --broker coinbase --sandbox
```

### CI/CD Checks
The following automated checks are now in place:
- ✅ No duplicate type definitions
- ✅ No deprecated imports from `live_trade.types`
- ✅ All tests use core interfaces
- ✅ Performance optimizations don't break existing functionality

## 📚 Documentation
- Updated [Coinbase README](docs/COINBASE_README.md) with performance tuning guide
- Added debugging tips for proxy/firewall issues
- Enhanced `.env.template` with performance settings

## 🐛 Bug Fixes
- Fixed ExecutionEngine using incorrect Order field names
- Resolved test failures from type mismatches
- Corrected facade imports and field access patterns

## 🔄 Migration Guide

### For Developers
1. Update all imports from `live_trade.types` to `brokerages.core.interfaces`
2. Change field accesses from local names to core names
3. Handle `qty` as Decimal instead of int
4. Test with both keep-alive enabled and disabled

### For Operations
1. Review new environment variables in `.env.template`
2. Monitor API rate limit warnings in logs
3. Check latency metrics after deployment
4. Have rollback plan ready (disable features via env vars)

## 📊 Metrics to Monitor
- API request latency (should decrease by 20-40ms)
- Rate limit warnings (should stay below 80%)
- Connection errors (should remain stable or decrease)
- Retry frequency (should have better distribution)

## 🙏 Acknowledgments
Thanks to the team for thorough code review and testing of these critical changes.

---

**Commit Range:** c261b60..HEAD
**Date:** 2025-08-30
**Next Release:** v2.2.0 (planned: WebSocket improvements)