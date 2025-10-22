# INTX Eligibility Verification & Fail-Closed Logic

## Overview

The INTX eligibility verification system ensures that derivatives trading only occurs when Coinbase International Exchange (INTX) permissions are properly configured and verified. It implements **fail-closed** behavior: if permissions cannot be verified, trading is blocked.

**Phase 2 Requirement**: US or INTX perps must programmatically verify eligibility and fail closed (no orders) if futures portfolio or permissions are absent.

---

## Why This Matters

Without eligibility verification:
- ❌ Bot could attempt derivatives orders without proper permissions
- ❌ Orders would be rejected by exchange (wasted API calls, confusion)
- ❌ No way to detect permission revocations during operation
- ❌ Manual verification required before every deployment

With eligibility verification:
- ✅ Bot fails startup if derivatives enabled but INTX unavailable
- ✅ Every derivatives order validated before placement
- ✅ Permission changes detected automatically
- ✅ Clear error messages guide troubleshooting

---

## Architecture

### Components

1. **IntxEligibilityChecker** - Core verification with caching
2. **IntxStartupValidator** - Validates on bot startup
3. **IntxPreTradeValidator** - Validates before every order
4. **IntxRuntimeMonitor** - Periodic re-verification

### Fail-Closed Philosophy

**Fail-closed** means: *when in doubt, don't trade*.

| Condition | Fail-Open (BAD) | Fail-Closed (GOOD) |
|-----------|-----------------|---------------------|
| Cannot verify eligibility | Allow trading anyway | **Block trading** |
| Missing portfolio UUID | Assume it's fine | **Block trading** |
| API error during check | Skip the check | **Block trading** |
| Unknown status | Allow by default | **Block trading** |

**Only ELIGIBLE status allows trading.**

---

## Quick Start

### 1. Startup Validation

```python
from bot_v2.orchestration.intx_eligibility import create_fail_closed_checker
from bot_v2.orchestration.intx_startup_validator import validate_intx_on_startup

# Create checker
eligibility_checker = create_fail_closed_checker(intx_portfolio_service)

# Validate on startup (will raise exception if not eligible)
validate_intx_on_startup(
    eligibility_checker=eligibility_checker,
    enable_derivatives=True,  # From your config
    fail_closed=True,  # Recommended for production
)
```

### 2. Pre-Trade Validation

```python
from bot_v2.features.live_trade.risk.intx_pre_trade import create_intx_validator

# Create pre-trade validator
intx_validator = create_intx_validator(
    eligibility_checker=eligibility_checker,
    event_store=event_store,
    enable_derivatives=True,
)

# Before placing each order:
try:
    intx_validator.validate_intx_eligibility(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("0.1"),
        product=product,
    )
    # Order allowed - proceed with placement
except IntxEligibilityViolation as e:
    # Order rejected - log and skip
    logger.error(f"Order rejected: {e}")
```

### 3. Runtime Monitoring

```python
from bot_v2.orchestration.intx_runtime_monitor import create_runtime_monitor

# Create monitor
monitor = create_runtime_monitor(
    eligibility_checker=eligibility_checker,
    event_store=event_store,
    enable_derivatives=True,
    check_interval_minutes=60,  # Check every hour
)

# In your main trading loop:
if monitor.check_if_due():
    monitor.run_periodic_check()  # Re-verify eligibility
```

---

## Verification Logic

### What Gets Checked

1. **API Mode**: Must be `advanced` (not `exchange`)
2. **INTX Support**: Broker must support INTX endpoints
3. **Portfolio UUID**: Must be able to resolve valid portfolio
4. **Permissions**: Must have derivatives entitlements on API key

### Verification Flow

```
START
  ↓
Check if derivatives enabled?
  ├─ No → Skip (allow spot trading)
  └─ Yes → Continue
       ↓
Check if broker supports_intx()?
  ├─ No → INELIGIBLE (wrong API mode)
  └─ Yes → Continue
       ↓
Try to resolve portfolio UUID
  ├─ Failed → INELIGIBLE (no permissions)
  ├─ None → INELIGIBLE (not enrolled)
  └─ Valid → ELIGIBLE ✅
```

### Caching Strategy

| Status | Cache TTL | Rationale |
|--------|-----------|-----------|
| ELIGIBLE | 1 hour | Minimize API calls while eligible |
| INELIGIBLE | 5 minutes | Retry sooner in case issue is fixed |
| UNKNOWN | No cache | Retry immediately |

---

## Configuration

### Environment Variables

```bash
# Required for INTX
export COINBASE_API_MODE=advanced  # MUST be 'advanced' not 'exchange'
export COINBASE_ENABLE_DERIVATIVES=1

# Optional: Override portfolio UUID
export COINBASE_INTX_PORTFOLIO_UUID=your-portfolio-uuid-here
```

### Eligibility Checker Settings

```python
IntxEligibilityChecker(
    intx_portfolio_service=service,
    cache_ttl_seconds=3600,  # 1 hour for eligible
    cache_ttl_ineligible_seconds=300,  # 5 min for ineligible
    require_portfolio_uuid=True,  # Strict requirement
)
```

### Runtime Monitor Settings

```python
IntxRuntimeMonitor(
    check_interval_minutes=60,  # Check every hour
    enable_derivatives=True,
)
```

---

## Integration Guide

### Complete Bot Integration

```python
class YourTradingBot:
    def __init__(self, config):
        # Step 1: Create INTX portfolio service
        self.intx_service = IntxPortfolioService(
            account_manager=self.account_manager,
            runtime_settings=self.runtime_settings,
        )

        # Step 2: Create eligibility checker
        self.eligibility_checker = create_fail_closed_checker(
            self.intx_service
        )

        # Step 3: Validate on startup
        validate_intx_on_startup(
            eligibility_checker=self.eligibility_checker,
            enable_derivatives=config.derivatives_enabled,
            fail_closed=True,  # Critical: fail startup if not eligible
        )

        # Step 4: Create pre-trade validator
        self.intx_validator = create_intx_validator(
            eligibility_checker=self.eligibility_checker,
            event_store=self.event_store,
            enable_derivatives=config.derivatives_enabled,
        )

        # Step 5: Create runtime monitor
        self.intx_monitor = create_runtime_monitor(
            eligibility_checker=self.eligibility_checker,
            event_store=self.event_store,
            enable_derivatives=config.derivatives_enabled,
        )

    def run(self):
        while self.running:
            # Periodic eligibility check
            if self.intx_monitor.check_if_due():
                self.intx_monitor.run_periodic_check()

            # Trading logic
            for signal in self.get_signals():
                try:
                    # Pre-trade validation
                    self.intx_validator.validate_intx_eligibility(
                        symbol=signal.symbol,
                        side=signal.side,
                        quantity=signal.quantity,
                        product=signal.product,
                    )

                    # Place order
                    self.place_order(signal)

                except IntxEligibilityViolation as e:
                    logger.warning(f"Order rejected: {e}")
                    continue
```

---

## Troubleshooting

### Startup Fails with "INTX Not Eligible"

**Symptoms**:
```
IntxStartupValidationError: INTX eligibility check failed
Status: ineligible
Error: No INTX portfolio UUID found
```

**Solutions**:

1. **Verify INTX Enrollment**:
   - Check that Coinbase account is approved for INTX
   - Complete institutional onboarding
   - Verify derivatives entitlements

2. **Check API Configuration**:
   ```bash
   # Ensure these are set correctly
   echo $COINBASE_API_MODE  # Should be 'advanced'
   echo $COINBASE_ENABLE_DERIVATIVES  # Should be '1'
   ```

3. **Test Portfolio Resolution**:
   ```python
   # In Python shell
   result = eligibility_checker.check_eligibility(force_refresh=True)
   print(result.to_dict())
   ```

4. **Override Portfolio UUID** (if needed):
   ```bash
   export COINBASE_INTX_PORTFOLIO_UUID=your-uuid-here
   ```

### Orders Rejected During Runtime

**Symptoms**:
```
IntxEligibilityViolation: INTX eligibility check failed
```

**Possible Causes**:
- API key revoked or changed
- INTX entitlements removed
- Portfolio UUID changed
- API mode reverted to 'exchange'

**Debug Steps**:
```python
# Check current status
status = eligibility_checker.check_eligibility(force_refresh=True)
print(f"Status: {status.status}")
print(f"Error: {status.error_message}")
print(f"Portfolio UUID: {status.portfolio_uuid}")

# Check monitoring stats
monitor_status = intx_monitor.get_status_summary()
print(monitor_status)

# Check eligibility change history
checker_stats = eligibility_checker.get_stats()
print(f"Eligibility changes: {len(checker_stats['eligibility_changes'])}")
```

### Permission Loss Detected

**Symptoms**:
```
⚠️  INTX PERMISSION LOSS DETECTED
Status: ineligible
```

**Actions**:
1. Check Coinbase account status immediately
2. Verify API key hasn't been revoked
3. Review account notifications/emails
4. Contact Coinbase support if unexpected

**Bot Behavior**:
- Existing positions remain open
- New orders will be rejected
- Bot continues running (doesn't crash)
- Permissions restored → trading resumes automatically

---

## Monitoring & Alerts

### Metrics Emitted

| Metric | When | Severity |
|--------|------|----------|
| `intx_eligibility_rejection` | Order rejected | Warning |
| `eligibility_status_change` | Status changed | Warning |
| `permission_loss` | Permissions lost | Critical |
| `permission_restored` | Permissions back | Info |

### Example Monitoring Query

```python
# Get all eligibility rejections in last hour
events = event_store.query(
    event_type="intx_eligibility_rejection",
    since=datetime.now() - timedelta(hours=1)
)

print(f"Rejections in last hour: {len(events)}")
```

### Recommended Alerts

1. **Critical**: Permission loss detected
   - Immediate notification
   - Check within 15 minutes

2. **Warning**: Eligibility check failures
   - Alert if > 3 failures in 10 minutes
   - May indicate API issues

3. **Info**: Eligibility status change
   - Log for audit trail
   - No immediate action needed

---

## Testing

### Test Fail-Closed Behavior

```python
# Mock ineligible service
class MockIneligibleService:
    def supports_intx(self):
        return False  # Simulate wrong API mode

    def get_portfolio_uuid(self, refresh=False):
        return None  # Simulate no portfolio

checker = create_fail_closed_checker(MockIneligibleService())

# This should fail
try:
    validate_intx_on_startup(
        eligibility_checker=checker,
        enable_derivatives=True,
        fail_closed=True,
    )
    assert False, "Should have raised exception"
except IntxStartupValidationError:
    print("✅ Fail-closed behavior working")
```

### Test Pre-Trade Rejection

```python
# Should reject derivatives orders
intx_validator = create_intx_validator(
    eligibility_checker=checker,
    event_store=event_store,
    enable_derivatives=True,
)

try:
    intx_validator.validate_intx_eligibility(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("0.1"),
        product=perp_product,
    )
    assert False, "Should have rejected"
except IntxEligibilityViolation:
    print("✅ Pre-trade rejection working")
```

---

## Phase 2 Exit Criteria Checklist

```
INTX Eligibility Verification (Workstream 2):

[ ] ✅ Eligibility checker implemented
    ├── Verifies API mode
    ├── Resolves portfolio UUID
    ├── Caches results with TTL
    └── Implements fail-closed logic

[ ] ✅ Startup validation implemented
    ├── Runs before bot starts trading
    ├── Fails startup if ineligible
    └── Provides clear error messages

[ ] ✅ Pre-trade validation implemented
    ├── Checks before every derivatives order
    ├── Rejects orders if ineligible
    ├── Emits rejection metrics
    └── Allows spot orders regardless

[ ] ✅ Runtime monitoring implemented
    ├── Periodic eligibility re-checks
    ├── Detects permission loss
    ├── Emits alerts on changes
    └── Auto-resumes if restored

[ ] ✅ Fail-closed tested
    ├── Startup fails with bad permissions
    ├── Orders rejected when ineligible
    ├── No false positives (eligible works)
    └── Spot trading unaffected

[ ] Next: Integration testing with real INTX credentials
```

---

## See Also

- [Derivatives Stress Testing](./stress_testing.md)
- [INTX Integration Guide](./intx_integration.md)
- [Production Deployment Checklist](./deployment_checklist.md)

---

## FAQ

**Q: What if I don't have INTX access yet?**
A: Disable derivatives trading (`COINBASE_ENABLE_DERIVATIVES=0`) and trade spot only until approved.

**Q: Can I bypass eligibility checks for testing?**
A: Set `fail_closed=False` in startup validator, but NEVER in production.

**Q: Does this work for US futures too?**
A: Yes, same eligibility logic applies to both INTX and US futures.

**Q: What if my portfolio UUID changes?**
A: Monitor detects change automatically and logs warning. Update env var if using override.

**Q: How often should I re-check eligibility?**
A: Default 1 hour is good. Reduce to 30 min if permissions are unstable.
