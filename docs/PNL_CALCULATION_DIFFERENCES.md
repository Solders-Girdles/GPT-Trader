# P&L Calculation Differences: Our System vs Coinbase

This document describes the known differences between our P&L calculations and what Coinbase reports through their API.

## Overview

When reconciling P&L between our internal calculations and Coinbase's reported values, we've identified several areas where calculations may differ. Understanding these differences is crucial for accurate position tracking and risk management.

## Key Differences

### 1. Funding Payments

**Our System:**
- Tracks funding payments separately in `position.funding_paid`
- Maintains distinct `realized_pnl` and `funding_paid` fields
- Funding is subtracted from total P&L when calculating net results

**Coinbase:**
- Often includes funding payments in the `realized_pnl` field
- May not provide separate funding tracking in position endpoints
- CFM (Cross Futures Margin) endpoints may combine funding with realized P&L

**Reconciliation:**
```python
# Our total economic impact
our_total = position.realized_pnl - position.funding_paid

# Coinbase total (funding included)
cb_total = coinbase_position.realized_pnl

# These should match
assert abs(our_total - cb_total) < tolerance
```

### 2. Fee Handling

**Our System:**
- Currently does not include trading fees in P&L calculations
- Fees would need to be tracked separately if required

**Coinbase:**
- May include fees in realized P&L
- Maker/taker fees affect the actual P&L realized

**Impact:**
- Our P&L may appear higher than Coinbase's by the amount of fees paid
- For accurate reconciliation, fees should be subtracted from our calculations

### 3. Mark Price Updates

**Our System:**
- Updates mark prices when we explicitly call `update_marks()`
- May have stale prices if not updated frequently

**Coinbase:**
- Uses real-time mark prices from their matching engine
- Always has the most current mark price for unrealized P&L

**Best Practice:**
- Fetch current mark prices from Coinbase before any P&L comparison
- Update our tracker with Coinbase's mark prices for consistency

### 4. Weighted Average Entry Price

**Our System:**
```python
# Precise weighted average calculation
new_avg = (old_avg * old_qty + new_price * new_qty) / total_qty
```

**Coinbase:**
- Should use the same weighted average formula
- May have slight rounding differences due to decimal precision

**Validation:**
- Both should arrive at the same weighted average within 0.01% tolerance

### 5. Position Flips

**Our System:**
- Handles position flips (long to short) in a single trade
- Calculates realized P&L for the closed portion
- Opens new position with remaining quantity

**Coinbase:**
- Should handle similarly but may report differently
- Check if they report two separate trades or one combined

### 6. Decimal Precision

**Our System:**
- Uses Python's `Decimal` type for arbitrary precision
- Maintains full precision throughout calculations

**Coinbase:**
- Returns values as strings that we convert to Decimal
- May round to specific decimal places (typically 2-8 decimals)

**Reconciliation Tolerance:**
- Use tolerance of max(0.01% of value, $0.10) for comparisons

## API Endpoints for P&L Data

### CFM Positions Endpoint
```
GET /api/v3/brokerage/cfm/positions/{product_id}
```

Returns:
- `unrealized_pnl`: Current unrealized P&L
- `realized_pnl`: Realized P&L (may include funding)
- `mark_price`: Current mark price
- `entry_price`: Weighted average entry price

### List Fills Endpoint
```
GET /api/v3/brokerage/orders/historical/fills
```

Use to reconstruct position history and calculate P&L from trades.

### Portfolio Endpoint
```
GET /api/v3/brokerage/portfolios/{portfolio_uuid}
```

May provide aggregated P&L across all positions.

## Reconciliation Process

1. **Fetch Current Positions from Coinbase**
   ```python
   positions = adapter.list_positions()
   cfm_data = adapter.client.cfm_position(product_id)
   ```

2. **Update Mark Prices in Our Tracker**
   ```python
   for pos in positions:
       tracker.update_marks({pos.symbol: pos.mark_price})
   ```

3. **Compare P&L Values**
   ```python
   tolerance = max(Decimal('0.10'), abs(cb_value) * Decimal('0.001'))
   if abs(our_value - cb_value) > tolerance:
       log_discrepancy(...)
   ```

4. **Account for Known Differences**
   - Add/subtract funding as needed
   - Adjust for fees if tracked
   - Use Coinbase's mark price

## Testing Strategy

### Unit Tests
- Test our calculations with deterministic data
- Verify calculation correctness independent of Coinbase

### Integration Tests
- Mock Coinbase responses with realistic data
- Test reconciliation logic with known differences

### Live Reconciliation
- Periodically reconcile PnL against Coinbase statements using the EventStore
  snapshots. (The historical `scripts/reconcile_pnl_with_coinbase.py` helper
  was removed during the 2025 cleanup.)
- Monitor for unexpected discrepancies
- Alert if differences exceed thresholds

## Common Issues and Solutions

### Issue: P&L Drift Over Time
**Cause:** Not updating mark prices frequently enough
**Solution:** Update marks at least every 30 seconds during active trading

### Issue: Realized P&L Mismatch
**Cause:** Funding payments included in Coinbase's realized P&L
**Solution:** Track funding separately and combine for comparison

### Issue: Missing Positions
**Cause:** Not tracking all products or missing fills
**Solution:** Fetch complete fill history and rebuild positions

### Issue: Large Discrepancies
**Cause:** Calculation bug or missed trades
**Solution:** Rebuild from fills and validate each trade's P&L

## Monitoring and Alerts

Set up alerts for:
- Discrepancies > 1% or $100 (whichever is smaller)
- Missing positions (in Coinbase but not our tracker)
- Stale mark prices (>60 seconds old)
- Unusual funding rates

## Future Improvements

1. **Real-time Reconciliation**
   - WebSocket feed for live position updates
   - Continuous P&L validation

2. **Fee Integration**
   - Track maker/taker fees
   - Include in P&L calculations

3. **Historical Analysis**
   - Store reconciliation reports
   - Analyze patterns in discrepancies

4. **Automated Corrections**
   - Auto-adjust for known differences
   - Self-healing position state

## Conclusion

While our P&L calculations are mathematically correct, differences with Coinbase's reported values are expected due to:
- Funding payment handling
- Fee inclusion
- Mark price timing
- Decimal rounding

The key is to understand these differences and account for them in reconciliation. Always treat Coinbase's values as the source of truth for actual P&L.
