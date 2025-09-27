# Week 3 Implementation Plan: Advanced Order Management & PnL Tracking

## Overview
Week 3 focuses on production-grade order management, impact-aware sizing, and comprehensive PnL/funding tracking.

## 1. Execution & Order Management

### 1.1 Order Types Enhancement
**File**: `src/bot_v2/features/live_trade/execution.py`

```python
class OrderTypeConfig:
    """Configuration for advanced order types."""
    
    # Limit orders
    enable_limit_orders: bool = True
    limit_price_offset_bps: Decimal = Decimal('5')  # Offset from mid
    
    # Stop orders
    enable_stop_orders: bool = True
    stop_trigger_offset_pct: Decimal = Decimal('0.02')  # 2% from entry
    
    # Stop-limit orders
    enable_stop_limit: bool = True
    stop_limit_spread_bps: Decimal = Decimal('10')
    
    # Post-only protection
    enable_post_only: bool = True
    reject_on_cross: bool = True  # Reject if post-only would cross
```

### 1.2 TIF Mapping
**Current**: Only GTC supported
**Target**: Full TIF support with validation

```python
TIF_MAPPING = {
    TimeInForce.GTC: "GOOD_TILL_CANCELLED",
    TimeInForce.IOC: "IMMEDIATE_OR_CANCEL",
    TimeInForce.FOK: None,  # Gate until confirmed supported
}
```

### 1.3 Cancel/Replace Flow
```python
async def cancel_and_replace(
    self,
    order_id: str,
    new_price: Optional[Decimal] = None,
    new_size: Optional[Decimal] = None
) -> Order:
    """Cancel and replace order atomically."""
    # Generate idempotent client_id
    client_id = f"{order_id}_replace_{int(time.time())}"
    
    # Cancel existing
    await self.cancel_order(order_id)
    
    # Place replacement with new params
    return await self.place_order(
        client_id=client_id,
        price=new_price,
        size=new_size,
        # ... other params
    )
```

## 2. Sizing & Impact

### 2.1 Impact-Aware Sizing
**File**: `src/bot_v2/features/live_trade/strategies/perps_baseline_v2.py`

```python
def calculate_impact_aware_size(
    self,
    target_notional: Decimal,
    market_snapshot: Dict[str, Any],
    max_impact_bps: Decimal
) -> Tuple[Decimal, Decimal]:
    """
    Calculate position size that respects slippage constraints.
    
    Returns:
        (adjusted_notional, expected_impact_bps)
    """
    l1_depth = market_snapshot.get('depth_l1', 0)
    l10_depth = market_snapshot.get('depth_l10', 0)
    
    # Binary search for max size within impact limit
    low, high = Decimal('0'), target_notional
    best_size = Decimal('0')
    
    while high - low > Decimal('1'):  # $1 precision
        mid = (low + high) / 2
        impact = self.estimate_impact(mid, l1_depth, l10_depth)
        
        if impact <= max_impact_bps:
            best_size = mid
            low = mid
        else:
            high = mid
    
    return best_size, self.estimate_impact(best_size, l1_depth, l10_depth)
```

### 2.2 Sizing Modes
```python
class SizingMode(Enum):
    CONSERVATIVE = "conservative"  # Downsize to fit
    STRICT = "strict"  # Reject if can't fit
    AGGRESSIVE = "aggressive"  # Allow higher impact
```

## 3. PnL & Funding Tracking

### 3.1 Integration Points
**File**: `scripts/run_perps_bot_v2.py`

```python
# In main loop
if position:
    # Calculate PnL and funding
    position_state = PositionState(
        symbol=symbol,
        avg_entry_price=position.avg_price,
        quantity=position.qty,
        side='long' if position.qty > 0 else 'short'
    )
    
    # Update with current mark
    pnl_data = position_state.update_pnl(
        mark_price=current_mark,
        funding_rate=product.funding_rate
    )
    
    # Append to event store
    event_store.append_metric(bot_id, {
        'type': 'pnl_update',
        'symbol': symbol,
        'realized_pnl': float(pnl_data['realized_pnl']),
        'unrealized_pnl': float(pnl_data['unrealized_pnl']),
        'funding_paid': float(pnl_data['funding_paid']),
        'timestamp': datetime.now().isoformat()
    })
```

### 3.2 Daily Recap Metrics
```python
def generate_daily_recap(event_store: EventStore, bot_id: str) -> Dict:
    """Generate daily performance metrics."""
    metrics = event_store.get_metrics_range(
        bot_id,
        start=datetime.now().replace(hour=0, minute=0),
        end=datetime.now()
    )
    
    return {
        'total_pnl': sum(m['realized_pnl'] for m in metrics),
        'win_rate': calculate_win_rate(metrics),
        'max_drawdown': calculate_drawdown(metrics),
        'sharpe': calculate_sharpe(metrics),
        'trades': len([m for m in metrics if m['type'] == 'trade']),
        'rejections': count_rejections(metrics)
    }
```

## 4. Sandbox E2E Validation

### 4.1 Test Scenarios
```bash
# Scenario 1: BTC-PERP Market + Limit
python scripts/run_perps_bot_v2.py \
    --symbols BTC-PERP \
    --use-limit-orders \
    --limit-offset-bps 5 \
    --max-spread-bps 10 \
    --min-depth-l1 50000

# Scenario 2: ETH-PERP with Stop-Limit
python scripts/run_perps_bot_v2.py \
    --symbols ETH-PERP \
    --enable-stops \
    --stop-pct 2 \
    --rsi-confirm \
    --liq-buffer-pct 25

# Scenario 3: Multi-symbol with strict filters
python scripts/run_perps_bot_v2.py \
    --symbols BTC-PERP,ETH-PERP \
    --max-spread-bps 5 \
    --min-vol-1m 100000 \
    --sizing-mode conservative
```

### 4.2 Validation Checklist
- [ ] Limit orders placed at correct offset from mid
- [ ] Stop orders trigger at expected levels
- [ ] Post-only orders rejected when crossing
- [ ] TIF mappings work (GTC, IOC)
- [ ] Cancel/replace maintains position intent
- [ ] PnL calculations match exchange reports
- [ ] Funding accruals tracked correctly
- [ ] Event store captures all metrics

## 5. Documentation Updates

### 5.1 Correct CLI Examples
```bash
# Correct flags (Week 2)
python scripts/run_perps_bot_v2.py \
    --max-spread-bps 5 \
    --min-depth-l1 100000 \
    --min-vol-1m 500000 \
    --rsi-confirm \
    --max-slippage-bps 10 \
    --liq-buffer-pct 25
```

### 5.2 Crossover Robustness Guide
```markdown
## Crossover Robustness Configuration

The strategy uses robust MA crossover detection to prevent false signals:

- **ma_cross_epsilon_bps**: Tolerance in basis points (default: 1)
  - Set to 0 for legacy raw comparison behavior
  - Increase to 2-3 for very noisy markets
  
- **ma_cross_confirm_bars**: Bars to confirm crossover (default: 0)
  - Set to 1-2 for additional debouncing
  - Useful in choppy/ranging markets

Example configuration:
```python
config = StrategyConfig(
    ma_cross_epsilon_bps=Decimal('2'),  # 2 bps tolerance
    ma_cross_confirm_bars=1,  # Wait 1 bar confirmation
    # ... other params
)
```
```

## Implementation Priority

1. **Day 1-2**: Order types and TIF mapping
2. **Day 2-3**: Impact-aware sizing
3. **Day 3-4**: PnL/funding integration
4. **Day 4-5**: Sandbox E2E validation
5. **Day 5**: Documentation and runbook updates

## Success Criteria

- All order types work in sandbox (market, limit, stop, stop-limit)
- TIF mappings validated (GTC, IOC)
- Impact-aware sizing prevents excessive slippage
- PnL tracking matches exchange calculations
- Funding accruals recorded correctly
- Daily recap metrics generated
- All sandbox scenarios pass validation

## Risk Considerations

1. **Order Rejection Handling**: Graceful fallback when post-only crosses
2. **TIF Compatibility**: Gate FOK until exchange confirms support
3. **Sizing Edge Cases**: Handle minimum order sizes correctly
4. **PnL Precision**: Use Decimal throughout, never float
5. **Funding Timing**: Account for timezone differences

## Testing Approach

1. Unit tests for each new component
2. Integration tests with mock broker
3. Sandbox validation with real API (small sizes)
4. Production monitoring with conservative limits

## Rollout Plan

1. **Phase 1**: Deploy with limit orders only
2. **Phase 2**: Add stop orders after validation
3. **Phase 3**: Enable impact-aware sizing
4. **Phase 4**: Full PnL/funding tracking
5. **Phase 5**: Production with all features