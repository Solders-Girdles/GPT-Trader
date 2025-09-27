# Coinbase Perpetuals Critical Fixes

## Executive Summary
Critical implementation gaps identified that will cause production failures. These must be fixed before Phase 1 deployment.

## 🔴 CRITICAL FIXES (Must fix before ANY production use)

### 1. Missing Funding Rate Methods
**Impact**: Product enrichment crashes
**Location**: `src/bot_v2/features/brokerages/coinbase/adapter.py:391-401`

```python
# BROKEN CODE:
def _enrich_with_funding(self, product: Product) -> Product:
    response = self.client.get_funding_rate(product.symbol)  # METHOD DOESN'T EXIST!
```

**FIX**:
```python
# Option A: Remove funding enrichment until proper endpoint identified
def _enrich_with_funding(self, product: Product) -> Product:
    # TODO: Implement when Coinbase exposes funding endpoint
    return product

# Option B: Use CFM endpoints if available
def _enrich_with_funding(self, product: Product) -> Product:
    if not self.config.enable_derivatives:
        return product
    try:
        # Use CFM balance summary which includes funding info
        data = self.client.cfm_balance_summary()
        # Extract funding from response
        return product
    except:
        return product
```

### 2. WebSocket Auth Broken
**Impact**: User channel subscriptions fail
**Location**: `src/bot_v2/features/brokerages/coinbase/adapter.py:418-422`

```python
# BROKEN CODE:
auth_provider = lambda: self.client.auth  # Returns object, not auth payload!
```

**FIX**:
```python
def _create_ws(self, user_channel: bool = False) -> CoinbaseWebSocket:
    auth_provider = None
    if user_channel:
        if self.config.auth_type == "JWT":
            # Generate JWT auth payload
            auth_provider = lambda: {
                "jwt": self.client.auth.generate_jwt(
                    method="GET",
                    path="/ws/user"
                )
            }
        else:
            # Generate HMAC auth payload
            auth_provider = lambda: {
                "api_key": self.client.auth.api_key,
                "signature": self.client.auth.sign("GET", "/ws/user", None)["CB-ACCESS-SIGN"],
                "timestamp": str(int(time.time())),
                "passphrase": self.client.auth.passphrase
            }
    
    return CoinbaseWebSocket(url=self.endpoints.ws_url, ws_auth_provider=auth_provider)
```

### 3. Missing IBrokerage.get_product()
**Impact**: Execution engine crashes
**Location**: `src/bot_v2/features/live_trade/execution_v3.py:229-231`

```python
# BROKEN CODE:
product = self.broker.get_product(symbol)  # METHOD NOT IN INTERFACE!
```

**FIX in interfaces.py**:
```python
class IBrokerage(Protocol):
    # Add this method
    def get_product(self, symbol: str) -> Product:
        """Get single product by symbol."""
        ...
```

**FIX in adapter.py**:
```python
def get_product(self, symbol: str) -> Product:
    """Get product from cache or fetch."""
    pid = normalize_symbol(symbol)
    return self.product_catalog.get(self.client, pid)
```

### 4. Order Status Filter Mismatch
**Impact**: list_orders() returns nothing or errors
**Location**: `src/bot_v2/features/brokerages/coinbase/adapter.py:282-289`

```python
# BROKEN CODE:
params["order_status"] = status.value  # Sends internal enum value!
```

**FIX**:
```python
# Map internal status to Coinbase values
STATUS_MAP = {
    OrderStatus.PENDING: "OPEN",
    OrderStatus.OPEN: "OPEN",
    OrderStatus.FILLED: "FILLED",
    OrderStatus.CANCELLED: "CANCELLED",
    OrderStatus.REJECTED: "FAILED",
    OrderStatus.EXPIRED: "EXPIRED"
}

def list_orders(self, status: Optional[OrderStatus] = None, symbol: Optional[str] = None) -> List[Order]:
    params: Dict[str, str] = {}
    if status:
        params["order_status"] = STATUS_MAP.get(status, "OPEN")
    # ...
```

### 5. Runner Using Wrong Client
**Impact**: No broker features work
**Location**: `scripts/run_perps_bot.py:269-272`

```python
# BROKEN CODE:
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
self.broker = CoinbaseClient()  # RAW CLIENT, NOT BROKER!
await self.broker.connect()  # CONNECT IS SYNC!
```

**FIX**:
```python
from bot_v2.orchestration.broker_factory import create_brokerage

async def initialize(self):
    # Create proper broker
    self.broker = create_brokerage()  # Returns IBrokerage
    
    # Connect (sync)
    if not self.broker.connect():
        raise RuntimeError("Failed to connect to broker")
```

## 🟡 HIGH PRIORITY FIXES

### 6. WebSocket Liveness Check
**Location**: `src/bot_v2/features/brokerages/coinbase/ws.py:111-121`

```python
# FIX: Update timestamp AFTER check
for msg in self._transport.stream():
    current_time = time.time()
    
    # Check liveness BEFORE updating
    if self._liveness_timeout > 0 and self._last_message_time:
        elapsed = current_time - self._last_message_time
        if elapsed > self._liveness_timeout:
            raise TimeoutError(f"No messages for {elapsed:.1f}s")
    
    # NOW update timestamp
    self._last_message_time = current_time
    yield msg
```

### 7. Add CoinbaseEndpoints Methods
**Location**: `src/bot_v2/features/brokerages/coinbase/endpoints.py`

```python
def supports_derivatives(self) -> bool:
    """Check if derivatives are enabled."""
    return self.enable_derivatives

def get_funding_endpoint(self, symbol: str) -> Optional[str]:
    """Get funding rate endpoint if available."""
    if not self.enable_derivatives:
        return None
    # Return actual endpoint when identified
    return None  # TODO: Find real endpoint
```

### 8. Symbol Normalization
**Location**: `src/bot_v2/features/brokerages/coinbase/models.py`

```python
def normalize_symbol(symbol: str) -> str:
    """Normalize symbol for consistency."""
    # Strip -PERP suffix for now (Coinbase uses BTC-USD internally)
    if symbol.endswith("-PERP"):
        base = symbol[:-5]
        return f"{base}-USD"
    return symbol
```

### 9. Reduce-Only Validation
**Location**: `src/bot_v2/features/brokerages/coinbase/adapter.py`

```python
def place_order(self, symbol: str, side: OrderSide, ..., reduce_only: Optional[bool] = None):
    # Add reduce-only validation
    if reduce_only:
        positions = self.list_positions()
        position = next((p for p in positions if p.symbol == symbol), None)
        
        if not position or position.qty == 0:
            raise ValidationError("No position to reduce")
        
        # Check direction
        if (side == OrderSide.BUY and position.qty > 0) or \
           (side == OrderSide.SELL and position.qty < 0):
            raise ValidationError("Order would increase position")
```

### 10. Balance Parsing for Both Modes
**Location**: `src/bot_v2/features/brokerages/coinbase/adapter.py:302-315`

```python
def list_balances(self) -> List[Balance]:
    data = self.client.get_accounts() or {}
    accounts = data.get("accounts") or []
    balances: List[Balance] = []
    
    for a in accounts:
        try:
            currency = a.get("currency", "")
            
            # Handle both Advanced Trade and Exchange formats
            if "available_balance" in a:
                # Advanced Trade format
                available = a.get("available_balance", {}).get("value", "0")
                hold = a.get("hold", {}).get("value", "0")
            else:
                # Exchange format
                available = a.get("available", "0")
                hold = a.get("hold", "0")
            
            total = Decimal(available) + Decimal(hold)
            balances.append(Balance(
                asset=currency,
                total=total,
                available=Decimal(available),
                hold=Decimal(hold)
            ))
        except Exception as e:
            logger.warning(f"Could not parse balance for {currency}: {e}")
    
    return balances
```

## 🟢 IMPORTANT BUT LESS URGENT

### 11. Use close_position Endpoint
```python
def close_position(self, symbol: str, qty: Optional[Decimal] = None) -> Order:
    """Close position using native endpoint when available."""
    if self.config.api_mode == "advanced":
        try:
            # Use native close_position endpoint
            response = self.client.close_position(symbol, size=qty)
            return to_order(response)
        except:
            pass  # Fall back to market order
    
    # Fallback: reduce-only market order
    # ... existing code ...
```

### 12. Sequence Gap Handling
- Make sequence tracking optional/configurable
- Only enforce for channels that guarantee monotonic sequences
- Add snapshot recovery on gap detection

### 13. Funding Calculator Fix
- Don't skip first period
- Test both long pays and short receives scenarios
- Add unit tests for edge cases

## Testing Requirements

### Must Unskip These Tests:
1. `test_derivatives_phase3.py` - WebSocket auth
2. `test_derivatives_phase4.py` - Funding/PnL
3. `test_http_request_layer.py` - Headers

### Must Add These Tests:
1. `test_get_product_contract.py` - Verify get_product() works
2. `test_order_status_mapping.py` - Verify status filter mapping
3. `test_reduce_only_validation.py` - Verify reduce-only guards
4. `test_ws_auth_payload.py` - Verify auth payload generation

## Deployment Checklist

### Before Phase 1:
- [ ] Fix all CRITICAL items (1-5)
- [ ] Fix HIGH PRIORITY items (6-10)
- [ ] Run all unskipped tests
- [ ] Verify with `validate_perps_client_week1.py`
- [ ] Test WebSocket auth with real credentials
- [ ] Verify symbol normalization consistency

### Before Phase 2 (Orders):
- [ ] Complete all IMPORTANT items (11-13)
- [ ] Add comprehensive order validation tests
- [ ] Test reduce-only enforcement
- [ ] Verify TIF mappings for all order types
- [ ] Load test with multiple concurrent orders

## Implementation Priority

1. **TODAY**: Fix items 1-5 (critical breaks)
2. **TOMORROW**: Fix items 6-10 (high priority)
3. **THIS WEEK**: Fix items 11-13 (important)
4. **NEXT WEEK**: Comprehensive testing and validation

## Risk Assessment

**Current State**: 🔴 **NOT PRODUCTION READY**
- Multiple critical failures that will crash on first use
- Authentication broken for user channels
- Core broker interface incomplete

**After Critical Fixes**: 🟡 **PHASE 1 POSSIBLE**
- Read-only operations should work
- Need extensive testing before orders

**After All Fixes**: 🟢 **PHASE 2 READY**
- Can proceed with cautious order placement
- Start with canary deployment

---

*Generated: December 2024*
*Severity: CRITICAL - Fix before any production use*