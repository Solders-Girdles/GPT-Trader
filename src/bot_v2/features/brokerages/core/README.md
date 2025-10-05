# Brokerages Core

**Purpose**: Shared brokerage interfaces, error definitions, and abstractions for broker integrations.

---

## Overview

The `brokerages/core` module provides:
- **IBrokerage Protocol**: Standard interface all brokers must implement
- **Common Data Models**: Order, Position, Balance, Product definitions
- **Error Hierarchy**: Standardized brokerage errors
- **Health Checking**: Broker connection status monitoring

**Coverage**: Part of `features/brokerages` (86.7% overall)

---

## Interface Contract

### Core Protocol

```python
from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage,
    OrderSide,
    OrderType,
    OrderStatus,
    MarketType
)
```

### Required Methods

Every broker adapter must implement:

#### Connection Management
```python
def connect(self) -> bool:
    """Establish connection to broker API."""

def disconnect(self) -> None:
    """Close connection to broker API."""

def is_connected(self) -> bool:
    """Check if currently connected."""

def check_health(self) -> BrokerHealth:
    """Check broker API health and responsiveness."""
```

#### Account & Balances
```python
def get_account_id(self) -> str:
    """Get broker account ID."""

def list_balances(self) -> list[Balance]:
    """Get all account balances."""

def get_balance(self, asset: str) -> Balance:
    """Get balance for specific asset."""
```

#### Products & Market Data
```python
def get_products(self) -> list[Product]:
    """List all tradeable products."""

def get_product(self, symbol: str) -> Product:
    """Get product details."""

def get_price(self, symbol: str) -> Decimal:
    """Get current market price."""

def get_orderbook(self, symbol: str, level: int = 1) -> Orderbook:
    """Get orderbook depth."""
```

#### Order Management
```python
def place_order(self, order_request: OrderRequest) -> OrderResult:
    """Place a new order."""

def cancel_order(self, order_id: str) -> bool:
    """Cancel existing order."""

def get_order(self, order_id: str) -> Order:
    """Get order status."""

def list_orders(self, status: OrderStatus | None = None) -> list[Order]:
    """List orders, optionally filtered by status."""
```

#### Position Management
```python
def list_positions(self) -> list[Position]:
    """Get all open positions."""

def get_position(self, symbol: str) -> Position | None:
    """Get position for specific symbol."""

def close_position(self, symbol: str) -> OrderResult:
    """Close position at market price."""
```

---

## Data Models

### Order
```python
@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide  # BUY or SELL
    type: OrderType  # MARKET, LIMIT, STOP, etc.
    quantity: Decimal
    price: Decimal | None
    status: OrderStatus  # PENDING, OPEN, FILLED, CANCELLED
    filled_quantity: Decimal
    average_fill_price: Decimal | None
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Position
```python
@dataclass
class Position:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    market_value: Decimal
    side: OrderSide  # LONG or SHORT
```

### Balance
```python
@dataclass
class Balance:
    asset: str
    total: Decimal
    available: Decimal
    hold: Decimal  # Amount locked in orders
```

### Product
```python
@dataclass
class Product:
    symbol: str
    base_currency: str
    quote_currency: str
    market_type: MarketType  # SPOT, FUTURES, PERPETUAL
    min_order_size: Decimal
    max_order_size: Decimal
    price_increment: Decimal
    size_increment: Decimal
    tradeable: bool
```

---

## Error Hierarchy

All brokerage errors inherit from `BrokerageError`:

```python
class BrokerageError(Exception):
    """Base brokerage error."""

class ConnectionError(BrokerageError):
    """Connection failed or lost."""

class AuthenticationError(BrokerageError):
    """API authentication failed."""

class InsufficientFundsError(BrokerageError):
    """Insufficient balance for order."""

class OrderRejectedError(BrokerageError):
    """Order rejected by broker."""

class RateLimitError(BrokerageError):
    """API rate limit exceeded."""

class MarketClosedError(BrokerageError):
    """Market is closed."""

class PermissionDeniedError(BrokerageError):
    """Insufficient permissions."""
```

### Error Handling Example
```python
from bot_v2.features.brokerages.core.interfaces import (
    InsufficientFundsError,
    RateLimitError
)

try:
    result = broker.place_order(order_request)
except InsufficientFundsError as e:
    logger.error(f"Insufficient funds: {e}")
    # Handle by reducing order size or skipping
except RateLimitError as e:
    logger.warning(f"Rate limited: {e}")
    # Handle by waiting and retrying
except BrokerageError as e:
    logger.error(f"Broker error: {e}")
    # Generic error handling
```

---

## Health Checking

```python
@dataclass
class BrokerHealth:
    connected: bool
    api_responsive: bool
    last_check_timestamp: float
    error_message: str | None = None
```

### Usage
```python
health = broker.check_health()

if not health.connected:
    logger.error("Broker disconnected")
    broker.connect()
elif not health.api_responsive:
    logger.warning("Broker API slow or unresponsive")
```

---

## Implementing a New Broker

### Step 1: Create Adapter Class
```python
from bot_v2.features.brokerages.core.interfaces import IBrokerage

class MyBrokerAdapter:
    """Adapter for MyBroker API."""

    def connect(self) -> bool:
        # Implement connection logic
        ...

    def place_order(self, order_request: OrderRequest) -> OrderResult:
        # Map order_request to broker-specific format
        # Call broker API
        # Map response to OrderResult
        ...

    # Implement all other IBrokerage methods
```

### Step 2: Add to Broker Factory
```python
# In orchestration/broker_factory.py
from bot_v2.features.brokerages.mybroker import MyBrokerAdapter

def create_brokerage() -> IBrokerage:
    broker_name = os.getenv("BROKER", "coinbase")

    if broker_name == "mybroker":
        return MyBrokerAdapter(...)
    # ...
```

### Step 3: Add Tests
```python
# tests/unit/bot_v2/features/brokerages/mybroker/
def test_place_order():
    broker = MyBrokerAdapter(...)
    result = broker.place_order(order_request)
    assert result.success
```

---

## Testing Strategy

### Unit Tests
- Mock broker API responses
- Test error mapping (broker errors → BrokerageError hierarchy)
- Validate data model conversions

### Integration Tests
- Use recorded API responses (VCR.py or similar)
- Test full order lifecycle: place → fill → get status
- Verify health checking

### Contract Tests
- Ensure all IBrokerage methods are implemented
- Verify return types match protocol
- Check error types are correct

---

## Best Practices

### Error Handling
1. Always map broker-specific errors to `BrokerageError` hierarchy
2. Include broker error code/message in exception metadata
3. Log original error for debugging

### Data Mapping
1. Normalize symbol formats (e.g., "BTC/USD" → "BTC-USD")
2. Use Decimal for all financial values (never float)
3. Convert timestamps to UTC datetime

### Rate Limiting
1. Implement exponential backoff for retries
2. Respect broker rate limits
3. Use batch operations when available

### Testing
1. Never hit real broker APIs in unit tests
2. Use recorded responses for integration tests
3. Test with both valid and invalid inputs

---

## Dependencies

### Internal
- `bot_v2.shared.types` - Shared type definitions

### External
- `decimal.Decimal` - Precision arithmetic
- `datetime` - Timestamp handling
- `typing.Protocol` - Interface definition

---

## Available Broker Implementations

### Coinbase (`../coinbase/`)
- ✅ Spot trading
- ✅ Perpetual futures (INTX)
- ✅ Advanced Trade API v3
- ✅ Legacy Exchange API (sandbox)
- See `../coinbase/README.md` for details

### Future Brokers (Planned)
- Interactive Brokers
- Alpaca
- Binance
- Kraken

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Stable)
