# Coinbase API Mode Compatibility Table

Quick reference for endpoint availability by API mode.

> **Note**: GPT-Trader only supports authenticated Advanced Trade (JWT). Exchange mode is limited
> to public endpoints; the tables below describe Coinbase API capabilities, not guaranteed
> GPT-Trader support.

## ✅ Available in Both Modes

| Method | Advanced Trade Path | Exchange Path |
|--------|-------------------|---------------|
| `get_time()` | `/api/v3/brokerage/time` | `/time` |
| `get_accounts()` | `/api/v3/brokerage/accounts` | `/accounts` |
| `get_account(id)` | `/api/v3/brokerage/accounts/{id}` | `/accounts/{id}` |
| `get_products()` | `/api/v3/brokerage/products` | `/products` |
| `get_product(id)` | `/api/v3/brokerage/products/{id}` | `/products/{id}` |
| `get_ticker(id)` | `/api/v3/brokerage/products/{id}/ticker` | `/products/{id}/ticker` |
| `get_candles(id)` | `/api/v3/brokerage/products/{id}/candles` | `/products/{id}/candles` |
| `get_product_book(id)` | `/api/v3/brokerage/product_book` | `/products/{id}/book` |
| `create_order()` | `/api/v3/brokerage/orders` | `/orders` |
| `cancel_orders()` | `/api/v3/brokerage/orders/batch_cancel` | `/orders/{id}` |
| `get_order(id)` | `/api/v3/brokerage/orders/historical/{id}` | `/orders/{id}` |
| `list_orders()` | `/api/v3/brokerage/orders/historical` | `/orders` |
| `get_fees()` | `/api/v3/brokerage/fees` | `/fees` |

## 🚫 Advanced Mode Only

These endpoints are **not available** in exchange mode and will raise `InvalidRequestError`:

| Method | Purpose | Error Message |
|--------|---------|---------------|
| `get_best_bid_ask()` | Best bid/ask across products | "Use get_ticker for individual products" |
| `get_key_permissions()` | API key permissions | "Set COINBASE_API_MODE=advanced" |
| `get_limits()` | Account limits | "Set COINBASE_API_MODE=advanced" |
| `convert_quote()` | Convert quote | "Set COINBASE_API_MODE=advanced" |
| `get_convert_trade()` | Convert trade details | "Set COINBASE_API_MODE=advanced" |

### Order Management (Advanced Only)
- `list_orders_batch()` - Batch order retrieval
- `preview_order()` - Order preview with fees
- `edit_order_preview()` - Preview order edits
- `edit_order()` - Modify existing orders
- `close_position()` - Close position

### Portfolio Management (Advanced Only)
- `list_portfolios()` - List all portfolios
- `get_portfolio()` - Portfolio details
- `get_portfolio_breakdown()` - Portfolio breakdown
- `move_funds()` - Move funds between portfolios

### Payment Methods (Advanced Only)
- `list_payment_methods()` - Available payment methods
- `get_payment_method()` - Payment method details

### INTX/Derivatives (Advanced Only)
- `intx_allocate()` - Allocate collateral
- `intx_balances()` - INTX balances
- `intx_portfolio()` - INTX portfolio
- `intx_positions()` - INTX positions
- `intx_position()` - Specific position
- `intx_multi_asset_collateral()` - Multi-asset collateral

> **Note:** INTX endpoints require Advanced Trade mode **and** institutional entitlements on the API key. When unavailable, read methods return empty payloads and `intx_allocate()` raises `InvalidRequestError`.

### CFM/Futures (Advanced Only)
- `cfm_balance_summary()` - CFM balance summary
- `cfm_positions()` - CFM positions
- `cfm_position()` - Specific CFM position
- `cfm_sweeps()` - CFM sweeps
- `cfm_sweeps_schedule()` - Sweep schedule
- `cfm_intraday_current_margin_window()` - Margin window
- `cfm_intraday_margin_setting()` - Margin settings

> **Note:** All CFM endpoints require Advanced Trade mode **and** derivatives enablement on the API key. In Exchange mode or when derivatives are disabled, the client raises `InvalidRequestError`.

## WebSocket Channels

| Channel | Advanced Mode | Exchange Mode |
|---------|--------------|---------------|
| ticker | ✅ | ✅ |
| ticker_batch | ✅ | ❌ |
| level2 | ✅ | ✅ |
| user | ✅ | ✅ |
| market_trades | ✅ | ❌ |
| status | ✅ | ✅ |
| heartbeats | ✅ | ✅ |
| candles | ✅ | ❌ |

## Authentication

| Auth Type | GPT-Trader Support |
|-----------|--------------------|
| CDP JWT | ✅ Advanced mode |
| HMAC (any) | ❌ Not implemented |

## Development Tips

1. **Testing**: Use the mock broker for sandbox-style runs
2. **Production**: Always use Advanced mode for full features
3. **Error Handling**: Catch `InvalidRequestError` when calling advanced-only methods
4. **Mode Detection**: Check `client.api_mode` to conditionally use features

## Quick Mode Check

```python
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient

if client.api_mode == "advanced":
    # Can use all features
    portfolios = client.list_portfolios()
    best_prices = client.get_best_bid_ask(["BTC-USD", "ETH-USD"])
else:
    # Limited to basic trading
    # Use individual ticker calls instead
    btc_ticker = client.get_ticker("BTC-USD")
```
