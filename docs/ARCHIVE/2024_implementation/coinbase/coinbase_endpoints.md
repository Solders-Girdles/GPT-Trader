# Coinbase Advanced Trade Endpoints (Registry)

This file summarizes the endpoints we plan to support and tracks coverage in code.

Note: Compiled from public docs knowledge; verify against latest Coinbase docs.

## Products & Market Data
- GET `/api/v3/brokerage/products` – List products
- GET `/api/v3/brokerage/products/{product_id}` – Product details
- GET `/api/v3/brokerage/products/{product_id}/ticker` – Bid/ask/last
- GET `/api/v3/brokerage/products/{product_id}/candles` – Candles
- GET `/api/v3/brokerage/product_book` – Order book (aggregated)
- GET `/api/v3/brokerage/market/products` – List products (market namespace)
- GET `/api/v3/brokerage/market/products/{product_id}` – Product details (market)
- GET `/api/v3/brokerage/market/products/{product_id}/ticker` – Ticker (market)
- GET `/api/v3/brokerage/market/products/{product_id}/candles` – Candles (market)
- GET `/api/v3/brokerage/market/product_book` – Order book (market)
- GET `/api/v3/brokerage/best_bid_ask` – Best bid/ask for given product_ids

## Accounts
- GET `/api/v3/brokerage/accounts` – Accounts and balances
- GET `/api/v3/brokerage/accounts/{account_uuid}` – Account detail
- GET `/api/v3/brokerage/key_permissions` – API key permissions
- GET `/api/v3/brokerage/time` – Server time

## Orders
- POST `/api/v3/brokerage/orders` – Create order (market/limit/stop/stop-limit)
- POST `/api/v3/brokerage/orders/preview` – Preview order
- POST `/api/v3/brokerage/orders/edit_preview` – Preview order edit
- POST `/api/v3/brokerage/orders/edit` – Edit order
- POST `/api/v3/brokerage/orders/close_position` – Close derivative position
- POST `/api/v3/brokerage/orders/batch_cancel` – Cancel orders by IDs
- GET `/api/v3/brokerage/orders/historical/{order_id}` – Get order
- GET `/api/v3/brokerage/orders/historical` – List historical orders
- GET `/api/v3/brokerage/orders/historical/batch` – Batch lookup
- GET `/api/v3/brokerage/orders/historical/fills` – Fills

## Fees & Limits
- GET `/api/v3/brokerage/fees` – Fee structure
- GET `/api/v3/brokerage/limits` – Trading limits
- GET `/api/v3/brokerage/transaction_summary` – Transaction summary

## Conversions (if enabled)
- POST `/api/v3/brokerage/convert/quote` – Create convert quote
- GET `/api/v3/brokerage/convert/trade/{trade_id}` – Convert trade status

## Payment Methods
- GET `/api/v3/brokerage/payment_methods` – List payment methods
- GET `/api/v3/brokerage/payment_methods/{payment_method_id}` – Payment method detail

## Portfolios
- GET `/api/v3/brokerage/portfolios` – List portfolios
- GET `/api/v3/brokerage/portfolios/{portfolio_uuid}` – Get portfolio
- POST `/api/v3/brokerage/portfolios/move_funds` – Move funds between portfolios

## Institutional (INTX)
- POST `/api/v3/brokerage/intx/allocate` – Allocation
- GET `/api/v3/brokerage/intx/balances/{portfolio_uuid}` – Balances
- GET `/api/v3/brokerage/intx/portfolio/{portfolio_uuid}` – Portfolio
- GET `/api/v3/brokerage/intx/positions/{portfolio_uuid}` – Positions
- GET `/api/v3/brokerage/intx/positions/{portfolio_uuid}/{symbol}` – Position detail
- GET `/api/v3/brokerage/intx/multi_asset_collateral` – MAC

## Derivatives (CFM)
- GET `/api/v3/brokerage/cfm/balance_summary` – Balance summary
- GET `/api/v3/brokerage/cfm/positions` – Positions
- GET `/api/v3/brokerage/cfm/positions/{product_id}` – Position detail
- GET `/api/v3/brokerage/cfm/sweeps` – Sweeps
- GET `/api/v3/brokerage/cfm/sweeps/schedule` – Sweep schedule
- GET `/api/v3/brokerage/cfm/intraday/current_margin_window` – Current margin window
- POST `/api/v3/brokerage/cfm/intraday/margin_setting` – Set margin setting

## Derivatives Notes
- Perpetual/futures products appear via `products` with `contract_type` metadata (e.g., `perpetual`, `future`).
- Order parameters may include leverage and reduce-only flags; user/account eligibility applies.
- Positions/funding endpoints vary by account availability; add once confirmed in docs.

Coverage is tracked in `src/bot_v2/features/brokerages/coinbase/endpoints.py` and client stubs.
