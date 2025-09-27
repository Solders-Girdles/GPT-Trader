Coinbase API Test Coverage Matrix

- Accounts: get_accounts, get_account, list_portfolios, get_portfolio, get_portfolio_breakdown, move_funds
  - tests/unit/bot_v2/features/brokerages/coinbase/test_account_endpoints.py

- Trading: place_order, cancel_orders, list_orders, list_orders_batch, list_fills, get_order_historical, preview_order, edit_order_preview, edit_order, close_position
  - tests/unit/bot_v2/features/brokerages/coinbase/test_trading_endpoints.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_endpoint_chains.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_endpoints.py

- Market Data: products, product, ticker, candles, order_book, best_bid_ask
  - tests/unit/bot_v2/features/brokerages/coinbase/test_market_data_endpoints.py

- System: time, fees, limits, key_permissions, convert trade
  - tests/unit/bot_v2/features/brokerages/coinbase/test_system_endpoints.py

- Conversions: convert_quote, get_convert_trade
  - tests/unit/bot_v2/features/brokerages/coinbase/test_conversion_endpoints.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_system_endpoints.py

- CFM (Perpetuals): balance_summary, positions, position, sweeps, sweeps schedule, intraday margin window, intraday margin setting
  - tests/unit/bot_v2/features/brokerages/coinbase/test_cfm_endpoints.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_endpoints.py

- INTX (Institutional): allocate, balances, portfolio, positions, position, multi_asset_collateral
  - tests/unit/bot_v2/features/brokerages/coinbase/test_intx_endpoints.py

- WebSocket: ticker, level2, trades, status, user, auth for user channel, reconnection, multiple subscriptions
  - tests/unit/bot_v2/features/brokerages/coinbase/test_websocket_streaming.py

- Exchange Mode vs Advanced Mode: path mapping and gating
  - tests/unit/bot_v2/features/brokerages/coinbase/test_exchange_mode.py
  - tests/unit/bot_v2/orchestration/test_broker_factory.py
  - tests/test_config_matrix.py

- Edge Cases & Errors: 401 auth, 429 rate limit (retry), 503 (retry), 408 timeout (no retry), pagination edge cases, query params
  - tests/unit/bot_v2/features/brokerages/coinbase/test_endpoint_errors.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_pagination_endpoints.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_query_params.py

Notes
- All tests are deterministic with mocked transports. No live network calls.
- For sandbox behavior and mode selection, see tests in orchestration and config.
