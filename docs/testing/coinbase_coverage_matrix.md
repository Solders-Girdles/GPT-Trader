Coinbase API Test Coverage Matrix

- Accounts: get_accounts, get_account, list_portfolios, get_portfolio, get_portfolio_breakdown, move_funds
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_integration.py::TestCoinbaseAccounts

- Trading & Order Management: place_order, cancel_orders, list_orders, list_orders_batch, list_fills, get_order_historical, preview_order, edit_order_preview, edit_order, close_position
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_integration.py::TestCoinbaseTrading
  - tests/unit/bot_v2/features/brokerages/coinbase/test_endpoint_chains.py

- Market Data & Models: products, product, ticker, candles, order_book, best_bid_ask, quote normalization
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_integration.py::TestCoinbaseMarketData
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_models.py

- System, Fees & Permissions: time, fees, limits, key_permissions, auth selection
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_integration.py::TestCoinbaseSystem
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_permissions.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_auth.py

- Conversions & Funding Flows: convert_quote, get_convert_trade, move funds, sweeps
  - tests/unit/bot_v2/features/brokerages/coinbase/test_conversion_endpoints.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_cfm_endpoints.py

- Perpetuals (CFM) Risk Controls: balance_summary, positions, sweeps schedule, quantization rules
  - tests/unit/bot_v2/features/brokerages/coinbase/test_cfm_endpoints.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_quantization_rules.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_specs_quantization.py

- Institutional (INTX): allocate, balances, portfolio, positions, multi_asset_collateral
  - tests/unit/bot_v2/features/brokerages/coinbase/test_intx_endpoints.py

- WebSocket & Streaming: ticker, level2, trades, status, user channel auth, reconnection, SequenceGuard coverage
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_websocket.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_adapter_streams.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_adapter_integration.py

- Exchange Mode vs Advanced Mode: path mapping, feature gating, broker configuration
  - tests/unit/bot_v2/features/brokerages/coinbase/test_adapter_integration.py::TestAPIModeBehavior
  - tests/unit/bot_v2/features/brokerages/coinbase/test_endpoint_registry.py
  - tests/unit/bot_v2/orchestration/test_broker_factory.py

- Error Handling & Pagination: auth failures, rate limits with retries, 503 recovery, pagination edge cases, query params
  - tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_errors.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_pagination_and_errors.py
  - tests/unit/bot_v2/features/brokerages/coinbase/test_http_request_layer.py

Notes
- All tests are deterministic with mocked transports. No live network calls.
- For sandbox behavior and mode selection, see tests in orchestration and config.
