# Coinbase API Links (Lite)

Purpose-built quicklinks for agents to jump straight to the right Coinbase docs.

## Most-Used Tasks
- Place/cancel orders: Advanced Trade → Orders (see sidebar from overview)
- Get fills/executions: Advanced Trade → Fills (from overview)
- List products/contracts: Advanced Trade → Products (from overview)
- Account/balances: Advanced Trade → Accounts (from overview)
- WebSocket channels: Advanced Trade → WS Overview (ticker, level2, matches)

Tip: Use the doc site search with keywords like “place order”, “get fills”, “products”, “accounts”, “websocket ticker”.

## Advanced Trade (Spot & Perps)
- Docs home: https://docs.cloud.coinbase.com/advanced-trade-api/docs/welcome
- Signed requests (CDP/JWT): https://docs.cloud.coinbase.com/advanced-trade-api/docs/signed-requests
- WebSocket overview: https://docs.cloud.coinbase.com/advanced-trade-api/docs/ws-overview
- REST base path: `https://api.coinbase.com/api/v3/brokerage`

Notes:
- Spot trading (default) uses Advanced Trade v3 with HMAC credentials.
- Perpetual futures (e.g., BTC-PERP, ETH-PERP) require INTX access; switch to CDP (JWT) only after approval.

## Exchange (Sandbox Spot)
- Docs home: https://docs.cloud.coinbase.com/exchange/docs/welcome
- Auth (HMAC): https://docs.cloud.coinbase.com/exchange/docs/authorization-and-authentication
- WebSocket overview: https://docs.cloud.coinbase.com/exchange/docs/websocket-overview
- REST base URL: `https://api.exchange.coinbase.com`

Notes:
- Sandbox supports spot only; perpetuals are NOT available in sandbox.
- Use HMAC auth for Exchange (legacy) endpoints.

## Operational Reminders
- Production vs Sandbox:
  - Production perpetuals → Advanced Trade v3 + CDP/JWT
  - Sandbox spot → Exchange v2 + HMAC
- In-code defaults: Advanced Trade uses `/api/v3/brokerage` under `https://api.coinbase.com`.

## Helpful Non-Endpoint Links
- API status: https://status.coinbase.com/
- Main docs portal: https://docs.cloud.coinbase.com/

## See Also (Internal)
- Complete reference: docs/reference/coinbase_complete.md
- Troubleshooting: docs/reference/compatibility_troubleshooting.md
- Test coverage: docs/testing/coinbase_coverage_matrix.md
