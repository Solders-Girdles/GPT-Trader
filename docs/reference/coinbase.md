# Coinbase Integration Reference

This page summarizes the current Coinbase integration. For the full, consolidated reference, see `docs/reference/coinbase_complete.md`.

Key highlights:
- **Spot-first**: BTC-USD, ETH-USD, etc. use Advanced Trade REST (HMAC) against production endpoints. Sandbox Exchange APIs are only for mock/paper flows.
- **Perpetual futures**: Still available, but only for INTX-enabled accounts. Set `COINBASE_ENABLE_DERIVATIVES=1` and wire in CDP (JWT) credentials.
- Preview gating (`ORDER_PREVIEW_ENABLED=1`) remains supported and auto-enabled in Advanced/PROD with JWT.
- Sandbox/Exchange mode supports spot only; the bot rejects live runs when `COINBASE_SANDBOX=1` to avoid behaviour drift.

For detailed endpoint mappings, payloads, WebSocket usage, and troubleshooting, open `docs/reference/coinbase_complete.md`.

Quick links to official docs: see `docs/reference/coinbase_api_links.md`.
