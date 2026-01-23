# Coinbase Integration

---
status: current
last-updated: 2026-01-23
---

This repository supports **Coinbase spot trading by default**. Any Coinbase details that drift (endpoint catalogs, message schemas, rate limits) should be verified against the **official Coinbase documentation** and the **current implementation** in this repo.

## Source of Truth

- Code: `src/gpt_trader/features/brokerages/coinbase/`
- Configuration: `config/environments/.env.template` and `src/gpt_trader/app/config/bot_config.py`

## API Modes

- **Advanced Trade** (default): `COINBASE_API_MODE=advanced`
  - JWT/CDP keys, authenticated endpoints, portfolios, trading.
- **Exchange (sandbox)**: `COINBASE_API_MODE=exchange` + `COINBASE_SANDBOX=1`
  - Public sandbox endpoints only (no authenticated trading flows).

## Authentication (JWT-only)

- Preferred: `COINBASE_CREDENTIALS_FILE=/path/to/cdp_key.json`
- Or: `COINBASE_CDP_API_KEY` + `COINBASE_CDP_PRIVATE_KEY`
- Implementation: `src/gpt_trader/features/brokerages/coinbase/auth.py`

## Derivatives Gating

- **CFM futures (US)**: `TRADING_MODES=cfm` + `CFM_ENABLED=1`
- **INTX perps**: requires INTX access + `COINBASE_ENABLE_INTX_PERPS=1`

## Common Troubleshooting

- **Zero balances**: verify portfolio scoping + permissions, and confirm you are calling Advanced Trade endpoints with a CDP key.
- **“Sandbox doesn’t trade”**: Advanced Trade has no authenticated sandbox; use `MOCK_BROKER=1` for deterministic testing.

## References

- Internal integration notes: `src/gpt_trader/features/brokerages/coinbase/README.md`
- Official docs: https://docs.cdp.coinbase.com/
