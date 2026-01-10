# Security Documentation

This document outlines the security controls and operational guidance for GPT-Trader.

## Security Overview

- Credentials and authentication live under `src/gpt_trader/features/brokerages/coinbase/`
  (`auth.py`, `credentials.py`) and use JWT-based CDP keys.
- Secrets management is handled by `src/gpt_trader/security/secrets_manager.py`, which
  supports HashiCorp Vault and an encrypted local file fallback in `~/.gpt_trader/secrets`.
- Runtime validation is centralized in `src/gpt_trader/security/security_validator.py` and
  is enforced by the TradingEngine guard stack during live order submission.
- IP allowlisting is enforced by `src/gpt_trader/security/ip_allowlist_enforcer.py`.
- Structured JSON logs redact sensitive keys via `src/gpt_trader/logging/json_formatter.py`.

## Required Environment Variables

### Coinbase credentials (JWT only)

Advanced Trade / CDP JWT:

- `COINBASE_CREDENTIALS_FILE` (JSON file with `name` + `privateKey`), or
- `COINBASE_CDP_API_KEY` + `COINBASE_CDP_PRIVATE_KEY`, or
- `COINBASE_PROD_CDP_API_KEY` + `COINBASE_PROD_CDP_PRIVATE_KEY`
- Legacy fallback: `COINBASE_API_KEY_NAME` + `COINBASE_PRIVATE_KEY`

Legacy Exchange/HMAC keys are not used by the current runtime; use JWT credentials above.

### Secrets and encryption

- `GPT_TRADER_ENCRYPTION_KEY` (required outside development if using SecretsManager)
- `VAULT_ADDR` / `VAULT_TOKEN` (required to enable Vault storage)

### IP allowlist (optional, recommended for production)

- `IP_ALLOWLIST_ENABLED=1`
- `IP_ALLOWLIST_<SERVICE>=ip1,ip2,cidr1` (example: `IP_ALLOWLIST_COINBASE_INTX`)

## Preflight Security Checks

Run production preflight to validate credentials, key permissions, and API connectivity:

```bash
uv run python scripts/production_preflight.py --profile canary
```

Preflight checks live in `src/gpt_trader/preflight/checks/` and verify JWT generation,
permissions, and portfolio readiness before live trading. Use `--warn-only` or
`GPT_TRADER_PREFLIGHT_WARN_ONLY=1` to downgrade failures to warnings.

## Sensitive Data Handling

- JSON logs redact keys like `api_key`, `private_key`, `token`, and `password`.
- Runtime data persists to `runtime_data/<profile>/` (SQLite `events.db`, `orders.db`).
  Avoid writing secrets into events or logs.

## Best Practices

- Use read-only keys for research/backtesting.
- Enable IP allowlisting in production and rotate credentials regularly.
- Store secrets in a vault or CI secret manager; keep `.env*` files out of version control.
