# Quick Start

---
status: current
last-updated: 2025-10-07
---

This walkthrough gets the development profile running in a few minutes. It
defaults to the safe `DeterministicBroker`, but also covers the env toggles required to
hit real Coinbase endpoints when you are ready.

## 1. Install Dependencies

```bash
python3 -m pip install --user pipx  # optional convenience
pipx install poetry                 # or use your existing Poetry install

poetry install
```

The project targets Python 3.12+. If you manage Python with `pyenv`, run
`pyenv local 3.12.4` (or later) before `poetry install`.

## 2. Configure Environment Files (optional)

Copy the templates to keep credentials out of version control:

```bash
cp config/environments/.env.template .env
cp deploy/bot_v2/docker/.env.example deploy/bot_v2/docker/.env
```

The root `.env` seeds runtime configuration. The base Compose stack reads
`deploy/bot_v2/docker/.env`; only the bot and Grafana credentials are required for the default
lightweight stack. Database, Redis, RabbitMQ, and Vault secrets are optional unless you load the
infrastructure override.

## 3. Launch the Local Stack

Use the `Makefile` helper to boot the dev profile services (bot only, Prometheus/Grafana opt-in):

```bash
make dev-up
```

Need metrics and tracing? Opt into the observability profile when required:

```bash
docker compose --project-directory deploy/bot_v2/docker \
  -f deploy/bot_v2/docker/docker-compose.yaml \
--profile observability up -d
```

Want the retired Postgres/Redis/RabbitMQ/Vault helpers? Layer the override and enable the
`infra` profile:

```bash
docker compose --project-directory deploy/bot_v2/docker \
  -f deploy/bot_v2/docker/docker-compose.yaml \
  -f deploy/bot_v2/docker/docker-compose.infrastructure.yaml \
  --profile infra up -d
```

> `make dev-up` automatically passes `deploy/bot_v2/docker/.env`, so the stack
> runs with the secrets you just populated. Override-only variables are ignored unless you opt in.

## 4. Smoke-Test the Dev Profile

```bash
poetry run coinbase-trader run --profile dev --dev-fast
```

What to expect:

- Mark prices are fetched via REST quotes for the top-ten USD spot markets (`BTC`, `ETH`, `SOL`, `XRP`, `LTC`, `ADA`, `DOGE`, `BCH`, `AVAX`, `LINK`).
- Orders are routed through the built-in `DeterministicBroker`, so no live trades are
  placed.
- Metrics land under `var/data/coinbase_trader/dev/` for inspection.

> Legacy alias: `poetry run perps-bot …` continues to work for existing automation, but new workflows should prefer `coinbase-trader`.

## 5. Enable Real Spot Trading (optional)

When you are ready to exercise Coinbase APIs, provide production Advanced Trade
credentials and lift the safety toggle:

```bash
export BROKER=coinbase
export COINBASE_API_KEY="..."
export COINBASE_API_SECRET="..."
export COINBASE_API_PASSPHRASE="..."
export SPOT_FORCE_LIVE=1

poetry run coinbase-trader run --profile spot --dev-fast
```

Use the `spot` profile for live spot execution; it keeps leverage at 1x and
disables reduce-only mode. The `dev` profile continues to assume mocks even if
the toggle is set.

## 6. (Future) Perpetuals Readiness

Perpetual futures remain gated behind Coinbase INTX. To exercise those paths you
must:

1. Obtain INTX approval for your account.
2. Set `COINBASE_ENABLE_DERIVATIVES=1`.
3. Provide CDP JWT credentials (`COINBASE_PROD_CDP_API_KEY` and
   `COINBASE_PROD_CDP_PRIVATE_KEY`).

Until then the code paths continue to compile and run in tests, but real order
placement is disabled by design.

## 7. Recommended Local Checks

```bash
# Full regression suite (spot + orchestration)
poetry run pytest -q

# Optional: offline verification bundle
poetry run python scripts/validation/verify_core.py --check all
```

These commands stay within local resources—no network calls are made unless you
explicitly enable live trading as described above.

### Adjusting Risk Limits

The spot profiles automatically load `config/risk/spot_top10.json`, which sets
per-symbol notional caps, leverage limits, and slippage guards for the ten USD
markets. Edit that file (or point `RISK_CONFIG_PATH` to a custom copy) if you
want different sizing rules.
