# Quick Start

This walkthrough gets the development profile running in a few minutes. It
defaults to the safe `MockBroker`, but also covers the env toggles required to
hit real Coinbase endpoints when you are ready.

## 1. Install Dependencies

```bash
python3 -m pip install --user pipx  # optional convenience
pipx install poetry                 # or use your existing Poetry install

poetry install
```

The project targets Python 3.12+. If you manage Python with `pyenv`, run
`pyenv local 3.12.4` (or later) before `poetry install`.

## 2. Create a .env File (optional)

Copy the template if you plan to store credentials locally:

```bash
cp config/environments/.env.template .env
```

For the dev profile nothing else is required—the mock broker and deterministic
fills are used automatically.

## 3. Smoke-Test the Dev Profile

```bash
poetry run perps-bot --profile dev --dev-fast
```

What to expect:

- Mark prices are fetched via REST quotes for the top-ten USD spot markets (`BTC`, `ETH`, `SOL`, `XRP`, `LTC`, `ADA`, `DOGE`, `BCH`, `AVAX`, `LINK`).
- Orders are routed through the enhanced `MockBroker`, so no live trades are
  placed.
- Metrics land under `var/data/perps_bot/dev/` for inspection.

## 4. Enable Real Spot Trading (optional)

When you are ready to exercise Coinbase APIs, provide production Advanced Trade
credentials and lift the safety toggle:

```bash
export BROKER=coinbase
export COINBASE_API_KEY="..."
export COINBASE_API_SECRET="..."
export COINBASE_API_PASSPHRASE="..."
export SPOT_FORCE_LIVE=1

poetry run perps-bot --profile spot --dev-fast
```

Use the `spot` profile for live spot execution; it keeps leverage at 1x and
disables reduce-only mode. The `dev` profile continues to assume mocks even if
the toggle is set.

## 5. (Future) Perpetuals Readiness

Perpetual futures remain gated behind Coinbase INTX. To exercise those paths you
must:

1. Obtain INTX approval for your account.
2. Set `COINBASE_ENABLE_DERIVATIVES=1`.
3. Provide CDP JWT credentials (`COINBASE_PROD_CDP_API_KEY` and
   `COINBASE_PROD_CDP_PRIVATE_KEY`).

Until then the code paths continue to compile and run in tests, but real order
placement is disabled by design.

## 6. Recommended Local Checks

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
