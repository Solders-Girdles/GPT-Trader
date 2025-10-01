# GPT-Trader CLI Reference (Spot Profiles)

The legacy `gpt-trader` command suite has been removed from the tree; pull it
from repository history if needed. The supported interface is the `perps-bot`
CLI, which powers both spot
development and production runs while keeping derivatives behind the Coinbase
INTX gate.

## Primary Entry Point

```bash
poetry run perps-bot [OPTIONS]
```

### Core Options

| Option | Description |
|--------|-------------|
| `--profile {dev,demo,prod,canary,spot}` | Selects the configuration slice. `dev` defaults to mock trading, `spot` expects live keys. |
| `--dev-fast` | Runs a single control loop for smoke tests, then exits. |
| `--dry-run` | Forces non-mutating execution even when the profile targets Coinbase. |
| `--symbols BTC-USD ETH-USD` | Overrides the symbol universe (comma-free list). Defaults come from the selected profile or `TRADING_SYMBOLS`. |
| `--interval 5` | Sets the control-loop cadence in seconds. |
| `--target-leverage 1` | Specifies desired leverage (derivatives-gated). |
| `--reduce-only` | Enables reduce-only behaviour for all orders. |
| `--tif {GTC,IOC,FOK}` | Overrides the default time in force. |
| `--enable-preview` | Requires interactive confirmation before new orders when running attached to a TTY. |
| `--account-interval 300` | Publishes account telemetry every N seconds. |

### Account & Treasury Utilities

| Command | Purpose |
|---------|---------|
| `poetry run perps-bot --account-snapshot` | Prints Coinbase limits, balances, and fee schedules then exits. |
| `poetry run perps-bot --convert USD:USDC:100` | Submits a convert quote and commits it. |
| `poetry run perps-bot --move-funds from_uuid:to_uuid:25` | Moves funds between sub-accounts (requires portfolio UUIDs). |

### Order Tooling

These commands assume derivatives remain disabled unless
`COINBASE_ENABLE_DERIVATIVES=1` **and** INTX credentials are supplied.

| Command | Description |
|---------|-------------|
| `--preview-order` | Generates an order payload for `--order-symbol`, `--order-side`, `--order-type`, and quantity/price arguments, returning the JSON preview. |
| `--edit-order-preview ORDER_ID` | Requests an edit preview for an existing order. Requires the same option set as `--preview-order`. |
| `--apply-order-edit ORDER_ID:PREVIEW_ID` | Commits the preview identified above. |

For order tooling, combine flags such as:

```bash
poetry run perps-bot --profile spot \
  --preview-order \
  --order-symbol BTC-USD \
  --order-side buy \
  --order-type limit \
  --order-quantity 0.01 \
  --order-price 42000 \
  --order-tif GTC
```

### Environment Tips

- `TRADING_SYMBOLS="BTC-USD,ETH-USD"` seeds a default universe when
  `--symbols` is omitted.
- `PERPS_DEBUG=1` raises broker/orchestration logging to DEBUG without touching
  the global log level.
- `COINBASE_ENABLE_DERIVATIVES=1` only activates derivatives after INTX access
  is confirmed; keep it unset for spot-only deployments.
