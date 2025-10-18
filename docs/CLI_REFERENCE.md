# GPT-Trader CLI Reference (Spot Profiles)

The legacy `gpt-trader` command suite has been removed from the tree; pull it
from repository history if needed. The supported interface is the `perps-bot`
CLI, which powers both spot
development and production runs while keeping derivatives behind the Coinbase
INTX gate.

## Primary Entry Point

```bash
poetry run perps-bot <command> [OPTIONS]
```

Commands: `run` (default if omitted), `account`, `orders`, and `treasury`.

### Run Command Options

| Option | Description |
|--------|-------------|
| `--profile {dev,demo,prod,canary,spot}` | Selects the configuration slice. `dev` defaults to mock trading, `spot` expects live keys. |
| `--dev-fast` | Runs a single control loop for smoke tests, then exits. |
| `--dry-run` | Forces non-mutating execution even when the profile targets Coinbase. |
| `--symbols BTC-USD ETH-USD` | Overrides the symbol universe (comma-free list). Defaults come from the selected profile or `TRADING_SYMBOLS`. |
| `--interval 5` | Sets the control-loop cadence in seconds (mapped to `update_interval`). |
| `--target-leverage 1` | Specifies desired leverage (derivatives-gated). |
| `--reduce-only` | Enables reduce-only behaviour for all orders. |
| `--tif {GTC,IOC,FOK}` | Overrides the default time in force. |
| `--enable-preview` | Requires interactive confirmation before new orders when running attached to a TTY. |
| `--account-interval 300` | Publishes account telemetry every N seconds. |

### Account & Treasury Utilities

| Command | Purpose |
|---------|---------|
| `poetry run perps-bot account snapshot` | Prints Coinbase limits, balances, and fee schedules then exits. |
| `poetry run perps-bot treasury convert --from USD --to USDC --amount 100` | Submits a convert quote and commits it. |
| `poetry run perps-bot treasury move --from-portfolio from_uuid --to-portfolio to_uuid --amount 25` | Moves funds between sub-accounts (requires portfolio UUIDs). |

### Order Tooling

Order tooling lives under the `orders` command and assumes
derivatives remain disabled unless `COINBASE_ENABLE_DERIVATIVES=1`
**and** INTX credentials are supplied.

| Command | Description |
|---------|-------------|
| `poetry run perps-bot orders preview --symbol BTC-USD --side buy --type limit --quantity 0.01 --price 42000` | Generates a preview for a new order and prints the JSON response. |
| `poetry run perps-bot orders edit-preview --order-id ORDER_ID --symbol BTC-USD --side buy --type limit --quantity 0.01 --price 42000` | Requests an edit preview for an existing order. |
| `poetry run perps-bot orders apply-edit --order-id ORDER_ID --preview-id PREVIEW_ID` | Commits a previously previewed edit. |

Order arguments accept the same flags as before (`--symbol`, `--side`, `--type`, `--quantity`, `--price`, `--stop`, `--tif`, `--client-id`, `--leverage`, `--reduce-only`).

### Environment Tips

- `TRADING_SYMBOLS="BTC-USD,ETH-USD"` seeds a default universe when
  `--symbols` is omitted.
- `PERPS_DEBUG=1` raises broker/orchestration logging to DEBUG without touching
  the global log level.
- `COINBASE_ENABLE_DERIVATIVES=1` only activates derivatives after INTX access
  is confirmed; keep it unset for spot-only deployments.

### Legacy Wrapper

The historical `scripts/stage3_runner.py` wrapper has been removed. Use the
`perps-bot` CLI directly for all workflows.
