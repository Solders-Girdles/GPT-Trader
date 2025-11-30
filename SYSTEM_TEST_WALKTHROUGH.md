# System Operation Test Walkthrough

## Objective
Verify the system's operational readiness by executing a full "dry run" of the trading bot using the CLI.

## Prerequisites
- Unit and Integration tests passing.
- `dev` profile configured for mock execution.

## Execution Steps

### 1. Fix Configuration Loading
Identified and fixed a bug in `src/gpt_trader/cli/services.py` where the `execution` section of the profile YAML (containing `mock_broker` and `dry_run` settings) was being ignored.
Also updated `src/gpt_trader/cli/options.py` to include `paper` as a valid profile choice.

### 2. Run CLI Command
Executed the following command to start the bot in development mode with a fast interval (1s) and single-cycle execution:

```bash
python -m gpt_trader.cli run --profile dev --dev-fast
```

### 3. Verify Output
The bot successfully started and produced the following logs:

```log
2025-11-30 01:52:52,265 - gpt_trader.orchestration.deterministic_broker - INFO - DeterministicBroker initialized with equity=100000
2025-11-30 01:52:52,270 - gpt_trader.features.live_trade.factory - INFO - Creating BaselinePerpsStrategy (RSI + MA crossover)
2025-11-30 01:52:52,270 - gpt_trader.orchestration.trading_bot.bot - INFO - TradingBot starting with symbols: ['BTC-USD', 'ETH-USD']
...
2025-11-30 01:52:52,281 - gpt_trader.features.live_trade.engines.strategy - INFO - Strategy Decision for BTC-USD: Action.HOLD (Insufficient data: 6/20 periods)
...
2025-11-30 01:52:53,376 - gpt_trader.monitoring.status_reporter - INFO - Status reporter stopped
```

## Findings
- **Configuration**: The `dev` profile correctly sets `mock_broker=True` (after fix).
- **Initialization**: `ApplicationContainer` correctly wires `DeterministicBroker` when `mock_broker` is enabled.
- **Strategy**: `BaselinePerpsStrategy` initializes and processes ticks.
- **Data Flow**: Mock market data is generated and fed to the strategy.
- **Shutdown**: The bot shuts down gracefully after the cycle completes.

## Next Steps
- **Paper Trading**: To test with real market data, configure `COINBASE_CREDENTIALS_FILE` and run with `--profile paper`.
- **Live Trading**: Once paper trading is verified, proceed to live trading with small capital.

## Paper Trading Test

### 1. Objective
Verify the system's ability to connect to real market data (Coinbase Advanced Trade) and execute simulated trades using the `paper` profile.

### 2. Fixes Implemented
- **ApplicationContainer**: Updated `src/gpt_trader/app/container.py` to support `COINBASE_CDP_API_KEY` and `COINBASE_CDP_PRIVATE_KEY` environment variables (used by CDP keys).
- **CoinbaseClient**: Updated `src/gpt_trader/features/brokerages/coinbase/client/client.py` to:
    - Implement `list_positions` returning `list[Position]` (adapting raw API dict).
    - Implement `list_balances` returning `list[Balance]` (adapting raw API dict).
    - Override `get_ticker` to normalize the API response (extracting `price` from `trades` list if missing at top level).

### 3. Execution
Executed the following command:
```bash
python -m gpt_trader.cli run --profile paper --dev-fast
```

### 4. Verification
The bot successfully connected to Coinbase and fetched real-time prices:
```log
2025-11-30 01:58:38,960 - gpt_trader.features.live_trade.engines.strategy - INFO - BTC-USD price: 91209.54
2025-11-30 01:58:39,090 - gpt_trader.features.live_trade.engines.strategy - INFO - ETH-USD price: 3007.24
```
The strategy processed the data and made decisions (HOLD due to insufficient history).

### 5. Conclusion
The system is now capable of running in paper trading mode with real market data. The `CoinbaseClient` adapter layer is functioning correctly to bridge the gap between the raw API and the `TradingEngine`'s domain model.

## Safety Improvements

### 1. Risk Manager Persistence
Implemented state persistence for `LiveRiskManager` to prevent "amnesia" on restarts.
-   **File**: `var/data/risk_state.json`
-   **Persisted Data**: `start_of_day_equity`, `daily_pnl_triggered`, `reduce_only_mode`.
-   **Behavior**: On startup, loads state if the date matches. If the bot crashes and restarts, it remembers the initial equity to correctly enforce the daily loss limit.

### 2. Configurable Order Validator
Refactored `OrderValidator` to accept dynamic limits from the bot configuration instead of using hard-coded values.
-   **Benefit**: Allows different risk profiles (e.g., higher leverage for specific strategies) without code changes.
-   **Integration**: `TradingEngine` now passes `config.risk` limits to the validator.

### 3. Order Reconciliation
Added an `_audit_orders` step to the trading cycle.
-   **Function**: Fetches open orders from the broker every cycle.
-   **Benefit**: Provides visibility into "blind" execution states (e.g., orders that were placed but not tracked internally due to network errors).
-   **Logs**: `AUDIT: Found X OPEN orders` or silence if none.

### 4. Verification
Verified changes by running the `dev` profile:
-   Confirmed `var/data/risk_state.json` is created and updated.
-   Confirmed `_audit_orders` runs without errors (added `list_orders` mock to `DeterministicBroker`).
