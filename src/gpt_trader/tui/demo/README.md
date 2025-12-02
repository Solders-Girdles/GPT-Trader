# TUI Demo Mode

Demo mode allows you to test and develop the TUI without needing live exchange connections or real trading activity.

## Quick Start

Launch the TUI in demo mode:

```bash
uv run gpt-trader run --tui --demo
```

The demo bot will:
- Generate realistic market price movements
- Simulate trading activity (positions, orders, trades)
- Update the TUI with live data every 2 seconds
- **Never connect to real exchanges or execute real trades**

## Available Scenarios

Choose different market conditions to test various UI states:

### Mixed (Default)
Balanced scenario with both winning and losing positions:
```bash
uv run gpt-trader run --tui --demo --scenario mixed
```

### Winning Day
All positions profitable, positive P&L trending up:
```bash
uv run gpt-trader run --tui --demo --scenario winning
```

### Losing Day
Positions underwater, negative P&L:
```bash
uv run gpt-trader run --tui --demo --scenario losing
```

### High Volatility
Large price swings, rapid activity, many orders:
```bash
uv run gpt-trader run --tui --demo --scenario volatile
```

### Quiet Market
No positions, low activity, waiting for opportunities:
```bash
uv run gpt-trader run --tui --demo --scenario quiet
```

### Risk Limit
Approaching daily loss limits, risk controls active:
```bash
uv run gpt-trader run --tui --demo --scenario risk_limit
```

## Keybindings

While the TUI is running:

- `s` - Start/Stop the demo bot
- `p` - Trigger panic mode (simulated flatten & stop)
- `c` - Show configuration
- `l` - Focus logs
- `q` - Quit

## Testing UI Changes

Demo mode is perfect for:

1. **Rapid iteration** - No need for exchange API keys or real accounts
2. **Widget testing** - Verify data displays correctly
3. **Layout adjustments** - Test different screen sizes and layouts
4. **Error handling** - Scenario testing without risking real trades
5. **Performance** - Stress test with high update frequency

## Architecture

```
DemoBot
  └── DemoEngine
       ├── DemoStatusReporter
       │    └── MockDataGenerator (generates realistic data)
       └── DemoContext
            └── DemoRuntimeState
```

- **MockDataGenerator**: Creates realistic market data with random walks
- **DemoStatusReporter**: Mimics real StatusReporter with observer pattern
- **Scenarios**: Pre-configured states for different market conditions

## Development

To create new scenarios, edit `scenarios.py`:

```python
def my_custom_scenario() -> MockDataGenerator:
    generator = MockDataGenerator(
        symbols=["BTC-USD", "ETH-USD"],
        starting_equity=10000.0,
        total_equity=11000.0,  # +10% gain
    )

    # Configure positions, orders, etc.
    generator.positions = {...}

    return generator
```

Then register it in `SCENARIOS` dict and update the CLI choices.

## Files

- `mock_data.py` - Data generator with random walks and trade simulation
- `demo_bot.py` - Mock bot that mimics real TradingBot interface
- `scenarios.py` - Pre-configured market scenarios
- `README.md` - This file

## Notes

- Demo data updates every 2 seconds by default (configurable)
- Price movements use ±0.5% random walk
- Trade execution is simulated with 20% probability per cycle
- All data is ephemeral (not persisted to database)
- Perfect for screenshots, demos, and UI development
