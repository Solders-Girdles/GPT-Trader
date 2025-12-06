# GPT-Trader TUI Guide

The **Terminal User Interface (TUI)** is the primary monitoring and control interface for GPT-Trader. Built with [Textual](https://textual.textualize.io/), it provides real-time visibility into the bot's state, account metrics, and trading activities directly from your terminal.

## Getting Started

Launch via the CLI entry point (preferred for env/logging setup):

```bash
uv run gpt-trader tui                 # Mode selector
uv run gpt-trader tui --mode demo     # Skip selection with a preset scenario
uv run gpt-trader tui --mode live     # Uses config/profiles/prod.yaml
```

Need the trading loop wired to a specific profile? The `run` command remains supported:

```bash
uv run gpt-trader run --profile dev --tui
```

## Technical Requirements

### StatusReporter Dependency

The TUI **requires** that the bot engine has a properly initialized `StatusReporter` instance. The TUI will fail-fast during startup if `bot.engine.status_reporter` is not available.

**Why:** The TUI relies on StatusReporter for:
- Real-time data updates via observer pattern
- Typed data contracts (BalanceEntry, DecisionEntry)
- Consistent state snapshots across all widgets

**What this means:**
- All bot engines must include StatusReporter initialization
- The TUI won't run in "degraded mode" with stale data
- You'll get a clear error message if StatusReporter is missing

### Data Type Contracts

The TUI uses strongly-typed dataclasses for all numeric data to ensure precision and eliminate parsing errors:

**Typed Contracts:**
- **BalanceEntry**: Account balances with `Decimal` amounts (total, available, hold)
- **DecisionEntry**: Strategy decisions with typed fields (symbol, action, reason, confidence, indicators, timestamp)
- **Decimal Types**: All prices, quantities, and P&L values use Python's `Decimal` type (not strings or floats)

**Benefits:**
- Type safety validated by mypy (see `status_reporter.py` and `state.py`)
- No defensive string→numeric parsing in widget code
- Consistent precision across all calculations
- Direct consumption of typed data from StatusReporter

## Interface Overview

The TUI is organized into a grid layout with several key widgets:

### 1. Header & Status
- **Bot Status**: Shows if the bot is `RUNNING` (Green) or `STOPPED` (Red).
- **Cycle**: Displays the current trading cycle timestamp.
- **Connection**: (Coming Soon) API latency and connection health.

### 2. Account Summary
Displays real-time account metrics:
- **Total Equity**: Current portfolio value in USD.
- **Available Balance**: Funds available for trading.
- **P&L**: Daily Profit and Loss.

### 3. Active Positions
A table listing all open positions:
- **Symbol**: e.g., BTC-USD.
- **Side**: LONG or SHORT.
- **Size**: Position size.
- **Unrealized PnL**: Current profit/loss for the position.

### 4. Recent Logs
A scrolling log window showing the latest bot activities, errors, and signals.

## Navigation & Controls

| Key | Action | Description |
|-----|--------|-------------|
| `s` | Start/Stop | Toggle the trading bot on or off. |
| `c` | Config | Open the configuration viewer. |
| `l` | Focus Logs | Expand/Focus the log widget for easier reading. |
| `p` | **PANIC** | **Emergency Stop**: Flattens all positions and stops the bot. |
| `q` | Quit | Exit the TUI (stops the bot if running). |

## Troubleshooting

### Common Issues

**TUI fails to start with RuntimeError about StatusReporter**
- **Cause**: The bot engine doesn't have StatusReporter initialized
- **Solution**: Ensure your bot engine setup includes StatusReporter creation
- **Example Error**: `RuntimeError: TUI requires bot.engine.status_reporter for data updates`

**TUI not starting?**
- Ensure you have installed the dependencies: `uv sync`
- Check that your bot configuration is valid
- Verify the bot engine is properly initialized

**Display issues?**
- The TUI works best in modern terminals like iTerm2, Alacritty, or Windows Terminal
- Ensure your terminal supports 256 colors
- Try resizing the terminal window if layouts appear broken

## Roadmap

See [TUI_ROADMAP.md](TUI_ROADMAP.md) for upcoming features and development phases.
