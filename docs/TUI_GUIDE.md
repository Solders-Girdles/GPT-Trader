# GPT-Trader TUI Guide

The **Terminal User Interface (TUI)** is the primary monitoring and control interface for GPT-Trader. Built with [Textual](https://textual.textualize.io/), it provides real-time visibility into the bot's state, account metrics, and trading activities directly from your terminal.

## Getting Started

To launch the TUI, use the `--tui` flag when running the bot:

```bash
uv run coinbase-trader run --tui
```

You can also specify a profile:

```bash
uv run coinbase-trader run --profile canary --tui
```

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

- **TUI not starting?** Ensure you have installed the dependencies: `uv sync`.
- **Display issues?** The TUI works best in modern terminals like iTerm2, Alacritty, or Windows Terminal.

## Roadmap

See [TUI_ROADMAP.md](TUI_ROADMAP.md) for upcoming features and development phases.
