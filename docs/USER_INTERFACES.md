# GPT-Trader User-Friendly Interfaces Guide

## Overview

GPT-Trader now includes a comprehensive set of user-friendly interfaces designed to make trading strategy development and deployment accessible to users of all experience levels.

## Quick Start

### First Time Setup

For new users, start with the setup wizard:

```bash
gpt-trader wizard
```

This will guide you through:
- Environment configuration
- Data source setup
- Strategy selection
- Risk management settings
- Profile creation

### Interactive Menu

Launch the main menu for easy navigation:

```bash
gpt-trader menu
# or simply
gpt-trader
```

The menu provides access to all features through an intuitive interface.

## New Commands

### 1. Interactive Menu (`menu`)

The main navigation hub for GPT-Trader:

```bash
gpt-trader menu
```

Features:
- Quick Start wizard
- Strategy Development tools
- Trading Operations
- Analysis & Reports
- Settings & Configuration

### 2. Trading Dashboard (`dashboard`)

Real-time monitoring and visualization:

```bash
# Overview dashboard
gpt-trader dashboard

# Live trading dashboard
gpt-trader dashboard --mode live

# Paper trading dashboard
gpt-trader dashboard --mode paper

# Backtest progress
gpt-trader dashboard --mode backtest
```

Dashboard modes:
- **Overview**: System status and active strategies
- **Live**: Real-time positions and P&L
- **Paper**: Paper trading simulation
- **Backtest**: Progress tracking during backtests

### 3. Setup Wizard (`wizard`)

Guided configuration for new users:

```bash
gpt-trader wizard
```

The wizard helps with:
- Directory setup
- Dependency checking
- API configuration
- Strategy selection
- Risk parameters
- Notification settings

### 4. Enhanced Help System (`help`)

Comprehensive help with examples:

```bash
# Quick reference
gpt-trader help

# Command-specific help
gpt-trader help backtest

# Tutorials
gpt-trader help --tutorial getting_started
gpt-trader help --tutorial strategies
gpt-trader help --tutorial risk_management

# FAQ
gpt-trader help --faq

# Search help content
gpt-trader help --search "moving average"
```

### 5. Command Shortcuts (`shortcuts`)

View and use command shortcuts:

```bash
# View all shortcuts
gpt-trader shortcuts

# Use shortcuts directly
gpt-trader bt    # Same as: gpt-trader backtest
gpt-trader qt    # Quick test
gpt-trader qb    # Quick backtest wizard
```

Available shortcuts:
- `bt` → backtest
- `opt` → optimize
- `wf` → walk-forward
- `pp` → paper
- `i` → interactive
- `qt` → quick test
- `qb` → quick backtest
- `qp` → quick paper trading

## Key Features

### 1. Rich Terminal UI

All interfaces use Rich library for enhanced visualization:
- Color-coded output
- Progress bars
- Live updating dashboards
- Formatted tables
- Interactive prompts

### 2. Progressive Disclosure

Information is presented in layers:
- Beginners see simplified options
- Advanced features available when needed
- Context-sensitive help

### 3. Guided Workflows

Step-by-step guidance for complex tasks:
- Backtest setup wizard
- Strategy optimization guide
- Paper trading setup
- Risk configuration

### 4. Smart Defaults

Sensible defaults for all parameters:
- Pre-configured strategies
- Standard risk settings
- Common symbol lists
- Typical date ranges

### 5. Profile Management

Save and reuse configurations:

```bash
# Create profile during wizard
gpt-trader wizard

# Use profile
gpt-trader backtest --profile myprofile
```

## Usage Examples

### Example 1: Complete Beginner

```bash
# First time setup
gpt-trader wizard

# Launch menu
gpt-trader

# Select "Quick Start" → "Run Simple Backtest"
# Follow prompts
```

### Example 2: Quick Testing

```bash
# Quick test with Apple stock
gpt-trader qt

# Quick backtest wizard
gpt-trader qb

# Quick paper trading
gpt-trader qp
```

### Example 3: Learning

```bash
# Get started tutorial
gpt-trader help --tutorial getting_started

# View strategy examples
gpt-trader help backtest

# Check FAQ
gpt-trader help --faq
```

### Example 4: Monitoring

```bash
# Live dashboard
gpt-trader dashboard --mode live

# Check system status
gpt-trader check

# View current positions
gpt-trader monitor --status
```

## Tips and Tricks

### 1. Keyboard Navigation

In interactive modes:
- **Arrow keys**: Navigate options
- **Enter**: Select
- **Ctrl+C**: Cancel/Exit
- **Tab**: Auto-complete

### 2. Quick Commands

Chain shortcuts for speed:

```bash
# Quick test then optimize
gpt-trader qt && gpt-trader opt

# Run backtest with shortcut
gpt-trader bt --symbol AAPL
```

### 3. Help Search

Find what you need quickly:

```bash
# Search for RSI information
gpt-trader help --search RSI

# Find optimization examples
gpt-trader help --search optimize
```

### 4. Profile Switching

Manage multiple configurations:

```bash
# Conservative profile
gpt-trader backtest --profile conservative

# Aggressive profile
gpt-trader backtest --profile aggressive
```

## Troubleshooting

### Common Issues

1. **Menu not launching**
   ```bash
   # Check installation
   gpt-trader --version

   # Run system check
   gpt-trader check
   ```

2. **Dashboard not updating**
   - Check terminal supports colors
   - Try `--no-color` flag
   - Ensure terminal width > 80 chars

3. **Wizard fails**
   - Run with verbose: `gpt-trader wizard -v`
   - Check permissions for ~/.gpt-trader
   - Ensure Python 3.8+

### Getting Help

```bash
# Built-in help
gpt-trader help

# System check
gpt-trader check

# Verbose mode for debugging
gpt-trader -vv <command>
```

## Advanced Features

### Custom Shortcuts

Add your own shortcuts:

```bash
# Add custom shortcut
gpt-trader shortcuts --add mytest "backtest --symbol TSLA --strategy demo_ma"

# Use custom shortcut
gpt-trader mytest
```

### Dashboard Customization

Configure dashboard refresh rate and layout in `~/.gpt-trader/config.yaml`:

```yaml
dashboard:
  refresh_rate: 1.0
  show_alerts: true
  max_positions: 10
  theme: dark
```

### Batch Operations

Use the menu system for batch operations:
1. Launch menu: `gpt-trader menu`
2. Select "Strategy Development"
3. Choose "Compare Strategies"
4. Select multiple strategies to test

## Summary

The new user-friendly interfaces make GPT-Trader accessible to everyone:

- **Beginners**: Start with `wizard` and `menu`
- **Regular Users**: Use `shortcuts` and `dashboard`
- **Power Users**: Leverage `help --search` and profiles
- **Everyone**: Enjoy the rich terminal UI and guided workflows

For the latest updates and features, run:

```bash
gpt-trader help --tutorial getting_started
```
