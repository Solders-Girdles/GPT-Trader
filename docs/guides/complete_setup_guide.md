# Complete Setup Guide

---
status: current
created: 2025-01-01
last-updated: 2025-10-07
consolidates:
  - README.md
  - docs/reference/coinbase_auth_guide.md
  - config/environments/.env.template
  - Various scattered setup instructions
---

## Overview

This guide provides the complete setup process for GPT-Trader. The bot ships **spot trading by default** and keeps perpetuals logic in place for accounts that are approved for Coinbase INTX. Follow these steps sequentially for a smooth setup experience.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Environment Configuration](#environment-configuration)
4. [API Key Setup](#api-key-setup)
5. [Profile Configuration](#profile-configuration)
6. [First Run](#first-run)
7. [Verification & Testing](#verification--testing)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Python**: 3.12 or higher
- **Operating System**: macOS, Linux, or Windows with WSL2
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free space

### Required Software
1. **Python 3.12+**
   ```bash
   python --version  # Should show 3.12.0 or higher
   ```

2. **uv** (Package Manager)
   ```bash
   # Install uv if not present
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Verify installation
   uv --version
   ```

3. **Git**
   ```bash
   git --version
   ```

4. **Coinbase Account** (for live trading)
   - Sign up at [coinbase.com](https://www.coinbase.com)
   - Complete KYC verification
   - Enable API access

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/GPT-Trader.git
cd GPT-Trader
```

### Step 2: Install Dependencies
```bash
# Install all dependencies
uv sync

# This installs ~50 packages and takes 1-2 minutes
```

### Step 3: Verify Installation
```bash
# Test import of main module
uv run python -c "from gpt_trader import cli; print('✅ Installation successful')"
```

## Environment Configuration

### Step 1: Create Environment File
```bash
# Copy the template
cp config/environments/.env.template .env

# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

### Step 2: Configure Environment Variables

The `.env` file contains all configuration settings. Configure JWT credentials first, then enable the market modes that match your Coinbase access level.

```bash
# ============================================
# Coinbase Credentials (JWT)
# ============================================

# Preferred: JSON key file
COINBASE_CREDENTIALS_FILE=/path/to/cdp_key.json
# Or set both env vars:
# COINBASE_CDP_API_KEY=organizations/{org_id}/apiKeys/{key_id}
# COINBASE_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----..."

# Trading modes
TRADING_MODES=spot      # spot only by default
CFM_ENABLED=0           # enable CFM futures (US) when approved

# INTX perps (international) - only if Coinbase approves INTX access
COINBASE_ENABLE_INTX_PERPS=0
# Legacy alias (still supported): COINBASE_ENABLE_DERIVATIVES=0

# Optional: enable paper/mock mode without real orders
# PERPS_PAPER=1

# ============================================
# Coinbase INTX Perpetuals (INTX Accounts Only)
# ============================================
# Uncomment once Coinbase approves INTX access
# COINBASE_ENABLE_INTX_PERPS=1
# COINBASE_PROD_CDP_API_KEY=organizations/{org_id}/apiKeys/{key_id}
# COINBASE_PROD_CDP_PRIVATE_KEY="""-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"""

# ============================================
# Risk & Execution Controls
# ============================================
RISK_DAILY_LOSS_LIMIT=100          # USD
RISK_MAX_LEVERAGE=5                # Applied when derivatives enabled
DRY_RUN=1                          # Set to 0 for live trading
LOG_LEVEL=INFO                     # DEBUG for verbose output
```

## API Key Setup

### JWT Credentials (Spot + Perpetuals)

Spot and perpetual trading both use JWT credentials. INTX access is required only for perpetuals.

1. **Generate CDP API Key**:
   - Log into Coinbase
   - Navigate to Settings → API
   - Create new CDP (Cloud Developer Platform) key
   - Save the key ID and private key securely

2. **Configure in .env**:
   ```bash
   COINBASE_CREDENTIALS_FILE=/path/to/cdp_key.json
   # or
   COINBASE_CDP_API_KEY=organizations/your_org/apiKeys/your_key_id
   COINBASE_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"
   ```

3. **Important Notes**:
   - Include the full private key with headers/footers
   - Never commit `.env` to version control
   - Keep backups of your keys securely

### Sandbox Setup (Testing Only)

Coinbase Advanced Trade does not provide an authenticated sandbox. For testing:
- Use the mock broker profile: `uv run gpt-trader run --profile dev --dev-fast`
- Or set `MOCK_BROKER=1` for deterministic fills

## Profile Configuration

GPT-Trader includes three trading profiles:

### Development Profile (--profile dev)
- Uses mock broker (no real trades)
- Simulated market data
- Instant order fills
- Perfect for testing strategies

### Canary Profile (--profile canary)
- Real market data
- Tiny position sizes ($10-50)
- Extra safety checks
- Ideal for production testing

### Production Profile (--profile prod)
- Full trading capabilities
- Normal position sizes
- Standard risk limits
- For experienced users only

## First Run

### Step 1: Run Preflight Checks
```bash
# Verify environment, credentials, and risk settings
uv run python scripts/production_preflight.py --profile canary

# Expected output: Python version, dependency checks, credential status, and risk toggle summary
```

### Step 2: Test with Development Profile
```bash
# Run with mock broker (no real trades)
uv run gpt-trader run --profile dev --dev-fast

# This runs for 60 seconds with simulated data
```

### Step 3: Test with Dry Run
```bash
# Test with real market data but no trades
uv run gpt-trader run --profile canary --dry-run

# Monitor the output for proper data reception
```

### Step 4: First Live Trade (Canary)
```bash
# When ready for real (tiny) trades
uv run gpt-trader run --profile canary

# This will trade with minimal position sizes
```

## Verification & Testing

### Run Test Suite
```bash
# Discover active tests
uv run pytest --collect-only -q

# Quick targeted run for the spot trading loop
uv run pytest -q
```

### Check Streaming Health
```bash
# Smoke test the trading loop (mock broker)
uv run gpt-trader run --profile dev --dev-fast

# Inspect heartbeat metrics and mark timestamps
uv run python scripts/perps_dashboard.py --profile dev --refresh 5 --window-min 5
```

### Monitor System Health
```bash
# Validate credentials, env, and risk settings
uv run python scripts/production_preflight.py --profile canary

# Export Prometheus-compatible metrics
uv run python scripts/monitoring/export_metrics.py --profile prod --runtime-root .
```

## Troubleshooting

### Common Issues

#### 1. "Module not found" Errors
**Solution**: Ensure dependencies are installed
```bash
uv sync  # Install dependencies
# Then prefix commands with: uv run
```

#### 2. API Authentication Failures
**Solutions**:
- Verify CDP API key format (organizations/{org}/apiKeys/{key_id})
- Check environment variables are loaded: `echo $COINBASE_CDP_API_KEY`
- Ensure private key includes headers/footers
- Confirm production vs sandbox settings

#### 3. WebSocket Connection Issues
**Solutions**:
- Check network connectivity
- Verify firewall allows WSS connections
- Confirm heartbeat metrics advance in `scripts/perps_dashboard.py`

#### 4. No Market Data
**Solutions**:
- Confirm market hours (crypto trades 24/7)
- Check product subscriptions in logs
- Verify WebSocket is connected

#### 5. Orders Not Executing
**Solutions**:
- Confirm DRY_RUN=0 for live trading
- Check position size limits and daily loss caps
- Verify account has sufficient balance/margin (derivatives require INTX access)
- Review risk management settings

### Debug Commands

```bash
# Check environment variables
uv run python -c "import os; print((os.getenv('COINBASE_CDP_API_KEY') or '')[:20])"

# Monitor real-time logs
tail -f ${COINBASE_TRADER_LOG_DIR:-var/logs}/coinbase_trader.log

# Emergency stop
export RISK_KILL_SWITCH_ENABLED=1 && pkill -f gpt-trader
```

### Getting Help

1. **Check Documentation**:
   - [Coinbase Integration Guide](../reference/coinbase_complete.md)
   - [Architecture Overview](../ARCHITECTURE.md)
   - [Trading Operations](../reference/trading_logic_perps.md)

2. **Review Logs**:
   - Main log: `${COINBASE_TRADER_LOG_DIR:-var/logs}/coinbase_trader.log`
   - Error details with: `--log-level DEBUG`

3. **Community Support**:
   - GitHub Issues for bug reports
   - Discussions for questions

## Next Steps

After successful setup:

1. **Learn the System**:
   - Read [Architecture Documentation](../ARCHITECTURE.md)
   - Understand [Spot Trading Logic](../reference/trading_logic_perps.md) (perps sections apply once INTX is enabled)
   - Review [Risk Management](../reference/coinbase_complete.md#order-types--compatibility)

2. **Develop Strategies**:
   - Start with development/paper trading
   - Graduate to canary profile for live spot orders
   - Gradually increase position sizes once telemetry looks healthy

3. **Monitor Performance**:
   - Use metrics exporter: `uv run python scripts/monitoring/export_metrics.py --profile prod --runtime-root .`
   - Track logs in `${COINBASE_TRADER_LOG_DIR:-var/logs}/coinbase_trader.log`
   - Schedule regular performance and risk reviews

---

*This guide consolidates all setup documentation into a single comprehensive resource. For additional details on specific topics, see the referenced documentation.*
