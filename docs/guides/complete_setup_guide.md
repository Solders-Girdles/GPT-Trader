# Complete Setup Guide

---
status: current
created: 2025-01-01
last-updated: 2025-10-07
consolidates:
  - docs/QUICK_START.md
  - docs/guides/api_key_setup.md
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

2. **Poetry** (Package Manager)
   ```bash
   # Install Poetry if not present
   curl -sSL https://install.python-poetry.org | python3 -

   # Verify installation
   poetry --version
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
poetry install

# This installs ~50 packages and takes 1-2 minutes
```

### Step 3: Verify Installation
```bash
# Test import of main module
poetry run python -c "from bot_v2 import cli; print('✅ Installation successful')"
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

The `.env` file contains all configuration settings. Start with spot trading credentials, then add the derivatives block only if your account has INTX access.

```bash
# ============================================
# Coinbase Spot Configuration (Default)
# ============================================

COINBASE_API_KEY=your_hmac_api_key
COINBASE_API_SECRET=your_hmac_api_secret
COINBASE_ENABLE_DERIVATIVES=0      # remains 0 unless INTX access is granted

# Optional: enable paper/mock mode without real orders
# PERPS_PAPER=1

# ============================================
# Coinbase Derivatives (INTX Accounts Only)
# ============================================
# Uncomment once Coinbase approves INTX access
# COINBASE_ENABLE_DERIVATIVES=1
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

### Production Setup (Perpetuals / INTX Access)

Only complete this section if Coinbase has approved your account for INTX derivatives trading. Spot trading does **not** require CDP credentials.

1. **Generate CDP API Key**:
   - Log into Coinbase
   - Navigate to Settings → API
   - Create new CDP (Cloud Developer Platform) key
   - Save the key ID and private key securely

2. **Configure in .env**:
   ```bash
   COINBASE_API_KEY=organizations/your_org/apiKeys/your_key_id
   COINBASE_API_SECRET=-----BEGIN EC PRIVATE KEY-----
   MHcCAQEE...your full private key...
   -----END EC PRIVATE KEY-----
   ```

3. **Important Notes**:
   - Include the full private key with headers/footers
   - Never commit `.env` to version control
   - Keep backups of your keys securely

### Sandbox Setup (Testing Only)

1. **Get Sandbox Credentials**:
   - Visit [sandbox.exchange.coinbase.com](https://sandbox.exchange.coinbase.com)
   - Create sandbox API key
   - Note: Sandbox only supports spot trading, not perpetuals

2. **Configure for Sandbox**:
   ```bash
   COINBASE_SANDBOX=1
   COINBASE_API_KEY=your_sandbox_key
   COINBASE_API_SECRET=your_sandbox_secret
   COINBASE_PASSPHRASE=your_sandbox_passphrase
   ```

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
poetry run python scripts/production_preflight.py --profile canary

# Expected output: Python version, dependency checks, credential status, and risk toggle summary
```

### Step 2: Test with Development Profile
```bash
# Run with mock broker (no real trades)
poetry run perps-bot run --profile dev --dev-fast

# This runs for 60 seconds with simulated data
```

### Step 3: Test with Dry Run
```bash
# Test with real market data but no trades
poetry run perps-bot run --profile canary --dry-run

# Monitor the output for proper data reception
```

### Step 4: First Live Trade (Canary)
```bash
# When ready for real (tiny) trades
poetry run perps-bot run --profile canary

# This will trade with minimal position sizes
```

## Verification & Testing

### Run Test Suite
```bash
# Discover active tests (markers deselect optional suites)
poetry run pytest --collect-only -q
# Expected: 1555 collected / 1554 selected / 1 deselected / 0 skipped

# Quick targeted run for spot orchestration
poetry run pytest -q
```

### Check Streaming Health
```bash
# Smoke test the trading loop (mock broker)
poetry run perps-bot run --profile dev --dev-fast

# Inspect heartbeat metrics and mark timestamps
poetry run python scripts/perps_dashboard.py --profile dev --refresh 5 --window-min 5
```

### Monitor System Health
```bash
# Validate credentials, env, and risk settings
poetry run python scripts/production_preflight.py --profile canary

# Export Prometheus-compatible metrics
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/prod/metrics.json
```

## Troubleshooting

### Common Issues

#### 1. "Module not found" Errors
**Solution**: Ensure Poetry environment is activated
```bash
poetry shell  # Activate environment
# or prefix commands with: poetry run
```

#### 2. API Authentication Failures
**Solutions**:
- Verify API key format (CDP vs HMAC)
- Check environment variables are loaded: `echo $COINBASE_API_KEY`
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
poetry run python -c "import os; print(os.getenv('COINBASE_API_KEY')[:20])"

# Monitor real-time logs
tail -f var/logs/perps_bot.log

# Emergency stop
export RISK_KILL_SWITCH_ENABLED=1 && pkill -f perps-bot
```

### Getting Help

1. **Check Documentation**:
   - [Coinbase Integration Guide](../reference/coinbase_complete.md)
   - [Architecture Overview](../ARCHITECTURE.md)
   - [Trading Operations](../reference/trading_logic_perps.md)

2. **Review Logs**:
   - Main log: `var/logs/perps_bot.log`
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
   - Use metrics exporter: `poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/prod/metrics.json`
   - Track logs in `var/logs/perps_bot.log`
   - Schedule regular performance and risk reviews

---

*This guide consolidates all setup documentation into a single comprehensive resource. For additional details on specific topics, see the referenced documentation.*
