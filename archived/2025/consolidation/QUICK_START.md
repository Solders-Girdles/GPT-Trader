# Quick Start Guide

---
status: current
tested: 2024-12-31
python: 3.12+
---

## Prerequisites

- Python 3.12 or higher
- Poetry installed ([installation guide](https://python-poetry.org/docs/#installation))
- Git
- Coinbase account (for live trading)

## Installation (2 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/your-username/GPT-Trader.git
cd GPT-Trader
```

### 2. Install Dependencies
```bash
poetry install
```
*Expected: Installation of ~50 packages, takes 1-2 minutes*

### 3. Verify Installation
```bash
poetry run perps-bot --help
```
*Expected: Display of CLI help menu*

## Environment Setup (3 minutes)

### 1. Create Environment File
```bash
cp .env.template .env
```

### 2. Configure for Development (Mock Broker)
No API keys needed! The dev profile uses a mock broker:
```bash
# Already configured in .env.template:
BROKER=coinbase
COINBASE_SANDBOX=1
```

### 3. Configure for Live Trading (Optional)
Edit `.env` and add your CDP keys:
```bash
# Production (Perpetuals)
COINBASE_PROD_CDP_API_KEY=your_api_key_here
COINBASE_PROD_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----
your_private_key_here
-----END EC PRIVATE KEY-----"

# Sandbox (Spot only - no perpetuals)
COINBASE_SANDBOX_API_KEY=your_key
COINBASE_SANDBOX_API_SECRET=your_secret
COINBASE_SANDBOX_API_PASSPHRASE=your_passphrase
```

## Running the Bot (1 minute)

### Development Mode (Recommended First Run)
```bash
poetry run perps-bot --profile dev --dev-fast
```
*Expected: Bot starts with mock broker, shows random price movements*

### Canary Mode (Ultra-Safe Production)
```bash
# Dry run first
poetry run perps-bot --profile canary --dry-run

# Live with tiny positions
poetry run perps-bot --profile canary
```
*Expected: 0.01 BTC max positions, $10 daily loss limit*

### Production Mode
```bash
poetry run perps-bot --profile prod
```
*Expected: Full trading with production limits*

## Running Tests (1 minute)

### Quick Test (Active Code Only)
```bash
poetry run pytest tests/unit/bot_v2 tests/unit/test_foundation.py -q
```
*Expected: 220 tests, 100% pass rate, ~10 seconds*

### Full Test Suite
```bash
poetry run pytest tests/ -q
```
*Expected: 435 tests total (includes legacy), 69% overall pass rate*

## Stage 3 Multi-Asset Runner

For testing multiple assets simultaneously:
```bash
poetry run python scripts/stage3_runner.py --duration-minutes 60
```
*Expected: Runs BTC-PERP, ETH-PERP, SOL-PERP for 60 minutes*

## Common Commands

### Check System Status
```bash
python scripts/preflight_check.py
```

### Validate WebSocket Connection
```bash
python scripts/ws_probe.py
```

### Test Guard Triggers
```bash
python scripts/test_guard_triggers.py --test all
```

### Check Sandbox Balance
```bash
python scripts/check_sandbox_balance.py
```

## Troubleshooting

### Issue: "Command not found: poetry"
**Solution:** Install Poetry first
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Issue: "No module named 'bot_v2'"
**Solution:** Ensure you're in the project directory and dependencies are installed
```bash
cd GPT-Trader
poetry install
```

### Issue: "Authentication failed"
**Solution:** Check your API keys in `.env`
```bash
# For CDP keys, ensure private key includes headers
# For sandbox, ensure you're using Exchange API keys (not CDP)
```

### Issue: "WebSocket disconnected"
**Solution:** This is normal in dev mode. For production:
```bash
# Check WebSocket connectivity
python scripts/ws_probe.py --sandbox
```

## Next Steps

1. **Understand the Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Learn Trading Logic**: Review [Trading Logic - Perpetuals](reference/trading_logic_perps.md)
3. **Configure Strategies**: See configuration files in `config/`
4. **Production Deployment**: Follow [Production Guide](guides/production.md)

## Getting Help

- **Documentation**: [docs/README.md](README.md)
- **AI Development**: [docs/guides/agents.md](guides/agents.md)
- **Issues**: GitHub Issues page
- **Tests**: Run test suite for validation

---

*Remember: Always test in dev mode first, then canary, then production!*