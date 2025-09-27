# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# Developer Setup Guide

## Quick Setup (For You Right Now)

Since we just removed `.venv`, you need to set up your environment:

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
source .venv/bin/activate    # Mac/Linux
# OR
.venv\Scripts\activate        # Windows

# 3. Install Poetry (Python package manager)
pip install poetry

# 4. Install project dependencies
cd config
poetry install
cd ..

# 5. Create your environment file
cp .env.template .env
# Edit .env with your API keys if needed

# 6. Test the setup
poetry run python src/bot_v2/test_all_slices.py
```

## Understanding the Structure

### Where Things Are
```
GPT-Trader/
├── config/           # Poetry and project config
│   ├── pyproject.toml
│   ├── poetry.lock
│   └── pytest.ini
├── src/bot_v2/      # All code
├── .knowledge/      # Documentation (you are here)
└── .env.template    # Environment template
```

### Running Commands

All Python commands use Poetry from the config directory:
```bash
# Format: poetry run python [file]
poetry run python src/bot_v2/test_backtest.py

# Or with pytest
cd config && poetry run pytest ../src/bot_v2/
```

## For Agent Tasks

When agents give you commands to run:
1. Make sure your virtual environment is activated
2. Commands starting with `poetry run` should work
3. If a command fails, check you're in the right directory

## Common Issues

### "poetry: command not found"
```bash
# Install poetry in your virtual environment
pip install poetry
```

### "No module named 'xxx'"
```bash
# Install dependencies
cd config && poetry install
```

### ".venv/bin/activate: No such file"
```bash
# Create the virtual environment first
python -m venv .venv
```

## Environment Variables

The `.env.template` shows what variables you might need:
- `ALPACA_API_KEY_ID` - For trading (optional for testing)
- `DATABASE_*` - Database config (optional for basic use)

For basic testing, you don't need any of these set.

## Testing Your Setup

Run this to verify everything works:
```bash
# Should show all slices passing
poetry run python src/bot_v2/test_all_slices.py
```

Expected output:
```
Testing backtest slice...
✅ backtest imports successfully
✅ backtest isolation verified
[... more slices ...]
System: 100% operational
```