# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# 🛠️ Available Tools & Libraries

## Data & Analysis
- **pandas** (2.2.2): DataFrames and time series analysis
- **numpy** (1.26.4): Numerical computing
- **yfinance** (0.2.40): Yahoo Finance data API
- **ta-lib** (0.6.5): Technical analysis indicators
- **scipy** (1.14.0): Scientific computing

## Visualization
- **matplotlib** (3.9.0): Plotting library
- **seaborn** (0.13.0): Statistical visualization
- **rich** (13.7.0): Terminal formatting and tables

## Trading & Execution
- **alpaca-py** (0.20.0): Alpaca broker API
- **websockets** (14.0): Real-time data streams

## ML & Distributed Computing
- **ray** (2.48.0): Distributed computing and ML
- **scikit-learn**: Machine learning (check if installed)
- **torch/tensorflow**: Deep learning (check if installed)

## Infrastructure
- **pydantic** (2.7.4): Data validation
- **python-dotenv** (1.0.1): Environment variables
- **requests** (2.32.3): HTTP client
- **aiohttp** (3.12.15): Async HTTP
- **psutil** (6.0.0): System monitoring
- **pyyaml** (6.0.1): YAML parsing
- **pytz** (2024.1): Timezone handling
- **tqdm** (4.67.1): Progress bars

## Testing
- **pytest**: Testing framework (in dev dependencies)
- **pytest-asyncio**: Async test support
- **pytest-timeout**: Test timeouts

## Code Quality
- **black**: Code formatter
- **mypy**: Type checking
- **ruff**: Fast linter

## What's NOT Available (May Need)
- ❌ Database ORM (SQLAlchemy)
- ❌ Redis client
- ❌ Kafka client
- ❌ AWS SDK (boto3)
- ❌ Docker SDK
- ❌ Prometheus client
- ❌ OpenAI/Anthropic APIs
- ❌ Slack/Discord notifiers

## Quick Import Reference

```python
# Data manipulation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Data sources
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient

# Technical analysis
import talib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

# Async/Parallel
import asyncio
import aiohttp
import ray

# System
import psutil
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Validation
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Progress
from tqdm import tqdm
```

## Environment Variables (.env.template)
Check what's configured:
- API keys
- Database connections
- Feature flags
- Rate limits

## Missing Critical Tools?
If you need tools not listed here, options:
1. Check if already in poetry.lock
2. Request addition to pyproject.toml
3. Use built-in Python alternatives
4. Implement lightweight version locally