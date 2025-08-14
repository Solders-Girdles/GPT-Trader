# Troubleshooting Guide

This guide helps you resolve common issues when using GPT-Trader. If you encounter a problem not covered here, please check the [GitHub Issues](https://github.com/your-username/GPT-Trader/issues) or open a new issue.

---

## 🚨 **Common Issues and Solutions**

### **Issue 1: Import Errors**

**Error:** `ImportError: attempted relative import beyond top-level package`

**Solution:**
```bash
# Run from the project root directory
cd /path/to/GPT-Trader
poetry run gpt-trader <command>
```

**Alternative Solution:**
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/GPT-Trader/src"
poetry run gpt-trader <command>
```

**Prevention:**
- Always run commands from the project root directory
- Use `poetry run` to ensure proper environment setup

### **Issue 2: Missing Required Arguments**

**Error:** `error: the following arguments are required: --symbols`

**Solution:**
```bash
# Provide required arguments
poetry run gpt-trader backtest --strategy trend_breakout --symbols AAPL,MSFT --start-date 2022-01-01 --end-date 2022-12-31

# Or use configuration profiles
poetry run gpt-trader backtest --profile my_profile
```

**Prevention:**
- Use configuration profiles for common settings
- Check the [Usage Guide](USAGE.md) for required parameters

### **Issue 3: Alpaca Credentials Not Found**

**Error:** `Alpaca credentials not found` or `Invalid API key`

**Solution:**
```bash
# Set environment variables
export ALPACA_API_KEY_ID="your_api_key_here"
export ALPACA_API_SECRET_KEY="your_secret_key_here"

# Or create .env file in project root
echo "ALPACA_API_KEY_ID=your_api_key_here" > .env
echo "ALPACA_API_SECRET_KEY=your_secret_key_here" >> .env
```

**Verification:**
```bash
# Test Alpaca connection
poetry run gpt-trader paper --symbols AAPL --strategy trend_breakout --dry-run
```

**Common Alpaca Issues:**
- **Paper Trading Only**: Ensure you're using paper trading API keys, not live trading
- **Account Status**: Check your Alpaca account is active and approved
- **API Limits**: Free accounts have rate limits; consider upgrading for heavy usage

### **Issue 4: Module Not Found**

**Error:** `ModuleNotFoundError: No module named 'bot.optimization.deployment_pipeline'`

**Solution:**
```bash
# Reinstall the package
poetry install

# Check if all dependencies are installed
poetry show

# Update dependencies
poetry update
```

**Alternative Solution:**
```bash
# Install missing dependencies manually
poetry add rich pytz pyyaml
```

### **Issue 5: Data File Not Found**

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'data/optimization/results.csv'`

**Solution:**
```bash
# Create the data directory structure
mkdir -p data/optimization
mkdir -p data/deployment
mkdir -p data/backtests
mkdir -p data/evolution

# Run optimization first to generate results
poetry run gpt-trader optimize-new --name test --strategy trend_breakout --symbols AAPL,MSFT --start-date 2023-01-01 --end-date 2023-12-31
```

### **Issue 6: Memory Issues**

**Error:** `MemoryError` or `Killed` during optimization

**Solution:**
```bash
# Reduce optimization scope
poetry run gpt-trader optimize-new \
  --name "small_optimization" \
  --strategy trend_breakout \
  --symbols "AAPL,MSFT" \
  --start-date 2023-01-01 \
  --end-date 2023-06-30 \
  --method grid \
  --max-combinations 1000
```

**Alternative Solutions:**
- Use smaller date ranges
- Reduce the number of symbols
- Use evolutionary optimization instead of grid search
- Increase system memory or use swap space

### **Issue 7: Slow Performance**

**Problem:** Optimization or backtesting is very slow

**Solutions:**
```bash
# Use parallel processing (if available)
poetry run gpt-trader optimize-new --method grid --parallel

# Use evolutionary optimization for large parameter spaces
poetry run gpt-trader optimize-new --method evolutionary --generations 30 --population-size 20

# Reduce data scope
poetry run gpt-trader backtest --symbols "AAPL,MSFT" --start-date 2023-01-01 --end-date 2023-06-30
```

**Performance Tips:**
- Use SSD storage for data files
- Close other applications to free memory
- Use smaller parameter ranges for initial testing

### **Issue 8: Strategy Not Trading**

**Problem:** Strategy shows no trades or very few trades

**Diagnosis:**
```bash
# Check strategy parameters
poetry run gpt-trader backtest --strategy trend_breakout --symbols AAPL --start-date 2022-01-01 --end-date 2022-12-31 --verbose

# Test with different parameters
poetry run gpt-trader backtest --strategy trend_breakout --symbols AAPL --donchian 20 --atr-k 1.5
```

**Common Causes:**
- **Parameter too restrictive**: Try different parameter combinations
- **Market conditions**: Strategy may not work in all market regimes
- **Data issues**: Check for missing or invalid data
- **Regime filters**: Disable regime filters to see if they're blocking trades

### **Issue 9: Poor Strategy Performance**

**Problem:** Strategy has low Sharpe ratio or high drawdown

**Solutions:**
```bash
# Run optimization to find better parameters
poetry run gpt-trader optimize-new --strategy trend_breakout --symbols AAPL,MSFT,GOOGL

# Use walk-forward validation
poetry run gpt-trader walk-forward --results results.csv --symbols AAPL,MSFT,GOOGL

# Try different strategies
poetry run gpt-trader backtest --strategy demo_ma --symbols AAPL,MSFT,GOOGL
```

**Performance Improvement Tips:**
- Use longer backtest periods (2+ years)
- Include transaction costs in analysis
- Test across multiple symbols and time periods
- Consider market regime filters

### **Issue 10: CLI Command Not Found**

**Error:** `command not found: gpt-trader`

**Solution:**
```bash
# Install the package in development mode
poetry install

# Or install globally
pip install -e .

# Verify installation
poetry run gpt-trader --help
```

### **Issue 11: Data Download Issues**

**Error:** `yfinance` download failures or missing data

**Solutions:**
```bash
# Update yfinance
poetry update yfinance

# Try alternative data source
poetry run gpt-trader backtest --data-source alpaca --symbols AAPL,MSFT

# Check symbol names
poetry run gpt-trader backtest --symbols "AAPL,MSFT,GOOGL" --validate-symbols
```

**Common Data Issues:**
- **Symbol delisting**: Some symbols may no longer exist
- **Market hours**: Data may be missing outside market hours
- **Rate limits**: yfinance has rate limits; add delays between requests

### **Issue 12: Paper Trading Issues**

**Problem:** Paper trading not executing trades or showing errors

**Diagnosis:**
```bash
# Test with dry run
poetry run gpt-trader paper --symbols AAPL --strategy trend_breakout --dry-run

# Check Alpaca account status
poetry run gpt-trader paper --symbols AAPL --strategy trend_breakout --check-account
```

**Common Issues:**
- **Market closed**: Paper trading only works during market hours
- **Insufficient buying power**: Check account balance
- **Invalid orders**: Ensure symbol names are correct
- **Rate limits**: Alpaca has API rate limits

---

## 🔧 **Testing Your Setup**

### **Step 1: Run Basic Tests**
```bash
# Test basic functionality
poetry run gpt-trader --version
poetry run gpt-trader --help

# Test backtest command
poetry run gpt-trader backtest --strategy trend_breakout --symbols AAPL --start-date 2023-01-01 --end-date 2023-01-31 --dry-run
```

### **Step 2: Test Optimization**
```bash
# Test small optimization
poetry run gpt-trader optimize-new \
  --name "test_optimization" \
  --strategy trend_breakout \
  --symbols "AAPL" \
  --start-date 2023-01-01 \
  --end-date 2023-03-31 \
  --method grid \
  --max-combinations 10
```

### **Step 3: Test Paper Trading (if Alpaca configured)**
```bash
# Test paper trading connection
poetry run gpt-trader paper --symbols AAPL --strategy trend_breakout --dry-run --duration 1
```

---

## 📋 **Environment Checklist**

### **Required Software**
- [ ] Python 3.9+ installed
- [ ] Poetry installed (`pip install poetry`)
- [ ] Git installed
- [ ] Sufficient disk space (1GB+ recommended)

### **Required Configuration**
- [ ] Project cloned and dependencies installed (`poetry install`)
- [ ] Data directories created (`mkdir -p data/{optimization,deployment,backtests}`)
- [ ] Alpaca credentials set (for paper trading)
- [ ] Pre-commit hooks installed (`pre-commit install`)

### **Optional but Recommended**
- [ ] Alpaca account for paper trading
- [ ] Slack webhook for monitoring alerts
- [ ] Additional Python packages for advanced features

---

## 🆘 **Getting Additional Help**

### **Before Asking for Help**
1. **Check this guide** for your specific error
2. **Search existing issues** on GitHub
3. **Try the examples** in the `examples/` directory
4. **Run the test suite** to verify your setup

### **When Reporting Issues**
Please include:
- **Error message** (full traceback)
- **Command used** (exact command line)
- **Environment details** (OS, Python version, etc.)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**

### **Support Channels**
- **GitHub Issues**: [Create an issue](https://github.com/your-username/GPT-Trader/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/your-username/GPT-Trader/discussions)
- **Documentation**: Check the [docs/](docs/) directory
- **Examples**: Review the [examples/](../examples/) directory

---

## 🔄 **Common Workarounds**

### **If Poetry is Slow**
```bash
# Use pip instead
pip install -r requirements.txt
python -m bot.cli <command>
```

### **If Data Download is Slow**
```bash
# Use cached data
poetry run gpt-trader backtest --use-cache --symbols AAPL,MSFT

# Or download data separately
poetry run gpt-trader download-data --symbols AAPL,MSFT,GOOGL --start-date 2020-01-01
```

### **If Optimization Takes Too Long**
```bash
# Use smaller parameter ranges
poetry run gpt-trader optimize-new --method grid --max-combinations 100

# Or use evolutionary optimization
poetry run gpt-trader optimize-new --method evolutionary --generations 20
```

---

*Last updated: December 2024*
