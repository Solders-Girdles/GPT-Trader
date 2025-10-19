# Compatibility & Troubleshooting Reference

---
status: current
last-updated: 2025-10-12
consolidates:
  - COMPATIBILITY.md
---

## System Compatibility

### Python Version Requirements
- **Minimum**: Python 3.12+
- **Recommended**: Python 3.12.1 or later
- **Package Manager**: Poetry 1.0+

### Operating System Support
| OS | Status | Notes |
|-----|--------|-------|
| macOS 12+ | ✅ Fully Supported | Native development environment |
| Ubuntu 20.04+ | ✅ Fully Supported | Recommended for production |
| Windows 11 | ⚠️ Limited Support | WSL2 recommended |
| Docker | ✅ Fully Supported | See deployment guide |

### Hardware Requirements

#### Minimum Requirements
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Storage: 1 GB free space
- Network: Stable internet connection

#### Recommended for Production
- CPU: 4+ cores, 3.0+ GHz
- RAM: 8+ GB
- Storage: 10+ GB SSD
- Network: Low-latency connection (<50ms to Coinbase)

### Dependencies

#### Core Dependencies
```toml
python = "^3.12"
pandas = "^2.2.0"
numpy = "^1.26.0"
requests = "^2.32.0"
websockets = ">=13.0,<14.0"
pydantic = "^2.7.0"
```

#### Optional Dependencies
```toml
# For ML strategies
scikit-learn = "^1.3.0"
xgboost = "^1.7.0"

# For advanced analytics
matplotlib = "^3.7.0"
plotly = "^5.15.0"

# For performance monitoring
psutil = "^6.0.0"
```

> **Coinbase SDK Dependency Note**
> The official `coinbase-advanced-py` SDK (v1.8.x) currently pins `websockets` to `<14`.
> To stay compatible with their WebSocket implementation we lock the project to `websockets >=13,<14`.
> If you upgrade `websockets`, verify that Coinbase has published a corresponding SDK release first.

## Environment Compatibility

### Coinbase API Versions

| Environment | API Version | Authentication | Products |
|-------------|-------------|----------------|----------|
| Production (default) | Advanced Trade v3 | HMAC | Spot (BTC-USD, ETH-USD, …) |
| Production (perps, INTX) | Advanced Trade v3 | CDP JWT | Perpetuals (BTC-PERP, ETH-PERP, …) |
| Sandbox | Exchange v2 | HMAC | Spot only (no perps) |

### Network Requirements
- Outbound HTTPS (port 443) to api.coinbase.com
- WebSocket connections to advanced-trade-ws.coinbase.com
- No inbound connections required

### Firewall Configuration
```bash
# Allow outbound HTTPS
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT

# Allow WebSocket connections
iptables -A OUTPUT -p tcp --dport 443 -d advanced-trade-ws.coinbase.com -j ACCEPT
```

## Common Compatibility Issues

### Python Version Issues
**Problem**: ImportError or syntax errors
**Cause**: Python version < 3.12
**Solution**: Upgrade Python or use pyenv

```bash
# Check Python version
python --version

# Install Python 3.12 with pyenv
pyenv install 3.12.1
pyenv local 3.12.1
```

### Poetry Installation Issues
**Problem**: Poetry not found or outdated
**Solution**: Install/update Poetry

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Update Poetry
poetry self update
```

### SSL Certificate Issues
**Problem**: SSL verification errors
**Cause**: Outdated certificates or corporate proxy
**Solutions**:

1. Update certificates:
```bash
# macOS
brew install ca-certificates

# Ubuntu
sudo apt-get update && sudo apt-get install ca-certificates
```

2. Corporate proxy setup:
```bash
# Set proxy environment variables
export HTTPS_PROXY=https://proxy.company.com:8080
export REQUESTS_CA_BUNDLE=/path/to/corporate/ca-bundle.crt
```

### WebSocket Connection Issues

#### Behind Corporate Firewalls
**Symptoms**: WebSocket connections timeout
**Solutions**:
1. Request firewall rule for advanced-trade-ws.coinbase.com:443
2. Use HTTP polling fallback if available
3. Configure proxy tunnel

#### Connection Drops
**Symptoms**: Frequent WebSocket disconnections
**Solutions**:
1. Enable automatic reconnection
2. Implement heartbeat/keepalive
3. Check network stability

### Memory Issues
**Problem**: High memory usage or out-of-memory errors
**Causes**: Large datasets, memory leaks
**Solutions**:

```python
# Configure pandas for lower memory usage
pd.options.mode.copy_on_write = True

# Use chunking for large datasets
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    process_chunk(chunk)

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## Performance Optimization

### Database Performance
- Use appropriate indexes
- Implement connection pooling
- Regular VACUUM operations (PostgreSQL)

### Network Optimization
- Enable HTTP/2 keep-alive
- Use connection pooling
- Implement request batching where possible

### CPU Optimization
- Use multiprocessing for CPU-intensive tasks
- Profile code to identify bottlenecks
- Consider Cython for hot paths

## Deployment Compatibility

### Docker Support
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

# Copy application
COPY . .

CMD ["poetry", "run", "coinbase-trader", "--profile", "prod"]
```

### Cloud Platform Support

#### AWS
- ✅ EC2 instances (t3.medium or larger)
- ✅ ECS/Fargate containers
- ✅ Lambda (for lightweight tasks)

#### Google Cloud
- ✅ Compute Engine
- ✅ Cloud Run
- ✅ GKE clusters

#### Azure
- ✅ Virtual Machines
- ✅ Container Instances
- ✅ App Service

## Testing Compatibility

### Test Environment Setup
```bash
# Install test dependencies
poetry install --with dev

# Run core unit suite
poetry run pytest -q

# Check environment and credentials
poetry run python scripts/production_preflight.py --profile canary
```

### CI/CD Pipeline Support
- ✅ GitHub Actions
- ✅ GitLab CI
- ✅ Jenkins
- ✅ CircleCI

Example GitHub Actions workflow:
```yaml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]

    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install Poetry
      run: pip install poetry

    - name: Install dependencies
      run: poetry install

    - name: Run tests
      run: poetry run pytest
```

## Version Compatibility Matrix

| Component | Min Version | Recommended | Notes |
|-----------|-------------|-------------|-------|
| Python | 3.12.0 | 3.12.1+ | Required for type hints |
| Poetry | 1.0.0 | 1.6.0+ | For dependency management |
| pandas | 2.0.0 | 2.1.0+ | For data processing |
| numpy | 1.24.0 | 1.25.0+ | Numerical operations |
| requests | 2.31.0 | Latest | HTTP client |
| websockets | 11.0.0 | Latest | WebSocket client |

## Getting Help

### Diagnostic Commands
```bash
# Full system check (env + credentials)
poetry run python scripts/production_preflight.py --profile canary

# Check specific component
python -c "import sys; print(sys.version)"
python -c "import pandas; print(pandas.__version__)"

# Test Coinbase connectivity
poetry run coinbase-trader run --profile dev --dev-fast
```

### Support Resources
- Python compatibility: https://docs.python.org/3/whatsnew/
- Poetry troubleshooting: https://python-poetry.org/docs/troubleshooting/
- Coinbase API status: https://status.coinbase.com/
