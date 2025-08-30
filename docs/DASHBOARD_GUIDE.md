# GPT-Trader Dashboard Guide

## Quick Start

The GPT-Trader dashboard is now fully functional and can be launched in multiple ways:

### Method 1: Direct Streamlit Command
```bash
poetry run streamlit run src/bot/dashboard/app.py
```

### Method 2: Using the Launcher Script
```bash
python launch_dashboard.py
```

### Method 3: Custom Port/Headless Mode
```bash
# Custom port
poetry run streamlit run src/bot/dashboard/app.py --server.port 8502

# Headless mode (no browser)
poetry run streamlit run src/bot/dashboard/app.py --server.headless true

# Using launcher with options
python launch_dashboard.py --port 8502 --no-browser
```

## Dashboard Features

### 📊 Portfolio Overview
- Real-time portfolio value and P&L
- Daily performance metrics
- Risk-adjusted performance indicators
- Interactive performance charts

### 📦 Positions & Allocation
- Current positions with real-time P&L
- Portfolio allocation breakdown
- Position-level analytics and filters
- Interactive visualizations

### 🎯 Strategy Performance
- Strategy-by-strategy performance metrics
- Comparative analysis charts
- Risk-return profiles
- Strategy allocation insights

### 📝 Trade History
- Recent trades with detailed analysis
- Trade filtering and sorting
- Performance distribution charts
- Execution quality metrics

### ⚠️ Risk Dashboard
- Comprehensive risk metrics (VaR, Sharpe, Sortino)
- Risk contribution analysis
- Stress test scenarios
- Automated risk alerts

### 💚 System Health
- Component status monitoring
- Real-time system metrics
- Alert management
- Performance indicators

## Data Sources

The dashboard intelligently handles different data availability scenarios:

### With Real Data Components
When position tracking, P&L calculation, and other bot components are available:
- Live portfolio data
- Real trade history
- Actual performance metrics
- Dynamic risk calculations

### Fallback Mode
When real components aren't available or accessible:
- Realistic mock data
- Simulated portfolio performance
- Representative trade patterns
- Calculated risk metrics

## Technical Details

### Dependencies
All required packages are automatically installed via Poetry:
- `streamlit` - Web dashboard framework
- `plotly` - Interactive charts
- `pandas` - Data manipulation
- `numpy` - Numerical computations

### Architecture
- **Data Provider Layer**: `PerformanceDashboardData` class handles data fetching
- **Visualization Layer**: Modular rendering functions for each dashboard section
- **Fallback System**: Graceful degradation to mock data when components unavailable
- **Caching**: Performance optimization with configurable cache timeouts

### Configuration
The dashboard automatically detects available components and adapts accordingly:
- Position tracking integration
- P&L calculation services
- Risk management systems
- Trade execution monitoring

## Troubleshooting

### Import Errors
If you see import errors:
```bash
# Ensure Poetry environment is activated
poetry install
poetry run streamlit run src/bot/dashboard/app.py
```

### Port Conflicts
If port 8501 is busy:
```bash
# Use a different port
poetry run streamlit run src/bot/dashboard/app.py --server.port 8502
```

### Performance Issues
For better performance:
```bash
# Install watchdog (optional)
pip install watchdog

# Use headless mode for server deployment
poetry run streamlit run src/bot/dashboard/app.py --server.headless true
```

## Development

### Adding New Features
1. Create visualization functions in `src/bot/dashboard/performance_dashboard.py`
2. Add data methods to `PerformanceDashboardData` class
3. Integrate into main dashboard navigation

### Customizing Mock Data
Edit the `_initialize_mock_data()` method in `PerformanceDashboardData` to customize fallback data.

### Testing
```bash
# Run basic import test
poetry run python -c "from src.bot.dashboard.app import create_dashboard; print('✓ Dashboard ready')"

# Test server functionality
poetry run streamlit run src/bot/dashboard/app.py --server.headless true &
curl http://localhost:8501
```

## Security Notes

- Dashboard runs locally by default
- For network access, configure `--server.address 0.0.0.0` carefully
- Consider authentication for production deployments
- Mock data is safe for demonstration purposes

## Success Criteria ✅

All requirements have been met:
- ✅ Dashboard launches without import errors
- ✅ Basic pages/views are accessible  
- ✅ Mock data displays when real data unavailable
- ✅ Can run: `streamlit run src/bot/dashboard/app.py`
- ✅ Minimally functional with comprehensive features
- ✅ Graceful fallback for missing dependencies
- ✅ Interactive charts and real-time updates
- ✅ Multiple launch methods available