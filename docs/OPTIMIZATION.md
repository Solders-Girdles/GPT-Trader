# Optimization Framework

The GPT-Trader optimization framework provides a comprehensive system for parameter optimization of trading strategies. It supports both grid search and evolutionary algorithms, with extensive analysis and visualization capabilities.

## Features

### 🎯 **Multiple Optimization Methods**
- **Grid Search**: Systematic exploration of parameter combinations
- **Evolutionary Search**: Genetic algorithm optimization
- **Hybrid Approach**: Combine both methods for comprehensive optimization

### 📊 **Advanced Analysis**
- Parameter sensitivity analysis
- Correlation analysis between parameters and performance
- Robustness testing
- Statistical summaries and recommendations

### 📈 **Rich Visualizations**
- Parameter sensitivity plots
- Optimization progress tracking
- Correlation matrices
- Performance scatter plots
- Interactive HTML dashboard

### ⚙️ **Flexible Configuration**
- Type-safe configuration with validation
- Support for multiple strategies
- Walk-forward testing
- Parallel processing
- Early stopping mechanisms

## Quick Start

### 1. Basic Grid Search

```python
from bot.optimization.config import OptimizationConfig, ParameterSpace, get_trend_breakout_config
from bot.optimization.engine import OptimizationEngine

# Create configuration
strategy_config = get_trend_breakout_config()
parameter_space = ParameterSpace(
    strategy=strategy_config,
    grid_ranges={
        "donchian_lookback": [40, 55, 70],
        "atr_k": [1.5, 2.0, 2.5],
        "entry_confirm": [1, 2],
    }
)

config = OptimizationConfig(
    name="my_optimization",
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2022-01-01",
    end_date="2022-12-31",
    method="grid",
    parameter_space=parameter_space,
)

# Run optimization
engine = OptimizationEngine(config)
summary = engine.run()
```

### 2. Evolutionary Search

```python
config = OptimizationConfig(
    name="evolutionary_opt",
    symbols=["AAPL", "MSFT"],
    start_date="2022-01-01",
    end_date="2022-12-31",
    method="evolutionary",
    generations=50,
    population_size=24,
    elite_size=4,
    mutation_rate=0.3,
    crossover_rate=0.7,
    parameter_space=parameter_space,
)

engine = OptimizationEngine(config)
summary = engine.run()
```

### 3. Command Line Interface

```bash
# Grid search (via consolidated CLI)
poetry run gpt-trader optimize-new \
  --name "grid_optimization" \
  --symbols "AAPL,MSFT,GOOGL" \
  --start-date "2022-01-01" \
  --end-date "2022-12-31" \
  --method grid

# Evolutionary search
poetry run gpt-trader optimize-new \
  --name "evolutionary_optimization" \
  --symbols "AAPL,MSFT" \
  --start-date "2022-01-01" \
  --end-date "2022-12-31" \
  --method evolutionary \
  --generations 50 \
  --population-size 24
```

## Configuration

### Strategy Configuration

Each strategy has a configuration that defines its parameters:

```python
from bot.optimization.config import get_trend_breakout_config

strategy_config = get_trend_breakout_config()
print(strategy_config.parameters)
# {
#     'donchian_lookback': ParameterDefinition(...),
#     'atr_k': ParameterDefinition(...),
#     'entry_confirm': ParameterDefinition(...),
#     ...
# }
```

### Parameter Space

Define the search space for optimization:

```python
parameter_space = ParameterSpace(
    strategy=strategy_config,
    # Grid search ranges
    grid_ranges={
        "donchian_lookback": [40, 55, 70, 85],
        "atr_k": [1.5, 2.0, 2.5, 3.0],
        "entry_confirm": [1, 2],
    },
    # Evolutionary bounds
    evolutionary_bounds={
        "donchian_lookback": {"min": 20, "max": 200},
        "atr_k": {"min": 0.5, "max": 5.0},
        "atr_period": {"min": 10, "max": 50},
    }
)
```

### Optimization Configuration

Main configuration for the optimization run:

```python
config = OptimizationConfig(
    # Basic settings
    name="my_optimization",
    description="Optimization run description",
    
    # Data settings
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2022-01-01",
    end_date="2022-12-31",
    
    # Walk-forward settings
    walk_forward=True,
    train_months=12,
    test_months=6,
    step_months=6,
    
    # Optimization settings
    method="both",  # grid, evolutionary, or both
    max_workers=4,  # Parallel processing
    
    # Grid search settings
    grid_search=True,
    grid_sample_size=100,  # Random sampling
    
    # Evolutionary settings
    evolutionary=True,
    generations=50,
    population_size=24,
    elite_size=4,
    mutation_rate=0.3,
    crossover_rate=0.7,
    
    # Early stopping
    early_stopping=True,
    patience=10,
    min_improvement=0.001,
    
    # Evaluation settings
    primary_metric="sharpe",
    min_trades=10,
    min_sharpe=0.5,
    max_drawdown=0.25,
    
    # Output settings
    output_dir="data/optimization/my_run",
    save_intermediate=True,
    create_plots=True,
    
    parameter_space=parameter_space,
)
```

## Analysis and Visualization

### Automatic Analysis

The framework automatically generates comprehensive analysis:

```python
from bot.optimization.analyzer import ResultAnalyzer

analyzer = ResultAnalyzer()
analysis = analyzer.analyze_results(results)

# Access different analysis components
print(analysis["summary_statistics"])
print(analysis["parameter_analysis"])
print(analysis["correlation_analysis"])
print(analysis["top_performers"])
print(analysis["robustness_analysis"])
print(analysis["sensitivity_analysis"])
```

### Visualization

Create visualizations of results:

```python
from bot.optimization.visualizer import ResultVisualizer

visualizer = ResultVisualizer()

# Individual plots
visualizer.plot_parameter_sensitivity(results, output_dir)
visualizer.plot_optimization_progress(results, output_dir)
visualizer.plot_correlation_matrix(results, output_dir)
visualizer.plot_parameter_distributions(results, output_dir)
visualizer.plot_performance_scatter(results, output_dir)

# Complete dashboard
visualizer.create_dashboard(results, output_dir)
```

## Output Structure

The optimization framework creates a well-organized output structure:

```
data/optimization/my_run/
├── config.json                 # Configuration used
├── summary.json               # Optimization summary
├── all_results.csv            # All evaluation results
├── grid_results.csv           # Grid search results (if applicable)
├── evolutionary_results.csv   # Evolutionary results (if applicable)
├── analysis/
│   ├── analysis.json          # Detailed analysis
│   ├── parameter_sensitivity.png
│   ├── optimization_progress.png
│   ├── correlation_matrix.png
│   ├── parameter_distributions.png
│   ├── performance_scatter.png
│   └── dashboard.html         # Interactive dashboard
└── intermediate/              # Intermediate results (if enabled)
    ├── grid_intermediate_20231201_143022.csv
    └── ...
```

## Advanced Features

### Walk-Forward Testing

Enable walk-forward testing for more robust optimization:

```python
config = OptimizationConfig(
    # ... other settings ...
    walk_forward=True,
    train_months=12,    # Training window
    test_months=6,      # Test window
    step_months=6,      # Step between windows
)
```

### Parallel Processing

Speed up optimization with parallel processing:

```python
config = OptimizationConfig(
    # ... other settings ...
    max_workers=4,  # Use 4 parallel workers
)
```

### Early Stopping

Configure early stopping to save computation time:

```python
config = OptimizationConfig(
    # ... other settings ...
    early_stopping=True,
    patience=10,           # Stop after 10 generations without improvement
    min_improvement=0.001, # Minimum improvement threshold
)
```

### Custom Metrics

Define custom evaluation metrics:

```python
config = OptimizationConfig(
    # ... other settings ...
    primary_metric="sharpe",  # Options: sharpe, cagr, sortino, calmar, max_drawdown
    secondary_metrics=["cagr", "max_drawdown"],
)
```

## Best Practices

### 1. Start Small
- Begin with a small parameter space and few symbols
- Use grid sampling for large parameter spaces
- Test with shorter time periods first

### 2. Use Walk-Forward Testing
- Always enable walk-forward testing for live trading
- Use appropriate train/test window sizes
- Validate results across multiple time periods

### 3. Monitor Progress
- Use intermediate result saving
- Check early stopping criteria
- Monitor computational resources

### 4. Analyze Results Thoroughly
- Review parameter sensitivity analysis
- Check for overfitting with robustness analysis
- Consider parameter correlations

### 5. Validate Findings
- Test best parameters on out-of-sample data
- Compare with baseline strategies
- Consider transaction costs and slippage

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce population size or use grid sampling
2. **Slow Performance**: Enable parallel processing or reduce parameter space
3. **Poor Results**: Check parameter bounds and evaluation criteria
4. **Overfitting**: Use walk-forward testing and robustness analysis

### Debugging

Enable debug logging:

```python
# Setup logging
from bot.logging import get_logger
logger = get_logger(__name__)
```

Or use the CLI:

```bash
python -m src.bot.optimization.cli --log-level DEBUG ...
```

## Examples

See `examples/optimization_example.py` for complete working examples.

## API Reference

### Core Classes

- `OptimizationConfig`: Main configuration class
- `ParameterSpace`: Parameter space definition
- `StrategyConfig`: Strategy-specific configuration
- `OptimizationEngine`: Main optimization orchestrator
- `GridOptimizer`: Grid search implementation
- `EvolutionaryOptimizer`: Evolutionary algorithm implementation
- `ResultAnalyzer`: Analysis and insights
- `ResultVisualizer`: Visualization tools

### Key Methods

- `OptimizationEngine.run()`: Run complete optimization
- `ResultAnalyzer.analyze_results()`: Analyze optimization results
- `ResultVisualizer.create_dashboard()`: Create comprehensive dashboard
- `OptimizationConfig.save()` / `load()`: Save/load configurations

For detailed API documentation, see the source code and docstrings.
