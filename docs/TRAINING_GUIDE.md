# GPT-Trader Training Guide

## Module 1: System Overview

### Learning Objectives
- Understand the GPT-Trader architecture
- Learn the ML pipeline components
- Master the trading workflow
- Configure and deploy the system

### Prerequisites
- Python programming (intermediate)
- Basic ML concepts
- Financial markets understanding
- Command line proficiency

## Module 2: Getting Started

### Environment Setup

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/GPT-Trader.git
cd GPT-Trader
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

4. **Start Database**
```bash
docker-compose -f deploy/postgres/docker-compose.yml up -d
```

5. **Configure Environment**
```bash
cp .env.template .env
# Edit .env with your settings
```

### First Backtest

Run your first backtest to verify setup:

```python
from src.bot.cli import main

# Simple moving average strategy
main([
    'backtest',
    '--symbol', 'AAPL',
    '--start', '2024-01-01',
    '--end', '2024-03-31',
    '--strategy', 'demo_ma'
])
```

Expected output:
- Backtest results summary
- Sharpe ratio > 0
- Trade count > 0
- No errors

## Module 3: Feature Engineering

### Understanding Features

Our system uses 50 optimized features across 6 categories:

```python
from src.bot.ml import OptimizedFeatureEngineer, FeatureConfig

# Configure features
config = FeatureConfig(
    price_features=True,      # Returns, log returns
    volume_features=True,      # Volume ratios, OBV
    technical_features=True,   # RSI, MACD, Bollinger
    statistical_features=True, # Volatility, skewness
    pattern_features=True,     # Support/resistance
    interaction_features=True  # Cross-features
)

# Generate features
engineer = OptimizedFeatureEngineer(config)
features = engineer.generate_features(ohlcv_data)
```

### Lab Exercise 1: Custom Features

Add a custom feature to the pipeline:

```python
def add_custom_feature(df):
    """Add your custom feature here"""
    # Example: Price acceleration
    df['price_acceleration'] = df['returns'].diff()
    return df

# Test your feature
data = load_sample_data()
data_with_feature = add_custom_feature(data)
print(data_with_feature['price_acceleration'].describe())
```

### Feature Selection

Learn to select the best features:

```python
from src.bot.ml import AdvancedFeatureSelector

selector = AdvancedFeatureSelector()

# Try different methods
methods = ['mutual_information', 'lasso', 'random_forest']
for method in methods:
    selected, importance = selector.select_features(
        X, y, method=method, n_features=30
    )
    print(f"{method}: {len(selected)} features selected")
```

## Module 4: Model Training

### Training Pipeline

Complete training workflow:

```python
from src.bot.ml import IntegratedMLPipeline
from xgboost import XGBClassifier

# Initialize pipeline
pipeline = IntegratedMLPipeline()

# Prepare data
X, y = prepare_training_data()

# Split data (time-aware)
split_date = '2023-01-01'
X_train = X[X.index < split_date]
y_train = y[y.index < split_date]
X_test = X[X.index >= split_date]
y_test = y[y.index >= split_date]

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

performance = pipeline.train_and_validate(
    XGBClassifier,
    X_train, y_train,
    X_test, y_test,
    model_params=model.get_params()
)

print(f"Accuracy: {performance.accuracy:.3f}")
print(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
```

### Lab Exercise 2: Model Comparison

Compare different models:

```python
models = {
    'XGBoost': XGBClassifier(n_estimators=100),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'Baseline': get_baseline_models()['ensemble']
}

results = {}
for name, model in models.items():
    # Train and evaluate
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

# Plot comparison
plot_model_comparison(results)
```

## Module 5: Walk-Forward Validation

### Understanding Walk-Forward

Walk-forward validation prevents lookahead bias:

```python
from src.bot.ml import WalkForwardValidator, WalkForwardConfig

config = WalkForwardConfig(
    train_window=504,     # 2 years training
    test_window=126,      # 6 months testing
    step_size=21,         # 1 month step
    purge_gap=5,          # 5 days gap
    expanding_window=True # Use all historical data
)

validator = WalkForwardValidator(config)
results = validator.validate(model, X, y, prices)

print(f"Mean Accuracy: {results.mean_test_accuracy:.3f}")
print(f"Stability Score: {results.stability_score:.3f}")
```

### Interpreting Results

Key metrics to evaluate:
- **Accuracy Stability**: Std < 0.05
- **Sharpe Consistency**: All folds > 0.5
- **No Degradation**: < 20% of folds degraded
- **Feature Stability**: Important features consistent

## Module 6: Model Calibration

### Probability Calibration

Calibrate model probabilities for better risk management:

```python
from src.bot.ml import ModelCalibrator, CalibrationConfig

config = CalibrationConfig(
    method="isotonic",        # or "sigmoid", "ensemble"
    optimize_threshold=True,   # Find best threshold
    kelly_fraction=0.25        # Conservative Kelly
)

calibrator = ModelCalibrator(config)
calibrated_model = calibrator.calibrate(
    model, X_train, y_train, X_val, y_val
)

# Check calibration quality
print(f"ECE: {calibrator.calibration_metrics.ece:.4f}")  # < 0.05 is good
print(f"Optimal Threshold: {calibrator.optimal_threshold:.3f}")
```

### Position Sizing

Use calibrated probabilities for position sizing:

```python
# Example predictions with probabilities
probabilities = [0.55, 0.65, 0.75, 0.85]
capital = 100000

for prob in probabilities:
    position = calibrator.calculate_position_size(prob, capital)
    print(f"P={prob:.2f}: ${position:,.0f} ({position/capital:.1%})")
```

## Module 7: Production Deployment

### Deployment Checklist

- [ ] Database configured and running
- [ ] Environment variables set
- [ ] Models trained and validated
- [ ] Monitoring configured
- [ ] Risk limits set
- [ ] Fallback strategies ready
- [ ] Logging enabled
- [ ] Alerts configured

### Running in Production

```python
# production.py
import logging
from src.bot.ml import IntegratedMLPipeline
from src.bot.trading import LiveTradingEngine
from src.bot.monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

# Initialize components
pipeline = IntegratedMLPipeline.from_config('config/production.yaml')
monitor = PerformanceMonitor()
engine = LiveTradingEngine(pipeline, monitor)

# Set risk limits
engine.set_max_drawdown(0.20)
engine.set_max_positions(10)
engine.set_position_size_limit(0.10)

# Start trading
try:
    engine.start()
except Exception as e:
    logging.error(f"Trading error: {e}")
    engine.stop()
```

## Module 8: Monitoring & Maintenance

### Performance Monitoring

Track key metrics in production:

```python
from src.bot.ml import ModelDegradationMonitor

monitor = ModelDegradationMonitor()

# Check model health
health = monitor.get_degradation_report('production_model')
print(f"Status: {health['status']}")
print(f"Current Accuracy: {health['performance_summary']['current']:.3f}")
print(f"Degradation Type: {health.get('degradation_type', 'None')}")

# Set up alerts
if health['status'] == 'degraded':
    send_alert("Model degradation detected!")
    trigger_retraining()
```

### Maintenance Tasks

Daily:
- Check model performance metrics
- Review trade logs
- Monitor system resources

Weekly:
- Analyze prediction accuracy
- Review risk metrics
- Update feature importance

Monthly:
- Retrain models with new data
- Recalibrate probabilities
- Update baselines
- Performance report

## Module 9: Troubleshooting

### Common Issues

#### Issue 1: Low Accuracy
```python
# Diagnostic steps
from src.bot.ml import diagnose_model

diagnosis = diagnose_model(model, X_test, y_test)
print(diagnosis.report)

# Common solutions:
# 1. Increase training data
# 2. Add more features
# 3. Adjust hyperparameters
# 4. Check data quality
```

#### Issue 2: Overfitting
```python
# Check for overfitting
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
overfit_ratio = train_score / test_score

if overfit_ratio > 1.2:
    print("Overfitting detected!")
    # Solutions:
    # 1. Reduce model complexity
    # 2. Add regularization
    # 3. Increase training data
    # 4. Use cross-validation
```

#### Issue 3: Slow Performance
```python
from src.bot.ml import EfficiencyAnalyzer

analyzer = EfficiencyAnalyzer()
metrics = analyzer.analyze_model_efficiency(
    model, X_train, y_train, X_test
)

print(f"Training Time: {metrics.training_time:.2f}s")
print(f"Bottlenecks: {metrics.bottlenecks}")
print(f"Suggestions: {metrics.optimization_suggestions}")
```

## Module 10: Advanced Topics

### Custom Strategies

Create your own trading strategy:

```python
from src.bot.strategy.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, model, threshold=0.6):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def generate_signals(self, data):
        # Generate features
        features = self.prepare_features(data)

        # Get predictions
        probabilities = self.model.predict_proba(features)[:, 1]

        # Generate signals
        signals = np.where(probabilities > self.threshold, 1, 0)

        return signals

    def calculate_position_size(self, signal, probability, capital):
        # Kelly criterion with safety
        kelly_fraction = 0.25
        edge = probability - 0.5
        position = capital * kelly_fraction * edge * 2
        return min(position, capital * 0.1)  # Max 10%
```

### Ensemble Models

Combine multiple models:

```python
from sklearn.ensemble import VotingClassifier

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    voting='soft'  # Use probabilities
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Evaluate
ensemble_pred = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"Ensemble Accuracy: {ensemble_acc:.3f}")
```

## Practical Exercises

### Exercise 1: Complete Pipeline
Build an end-to-end pipeline from data to predictions.

### Exercise 2: Strategy Optimization
Optimize a trading strategy using walk-forward validation.

### Exercise 3: Risk Management
Implement position sizing and stop-loss logic.

### Exercise 4: Performance Analysis
Create a comprehensive performance report.

## Assessment Questions

1. What is the purpose of the purging gap in walk-forward validation?
2. How does probability calibration improve trading performance?
3. What are the key differences between expanding and rolling windows?
4. How do you detect model degradation in production?
5. What is the Kelly criterion and how is it used for position sizing?

## Resources

- **Documentation**: `/docs/ML_SYSTEM_DOCUMENTATION.md`
- **API Reference**: `/docs/API_REFERENCE.md`
- **Code Examples**: `/examples/`
- **Test Data**: `/tests/fixtures/`
- **Support**: GitHub Issues

## Certification Path

1. **Basic**: Complete Modules 1-5
2. **Intermediate**: Complete Modules 6-8
3. **Advanced**: Complete Modules 9-10 + Practical Exercises
4. **Expert**: Contribute to codebase + Deploy to production

---

*This training guide provides comprehensive coverage of the GPT-Trader system. Practice with real data and always test in paper trading before live deployment.*
