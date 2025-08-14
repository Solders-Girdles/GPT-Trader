# GPT-Trader ML System Documentation

## Overview

GPT-Trader's Phase 2.5 introduces a production-ready ML-powered autonomous portfolio management system with comprehensive feature engineering, model validation, and performance benchmarking capabilities.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [ML Pipeline Components](#ml-pipeline-components)
3. [Feature Engineering](#feature-engineering)
4. [Model Training & Validation](#model-training--validation)
5. [Performance Monitoring](#performance-monitoring)
6. [Deployment Guide](#deployment-guide)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## System Architecture

### High-Level Architecture

```
GPT-Trader ML System
├── Data Pipeline
│   ├── Real-time WebSocket feeds
│   ├── Historical data management
│   └── Data validation & cleaning
├── Feature Engineering
│   ├── 6 feature categories
│   ├── 50 optimized features
│   └── Feature selection & reduction
├── Model Pipeline
│   ├── Training & calibration
│   ├── Walk-forward validation
│   └── Threshold optimization
├── Trading Engine
│   ├── Signal generation
│   ├── Position sizing (Kelly)
│   └── Risk management
└── Monitoring
    ├── Performance tracking
    ├── Degradation detection
    └── Automated retraining
```

### Technology Stack

- **Database**: PostgreSQL with TimescaleDB
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Technical Indicators**: TA-Lib
- **Monitoring**: Custom performance tracking
- **Deployment**: Docker, Redis

## ML Pipeline Components

### 1. Feature Engineering (`src/bot/ml/feature_engineering_v2.py`)

Generates 50+ optimized features across 6 categories:

```python
from src.bot.ml import OptimizedFeatureEngineer, FeatureConfig

config = FeatureConfig(
    price_features=True,
    volume_features=True,
    technical_features=True,
    statistical_features=True,
    pattern_features=True,
    interaction_features=True,
    max_features=50
)

engineer = OptimizedFeatureEngineer(config)
features = engineer.generate_features(data)
```

### 2. Feature Selection (`src/bot/ml/feature_selector.py`)

Advanced feature selection with multiple methods:

```python
from src.bot.ml import AdvancedFeatureSelector, SelectionMethod

selector = AdvancedFeatureSelector()
selected_features, importance = selector.select_features(
    X, y,
    method=SelectionMethod.ENSEMBLE,
    n_features=50
)
```

### 3. Model Validation (`src/bot/ml/walk_forward_validator.py`)

Walk-forward validation with backtesting:

```python
from src.bot.ml import WalkForwardValidator, WalkForwardConfig

config = WalkForwardConfig(
    train_window=504,  # 2 years
    test_window=126,   # 6 months
    step_size=21,      # 1 month
    backtest_each_fold=True
)

validator = WalkForwardValidator(config)
results = validator.validate(model, X, y, prices)
```

### 4. Model Calibration (`src/bot/ml/model_calibrator.py`)

Probability calibration and threshold optimization:

```python
from src.bot.ml import ModelCalibrator, CalibrationConfig

config = CalibrationConfig(
    method="isotonic",
    optimize_threshold=True,
    kelly_fraction=0.25
)

calibrator = ModelCalibrator(config)
calibrated_model = calibrator.calibrate(model, X_train, y_train, X_val, y_val)
```

## Feature Engineering

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Price | 8 | Returns, log returns, price ratios |
| Volume | 6 | Volume ratio, OBV, volume MA |
| Technical | 12 | RSI, MACD, Bollinger Bands |
| Statistical | 10 | Volatility, skewness, kurtosis |
| Pattern | 8 | Support/resistance, trend strength |
| Interaction | 6 | Price×volume, momentum×volatility |

### Feature Selection Process

1. **Variance Threshold**: Remove low-variance features
2. **Correlation Removal**: Remove highly correlated (>0.7)
3. **Multiple Methods**:
   - Mutual Information
   - Lasso regularization
   - Recursive Feature Elimination
   - Random Forest importance
   - XGBoost importance
4. **Ensemble Voting**: Combine methods for robustness

## Model Training & Validation

### Training Pipeline

```python
from src.bot.ml import IntegratedMLPipeline

pipeline = IntegratedMLPipeline(
    feature_config=FeatureConfig(max_features=50),
    selection_config=FeatureSelectionConfig(
        n_features_target=50,
        correlation_threshold=0.7
    ),
    validation_config=ValidationConfig(
        n_splits=5,
        test_size=252
    )
)

# Prepare features
features = pipeline.prepare_features(data, target, use_selection=True)

# Train and validate
performance = pipeline.train_and_validate(
    XGBClassifier, X_train, y_train, X_test, y_test
)
```

### Walk-Forward Validation

- **Purpose**: Avoid lookahead bias in time series
- **Method**: Expanding or rolling windows
- **Purging Gap**: 5 days between train/test
- **Backtesting**: On each fold for realistic metrics

### Performance Targets

| Metric | Minimum | Target | Maximum |
|--------|---------|--------|---------|
| Accuracy | 52% | 58% | 65% |
| Sharpe Ratio | 0.5 | 1.0 | 1.5 |
| Max Drawdown | - | 12% | 20% |
| Win Rate | 45% | 52% | 60% |

## Performance Monitoring

### Degradation Detection

```python
from src.bot.ml import ModelDegradationMonitor, RetrainingTrigger

triggers = RetrainingTrigger(
    accuracy_drop_threshold=0.05,
    min_accuracy_threshold=0.55,
    max_days_without_retrain=30
)

monitor = ModelDegradationMonitor(triggers=triggers)
monitor.update_performance(model_id, y_true, y_pred, confidence)
```

### Key Metrics Tracked

- **Accuracy Metrics**: Accuracy, precision, recall, F1
- **Trading Metrics**: Sharpe, returns, drawdown, win rate
- **Efficiency Metrics**: Training time, inference speed, memory
- **Stability Metrics**: Performance variance, feature stability

## Deployment Guide

### Prerequisites

```bash
# System requirements
- Python 3.8+
- PostgreSQL 15+
- Redis 7+
- 8GB RAM minimum
- 4 CPU cores recommended
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/GPT-Trader.git
cd GPT-Trader

# Install dependencies
pip install -r requirements.txt

# Set up database
docker-compose -f deploy/postgres/docker-compose.yml up -d

# Run migrations
python scripts/migrate_to_postgres.py
```

### Configuration

```python
# config/ml_config.yaml
ml_pipeline:
  features:
    max_features: 50
    correlation_threshold: 0.7

  model:
    type: XGBClassifier
    params:
      n_estimators: 100
      max_depth: 5
      learning_rate: 0.1

  validation:
    walk_forward: true
    train_window: 504
    test_window: 126

  monitoring:
    degradation_check: true
    retraining_threshold: 0.05
```

### Running the System

```python
# main.py
from src.bot.ml import IntegratedMLPipeline
from src.bot.dataflow import RealtimeFeed
from src.bot.trading import TradingEngine

# Initialize components
pipeline = IntegratedMLPipeline.from_config('config/ml_config.yaml')
feed = RealtimeFeed(['AAPL', 'GOOGL', 'MSFT'])
engine = TradingEngine(pipeline, feed)

# Start trading
engine.start()
```

## API Reference

### Core Classes

#### `IntegratedMLPipeline`
Main pipeline orchestrator for ML workflow.

**Methods:**
- `prepare_features(data, target)`: Generate and select features
- `train_and_validate(model_class, X_train, y_train, X_test, y_test)`: Train and validate model
- `predict(X)`: Generate predictions
- `check_model_health()`: Check for degradation

#### `WalkForwardValidator`
Time series cross-validation with backtesting.

**Methods:**
- `validate(model, X, y, prices)`: Run walk-forward validation
- `plot_results(results)`: Visualize validation results

#### `ModelCalibrator`
Probability calibration and threshold optimization.

**Methods:**
- `calibrate(model, X_train, y_train, X_val, y_val)`: Calibrate model
- `optimize_threshold(y_true, y_prob)`: Find optimal decision threshold
- `calculate_position_size(probability)`: Kelly criterion sizing

## Troubleshooting

### Common Issues

#### 1. Low Model Accuracy
```
Symptom: Accuracy < 52%
Solutions:
- Check data quality and completeness
- Increase training data size (minimum 2 years)
- Review feature engineering
- Try different model types
```

#### 2. High Memory Usage
```
Symptom: Memory > 4GB during training
Solutions:
- Reduce feature count
- Use sparse matrices
- Enable incremental learning
- Reduce model complexity
```

#### 3. Slow Training
```
Symptom: Training > 10 minutes
Solutions:
- Enable parallel processing (n_jobs=-1)
- Reduce hyperparameter search space
- Use early stopping
- Consider GPU acceleration
```

#### 4. Model Degradation
```
Symptom: Performance drop > 5%
Solutions:
- Check for data distribution shift
- Retrain with recent data
- Review feature importance changes
- Adjust retraining frequency
```

### Performance Optimization

1. **Feature Optimization**
   - Keep features under 100
   - Remove highly correlated features
   - Use feature importance for selection

2. **Model Optimization**
   - Use XGBoost for best performance
   - Enable early stopping
   - Optimize hyperparameters carefully

3. **Infrastructure Optimization**
   - Use connection pooling for database
   - Implement caching for predictions
   - Use batch processing where possible

## Best Practices

### 1. Data Management
- Always validate incoming data
- Handle missing values appropriately
- Maintain data versioning
- Use proper time series splits

### 2. Model Development
- Start with baselines for comparison
- Use walk-forward validation
- Calibrate probabilities
- Optimize thresholds for trading

### 3. Production Deployment
- Monitor performance continuously
- Set up automated retraining
- Implement fallback strategies
- Log all predictions and trades

### 4. Risk Management
- Use position sizing (Kelly criterion)
- Set maximum drawdown limits
- Implement circuit breakers
- Diversify across models/assets

## Support

For issues or questions:
- GitHub Issues: [Report bugs](https://github.com/yourusername/GPT-Trader/issues)
- Documentation: [Full docs](https://docs.gpt-trader.com)
- Community: [Discord server](https://discord.gg/gpt-trader)
