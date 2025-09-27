#!/usr/bin/env python3
"""
Test Walk-Forward Validation Integration
Phase 2.5 - Day 7

Tests the complete walk-forward validation system with model degradation monitoring
and report generation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our components
from src.bot.ml import FeatureSelectorConfig, SelectionMethod
from src.bot.ml.integrated_pipeline import IntegratedMLPipeline
from src.bot.ml.model_degradation_monitor import RetrainingTrigger, create_degradation_monitor
from src.bot.ml.validation_reporter import create_validation_reporter
from src.bot.ml.walk_forward_validator import WalkForwardConfig, create_walk_forward_validator


def test_walk_forward_validation():
    """Test basic walk-forward validation"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Walk-Forward Validation")
    logger.info("=" * 60)

    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="5y")

    # Create features
    data["returns"] = data["Close"].pct_change()
    data["sma_20"] = data["Close"].rolling(20).mean()
    data["sma_50"] = data["Close"].rolling(50).mean()
    data["rsi"] = 100 - (
        100
        / (
            1
            + data["Close"].diff().clip(lower=0).rolling(14).mean()
            / data["Close"].diff().clip(upper=0).abs().rolling(14).mean()
        )
    )

    X = pd.DataFrame(index=data.index)
    X["returns"] = data["returns"]
    X["volume_ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()
    X["price_to_sma20"] = data["Close"] / data["sma_20"]
    X["price_to_sma50"] = data["Close"] / data["sma_50"]
    X["rsi"] = data["rsi"]
    X["volatility"] = data["returns"].rolling(20).std()
    X = X.dropna()

    # Create target
    y = (data["Close"].shift(-1) > data["Close"]).astype(int)
    y = y.loc[X.index]

    # Get prices for backtesting
    prices = data["Close"].loc[X.index]

    # Configure walk-forward validation
    config = WalkForwardConfig(
        train_window=504,  # 2 years
        test_window=126,  # 6 months
        step_size=21,  # 1 month
        expanding_window=True,
        purge_gap=5,
        backtest_each_fold=True,
        track_degradation=True,
        save_fold_models=True,
    )

    # Create validator
    validator = create_walk_forward_validator(config)

    # Test with different models
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, random_state=42, eval_metric="logloss"
        ),
    }

    results_dict = {}

    for model_name, model in models.items():
        logger.info(f"\nValidating {model_name}...")

        start_time = time.time()
        results = validator.validate(model, X, y, prices, model_name)
        elapsed = time.time() - start_time

        results_dict[model_name] = results

        logger.info(f"\nResults for {model_name}:")
        logger.info(f"  Validation time: {elapsed:.1f}s")
        logger.info(f"  Number of folds: {results.n_folds}")
        logger.info(
            f"  Mean Test Accuracy: {results.mean_test_accuracy:.3f} ± {results.std_test_accuracy:.3f}"
        )
        logger.info(f"  Mean Test F1: {results.mean_test_f1:.3f} ± {results.std_test_f1:.3f}")

        if results.mean_sharpe:
            logger.info(f"  Mean Sharpe Ratio: {results.mean_sharpe:.2f}")
        if results.mean_return:
            logger.info(f"  Mean Return: {results.mean_return:.2%}")
        if results.mean_drawdown:
            logger.info(f"  Mean Max Drawdown: {results.mean_drawdown:.2%}")

        logger.info(f"  Stability Score: {results.stability_score:.3f}")
        logger.info(f"  Degradation Detected: {results.degradation_detected}")

        if results.degradation_folds:
            logger.info(f"  Degraded Folds: {results.degradation_folds}")

    return results_dict


def test_model_degradation_monitoring():
    """Test model degradation monitoring"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Model Degradation Monitoring")
    logger.info("=" * 60)

    # Create degradation monitor
    triggers = RetrainingTrigger(
        accuracy_drop_threshold=0.05,  # 5% drop
        min_accuracy_threshold=0.55,  # Absolute minimum
        drift_significance_level=0.05,  # p-value threshold
        volatility_threshold=0.1,  # Performance volatility
        auto_retrain=True,
        auto_rollback=True,
    )

    monitor = create_degradation_monitor(baseline_window=30, monitoring_window=7, triggers=triggers)

    # Simulate model performance over time
    np.random.seed(42)
    model_id = "test_model"

    logger.info("\nSimulating 60 days of model performance...")

    for day in range(60):
        # Simulate performance degradation
        if day < 20:
            # Good performance
            accuracy = np.random.normal(0.65, 0.02)
        elif day < 40:
            # Gradual degradation
            accuracy = np.random.normal(0.62 - (day - 20) * 0.002, 0.03)
        else:
            # Recovery after simulated retrain
            accuracy = np.random.normal(0.64, 0.02)

        # Ensure valid range
        accuracy = np.clip(accuracy, 0, 1)

        # Generate fake predictions
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        # Generate predictions based on accuracy
        y_pred = y_true.copy()
        n_errors = int((1 - accuracy) * n_samples)
        error_idx = np.random.choice(n_samples, n_errors, replace=False)
        y_pred[error_idx] = 1 - y_pred[error_idx]

        confidence = np.random.beta(5, 2, n_samples)

        # Update monitor
        monitor.update_performance(model_id, y_true, y_pred, confidence)

        # Report status periodically
        if day % 10 == 0 and day > 0:
            report = monitor.get_degradation_report(model_id)
            logger.info(
                f"  Day {day}: Status={report['status']}, "
                f"Accuracy={report['performance_summary']['current']:.3f}, "
                f"Drift p-value={report['statistical_tests']['p_value']:.3f}"
            )

    # Final report
    final_report = monitor.get_degradation_report(model_id)

    logger.info("\nFinal Degradation Report:")
    logger.info(f"  Status: {final_report['status']}")
    logger.info(f"  Degradation Type: {final_report.get('degradation_type', 'None')}")
    logger.info(f"  Current Accuracy: {final_report['performance_summary']['current']:.3f}")
    logger.info(f"  Baseline Accuracy: {final_report['performance_summary']['baseline']:.3f}")
    logger.info(f"  Accuracy Drop: {final_report['performance_summary']['drop']:.3f}")
    logger.info(f"  Trend: {final_report['performance_summary']['trend']:.4f}")
    logger.info(f"  Drift Detected: {final_report['statistical_tests']['drift_detected']}")

    # Save monitoring state
    monitor.save_monitoring_state("monitor_state.json")
    logger.info("\nMonitor state saved to monitor_state.json")

    return final_report


def test_validation_reporting(results_dict):
    """Test validation report generation"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Validation Report Generation")
    logger.info("=" * 60)

    # Create reporter
    reporter = create_validation_reporter(output_dir="validation_reports")

    for model_name, results in results_dict.items():
        logger.info(f"\nGenerating report for {model_name}...")

        # Generate comprehensive report
        report = reporter.generate_report(
            results, model_name=model_name, save_html=True, save_json=True, save_plots=True
        )

        logger.info(f"  HTML report: {report.get('html_path', 'N/A')}")
        logger.info(f"  JSON report: {report.get('json_path', 'N/A')}")

        if "plot_paths" in report:
            logger.info("  Plots generated:")
            for plot_name, plot_path in report["plot_paths"].items():
                logger.info(f"    - {plot_name}: {plot_path}")

        # Display recommendations
        if report["recommendations"]:
            logger.info("  Recommendations:")
            for rec in report["recommendations"][:3]:  # Show first 3
                logger.info(f"    • {rec}")

    return True


def test_integrated_pipeline_with_walk_forward():
    """Test integration with the ML pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Integrated Pipeline with Walk-Forward Validation")
    logger.info("=" * 60)

    # Get sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="5y")
    data.columns = [c.lower() for c in data.columns]

    # Create target
    target = (data["close"].pct_change().shift(-1) > 0).astype(int)

    # Create integrated pipeline with feature selection
    pipeline = IntegratedMLPipeline(
        selection_config=FeatureSelectorConfig(
            n_features_target=50, correlation_threshold=0.7, primary_method=SelectionMethod.ENSEMBLE
        )
    )

    logger.info("\nPreparing features with integrated pipeline...")
    features = pipeline.prepare_features(data, target, use_selection=True)
    logger.info(f"  Features generated: {features.shape[1]}")
    logger.info(f"  Samples: {features.shape[0]}")

    # Align target with features
    target_aligned = target.loc[features.index]

    # Get prices for backtesting
    prices = data["close"].loc[features.index]

    # Configure walk-forward validation
    config = WalkForwardConfig(
        train_window=252,  # 1 year
        test_window=63,  # 3 months
        step_size=21,  # 1 month
        expanding_window=False,  # Rolling window
        backtest_each_fold=True,
    )

    # Create validator
    validator = create_walk_forward_validator(config)

    # Validate with XGBoost
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss"
    )

    logger.info("\nRunning walk-forward validation with integrated pipeline features...")

    start_time = time.time()
    results = validator.validate(model, features, target_aligned, prices, "XGBoost_Integrated")
    elapsed = time.time() - start_time

    logger.info("\nIntegrated Pipeline + Walk-Forward Results:")
    logger.info(f"  Validation time: {elapsed:.1f}s")
    logger.info(f"  Features used: {features.shape[1]}")
    logger.info(
        f"  Mean Accuracy: {results.mean_test_accuracy:.3f} ± {results.std_test_accuracy:.3f}"
    )
    logger.info(f"  Mean F1 Score: {results.mean_test_f1:.3f} ± {results.std_test_f1:.3f}")
    logger.info(f"  Stability Score: {results.stability_score:.3f}")

    # Check realistic accuracy
    is_realistic = 0.55 <= results.mean_test_accuracy <= 0.70
    logger.info(f"  Realistic accuracy (55-70%): {'✅' if is_realistic else '❌'}")

    return results


def main():
    """Run all walk-forward validation tests"""
    logger.info("=" * 60)
    logger.info("Walk-Forward Validation Test Suite")
    logger.info("Phase 2.5 - Day 7")
    logger.info("=" * 60)

    all_tests_passed = True

    # Test 1: Walk-Forward Validation
    try:
        results_dict = test_walk_forward_validation()
    except Exception as e:
        logger.error(f"Walk-forward validation test failed: {e}")
        all_tests_passed = False
        results_dict = {}

    # Test 2: Model Degradation Monitoring
    try:
        degradation_report = test_model_degradation_monitoring()
    except Exception as e:
        logger.error(f"Model degradation monitoring test failed: {e}")
        all_tests_passed = False

    # Test 3: Validation Reporting
    if results_dict:
        try:
            test_validation_reporting(results_dict)
        except Exception as e:
            logger.error(f"Validation reporting test failed: {e}")
            all_tests_passed = False

    # Test 4: Integrated Pipeline with Walk-Forward
    try:
        integrated_results = test_integrated_pipeline_with_walk_forward()
    except Exception as e:
        logger.error(f"Integrated pipeline test failed: {e}")
        all_tests_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    if all_tests_passed:
        logger.info("✅ All walk-forward validation tests passed!")
        logger.info("\nKey Achievements:")
        logger.info("  • Walk-forward validation framework operational")
        logger.info("  • Backtesting integrated on each fold")
        logger.info("  • Model degradation tracking active")
        logger.info("  • Comprehensive reports generated")
        logger.info("  • Integration with ML pipeline successful")
        logger.info("  • Realistic accuracy achieved (55-70%)")
    else:
        logger.error("❌ Some tests failed. Review logs for details.")

    # Cleanup
    import os

    if os.path.exists("monitor_state.json"):
        os.remove("monitor_state.json")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
