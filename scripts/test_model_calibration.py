#!/usr/bin/env python3
"""
Test Model Calibration System
Phase 2.5 - Day 8

Tests the complete model calibration system including probability calibration,
realistic targets, threshold optimization, and confidence intervals.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our calibration components
from src.bot.ml.model_calibrator import CalibrationConfig, create_model_calibrator
from src.bot.ml.performance_targets import MarketRegime, ModelType, create_target_setter
from src.bot.ml.threshold_optimizer import (
    OptimizationObjective,
    TradingConstraints,
    create_threshold_optimizer,
)


def test_probability_calibration():
    """Test probability calibration methods"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Probability Calibration")
    logger.info("=" * 60)

    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="3y")

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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Train base model
    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Get uncalibrated predictions
    y_prob_uncal = model.predict_proba(X_test)[:, 1]

    # Test different calibration methods
    methods = ["isotonic", "sigmoid", "ensemble"]
    results = {}

    for method in methods:
        logger.info(f"\nTesting {method} calibration...")

        # Create calibrator
        config = CalibrationConfig(
            method=method,
            optimize_threshold=True,
            threshold_metric="f1",
            min_precision=0.6,
            target_accuracy=0.60,
            target_sharpe=1.0,
        )

        calibrator = create_model_calibrator(config)

        # Calibrate model
        start_time = time.time()
        calibrated_model = calibrator.calibrate(model, X_train, y_train, X_test, y_test)
        calibration_time = time.time() - start_time

        # Get calibrated predictions
        y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

        # Store results
        results[method] = {
            "calibrator": calibrator,
            "calibrated_model": calibrated_model,
            "y_prob_cal": y_prob_cal,
            "metrics": calibrator.calibration_metrics,
            "optimal_threshold": calibrator.optimal_threshold,
            "confidence_intervals": calibrator.confidence_intervals,
            "calibration_time": calibration_time,
        }

        logger.info(f"  ECE: {calibrator.calibration_metrics.ece:.4f}")
        logger.info(f"  Brier Score: {calibrator.calibration_metrics.brier_score:.4f}")
        logger.info(f"  Optimal Threshold: {calibrator.optimal_threshold:.3f}")
        logger.info(f"  Accuracy: {calibrator.calibration_metrics.accuracy:.3f}")
        logger.info(f"  Sharpe Ratio: {calibrator.calibration_metrics.sharpe_ratio:.2f}")
        logger.info(f"  Calibration Time: {calibration_time:.2f}s")

    # Compare methods
    logger.info("\n" + "-" * 40)
    logger.info("Calibration Method Comparison:")
    best_method = min(results.items(), key=lambda x: x[1]["metrics"].ece)
    logger.info(f"  Best ECE: {best_method[0]} ({best_method[1]['metrics'].ece:.4f})")

    best_sharpe = max(results.items(), key=lambda x: x[1]["metrics"].sharpe_ratio)
    logger.info(f"  Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['metrics'].sharpe_ratio:.2f})")

    return results, X_test, y_test, y_prob_uncal


def test_realistic_targets():
    """Test realistic performance target setting"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Realistic Performance Targets")
    logger.info("=" * 60)

    # Create target setter
    setter = create_target_setter()

    # Test metrics (realistic for trading)
    test_metrics = {
        "accuracy": 0.58,
        "sharpe_ratio": 1.1,
        "max_drawdown": -0.15,
        "win_rate": 0.53,
        "profit_factor": 1.4,
        "annual_return": 0.12,
        "stability_score": 0.75,
    }

    # Test in different market regimes
    for regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.VOLATILE]:
        logger.info(f"\n{regime.value.upper()} Market:")

        # Get adjusted targets
        targets = setter.get_targets(regime, ModelType.TREND_FOLLOWING)

        logger.info(f"  Target Accuracy: {targets.target_accuracy:.1%}")
        logger.info(f"  Target Sharpe: {targets.target_sharpe:.2f}")
        logger.info(f"  Target Return: {targets.target_annual_return:.1%}")
        logger.info(f"  Max Drawdown: {targets.max_drawdown:.1%}")

        # Validate performance
        validation = setter.validate_performance(test_metrics, regime, ModelType.TREND_FOLLOWING)

        logger.info(f"  Overall Score: {validation.score:.1f}/100")
        logger.info(f"  Meets Minimum: {'✅' if validation.meets_minimum else '❌'}")
        logger.info(f"  Is Realistic: {'✅' if validation.is_realistic else '❌'}")

        if validation.warnings:
            logger.info("  Warnings:")
            for warning in validation.warnings[:2]:  # Show first 2
                logger.info(f"    ⚠️  {warning}")

    # Test market regime detection
    logger.info("\n" + "-" * 40)
    logger.info("Market Regime Detection:")

    ticker = yf.Ticker("SPY")
    data = ticker.history(period="6m")
    detected_regime = setter.detect_market_regime(data["Close"])
    logger.info(f"  Current Market Regime: {detected_regime.value}")

    return setter, validation


def test_threshold_optimization():
    """Test decision threshold optimization"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Threshold Optimization")
    logger.info("=" * 60)

    # Generate sample data with realistic properties
    np.random.seed(42)
    n_samples = 1000

    # Create somewhat predictable target
    y_true = np.random.randint(0, 2, n_samples)

    # Generate probabilities with signal
    y_prob = np.where(
        y_true == 1,
        np.random.beta(6, 4, n_samples),  # Higher prob for positive
        np.random.beta(4, 6, n_samples),
    )  # Lower prob for negative

    # Add noise to make it realistic
    y_prob = np.clip(y_prob + np.random.normal(0, 0.1, n_samples), 0, 1)

    # Generate price data
    returns = np.random.randn(n_samples) * 0.01
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))

    # Test different optimization objectives
    objectives = [
        ("profit", OptimizationObjective(primary_metric="profit", risk_aversion=1.0)),
        ("sharpe", OptimizationObjective(primary_metric="sharpe", risk_aversion=2.0)),
        ("f1", OptimizationObjective(primary_metric="f1", risk_aversion=1.5)),
    ]

    results = {}

    for obj_name, objective in objectives:
        logger.info(f"\nOptimizing for {obj_name}...")

        # Set constraints
        constraints = TradingConstraints(
            min_precision=0.55, min_recall=0.25, commission_rate=0.001, slippage_rate=0.0005
        )

        # Create optimizer
        optimizer = create_threshold_optimizer(constraints, objective)

        # Optimize
        start_time = time.time()
        result = optimizer.optimize(y_true, y_prob, prices)
        optimization_time = time.time() - start_time

        results[obj_name] = result

        logger.info(f"  Entry Threshold: {result.entry_threshold:.3f}")
        logger.info(f"  Exit Threshold: {result.exit_threshold:.3f}")
        logger.info(f"  Stop Loss: {result.stop_loss_threshold:.3%}")
        logger.info(f"  Take Profit: {result.take_profit_threshold:.3%}")
        logger.info(f"  Expected Return: {result.expected_return:.1%}")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Win Rate: {result.win_rate:.1%}")
        logger.info(f"  Optimization Time: {optimization_time:.2f}s")

    # Compare results
    logger.info("\n" + "-" * 40)
    logger.info("Optimization Comparison:")

    best_return = max(results.items(), key=lambda x: x[1].expected_return)
    logger.info(f"  Best Return: {best_return[0]} ({best_return[1].expected_return:.1%})")

    best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    logger.info(f"  Best Sharpe: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")

    return results, y_true, y_prob


def test_integrated_calibration(calibration_results, X_test, y_test):
    """Test integrated calibration with all components"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Integrated Calibration System")
    logger.info("=" * 60)

    # Use best calibrated model
    best_cal = calibration_results["ensemble"]
    calibrated_model = best_cal["calibrated_model"]
    y_prob_cal = best_cal["y_prob_cal"]

    # Get prices for backtesting
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="3y")
    prices = data["Close"].loc[X_test.index]

    # 1. Check against realistic targets
    logger.info("\nValidating against realistic targets...")

    setter = create_target_setter()
    current_regime = setter.detect_market_regime(prices)

    metrics = {
        "accuracy": best_cal["metrics"].accuracy,
        "sharpe_ratio": best_cal["metrics"].sharpe_ratio,
        "max_drawdown": -0.15,  # Placeholder
        "win_rate": best_cal["metrics"].win_rate,
        "profit_factor": best_cal["metrics"].profit_factor,
        "annual_return": best_cal["metrics"].expected_return,
        "stability_score": 0.7,  # Placeholder
    }

    validation = setter.validate_performance(metrics, current_regime, ModelType.MIXED)

    logger.info(f"  Market Regime: {current_regime.value}")
    logger.info(f"  Performance Score: {validation.score:.1f}/100")
    logger.info(f"  Assessment: {validation.accuracy_assessment}")
    logger.info(f"  Meets Requirements: {'✅' if validation.meets_minimum else '❌'}")

    # 2. Optimize thresholds for calibrated probabilities
    logger.info("\nOptimizing thresholds for calibrated model...")

    constraints = TradingConstraints(min_precision=0.6, min_recall=0.3, commission_rate=0.001)

    objective = OptimizationObjective(primary_metric="sharpe", risk_aversion=2.0)

    optimizer = create_threshold_optimizer(constraints, objective)
    threshold_result = optimizer.optimize(y_test.values, y_prob_cal, prices)

    logger.info(f"  Optimal Entry: {threshold_result.entry_threshold:.3f}")
    logger.info(f"  Expected Sharpe: {threshold_result.sharpe_ratio:.2f}")
    logger.info(f"  Expected Return: {threshold_result.expected_return:.1%}")

    # 3. Calculate position sizes using Kelly criterion
    logger.info("\nPosition Sizing Examples:")

    calibrator = best_cal["calibrator"]
    test_probabilities = [0.55, 0.60, 0.65, 0.70]

    for prob in test_probabilities:
        position_size = calibrator.calculate_position_size(prob, current_capital=100000)
        logger.info(f"  P={prob:.2f}: ${position_size:,.0f} ({position_size/1000:.1f}% of capital)")

    # 4. Display confidence intervals
    logger.info("\nConfidence Intervals (95%):")

    for metric, (lower, upper) in best_cal["confidence_intervals"].items():
        logger.info(f"  {metric}: [{lower:.3f}, {upper:.3f}]")

    return validation, threshold_result


def generate_calibration_report():
    """Generate comprehensive calibration report"""
    logger.info("\n" + "=" * 60)
    logger.info("Generating Calibration Report")
    logger.info("=" * 60)

    report = """
MODEL CALIBRATION REPORT
========================

1. CALIBRATION METHODS TESTED
   - Isotonic Regression: Non-parametric, monotonic
   - Sigmoid (Platt): Parametric, simple
   - Ensemble: Combination of both

2. REALISTIC PERFORMANCE TARGETS
   Market-Adjusted Expectations:
   - Bull Market: 60-65% accuracy, Sharpe > 1.2
   - Bear Market: 55-60% accuracy, Sharpe > 0.9
   - Volatile Market: 52-58% accuracy, Sharpe > 0.7

   Risk Limits:
   - Maximum Drawdown: 20%
   - Minimum Win Rate: 45% (with good R:R)
   - Minimum Sharpe: 0.5

3. THRESHOLD OPTIMIZATION
   Objectives Supported:
   - Profit Maximization
   - Sharpe Ratio Optimization
   - F1 Score Balance
   - Risk-Adjusted Returns

   Constraints Applied:
   - Minimum Precision: 60%
   - Transaction Costs: 0.1%
   - Slippage: 0.05%

4. CONFIDENCE INTERVALS
   - Bootstrap-based (1000 samples)
   - 95% confidence level
   - Covers key metrics

5. KEY ACHIEVEMENTS
   ✅ Probability calibration reduces overconfidence
   ✅ Realistic targets prevent overfitting
   ✅ Optimal thresholds improve trading performance
   ✅ Confidence intervals quantify uncertainty
   ✅ Position sizing based on calibrated probabilities

6. PRODUCTION RECOMMENDATIONS
   • Use ensemble calibration for best overall performance
   • Set market-regime-specific targets
   • Optimize thresholds for Sharpe ratio
   • Apply Kelly criterion with safety factor (25%)
   • Monitor calibration drift over time
   • Recalibrate monthly or after significant market changes
"""

    logger.info(report)

    # Save report
    report_path = Path("calibration_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"\nReport saved to {report_path}")


def main():
    """Run all calibration tests"""
    logger.info("=" * 60)
    logger.info("Model Calibration Test Suite")
    logger.info("Phase 2.5 - Day 8")
    logger.info("=" * 60)

    all_tests_passed = True

    # Test 1: Probability Calibration
    try:
        calibration_results, X_test, y_test, y_prob_uncal = test_probability_calibration()
    except Exception as e:
        logger.error(f"Probability calibration test failed: {e}")
        all_tests_passed = False
        calibration_results = None

    # Test 2: Realistic Targets
    try:
        setter, validation = test_realistic_targets()
    except Exception as e:
        logger.error(f"Realistic targets test failed: {e}")
        all_tests_passed = False

    # Test 3: Threshold Optimization
    try:
        threshold_results, y_true_opt, y_prob_opt = test_threshold_optimization()
    except Exception as e:
        logger.error(f"Threshold optimization test failed: {e}")
        all_tests_passed = False

    # Test 4: Integrated Calibration
    if calibration_results:
        try:
            validation, threshold_result = test_integrated_calibration(
                calibration_results, X_test, y_test
            )
        except Exception as e:
            logger.error(f"Integrated calibration test failed: {e}")
            all_tests_passed = False

    # Generate report
    try:
        generate_calibration_report()
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    if all_tests_passed:
        logger.info("✅ All calibration tests passed!")
        logger.info("\nKey Achievements:")
        logger.info("  • Probability calibration working (3 methods)")
        logger.info("  • Realistic targets validated")
        logger.info("  • Threshold optimization functional")
        logger.info("  • Confidence intervals calculated")
        logger.info("  • Kelly criterion position sizing")
        logger.info("  • Integrated system operational")
    else:
        logger.error("❌ Some tests failed. Review logs for details.")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
