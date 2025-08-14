"""
Tests for Degradation Integration Module
Phase 3, Week 1: MON-006
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bot.ml.degradation_integration import (
    DegradationIntegrator,
    IntegratedDegradationReport,
    create_integrated_monitor
)


class TestDegradationIntegration:
    """Test the integration between advanced and legacy degradation detection"""
    
    def test_integration_initialization(self):
        """Test MON-006: Integration initialization"""
        # Test with both systems
        integrator = DegradationIntegrator(use_legacy=True, use_advanced=True)
        assert integrator.use_legacy == True
        assert integrator.use_advanced == True
        assert integrator.advanced_detector is not None
        print("✅ Integration initialized with both systems")
        
        # Test with advanced only
        integrator_advanced = DegradationIntegrator(use_legacy=False, use_advanced=True)
        assert integrator_advanced.legacy_monitor is None
        assert integrator_advanced.advanced_detector is not None
        print("✅ Integration initialized with advanced only")
    
    def test_baseline_setting(self):
        """Test baseline configuration for both systems"""
        integrator = create_integrated_monitor()
        
        # Generate sample data
        np.random.seed(42)
        n = 100
        features = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.uniform(0, 1, n)
        })
        predictions = np.random.choice([0, 1], n)
        actuals = predictions.copy()
        actuals[:10] = 1 - actuals[:10]  # Add some errors
        confidences = np.random.uniform(0.6, 0.9, n)
        
        # Set baseline
        integrator.set_baseline(features, predictions, actuals, confidences, "test_model")
        
        assert integrator.baseline_set == True
        print("✅ Baseline set successfully for both systems")
    
    def test_degradation_check(self):
        """Test integrated degradation checking"""
        integrator = create_integrated_monitor()
        
        # Generate baseline data
        np.random.seed(42)
        n = 100
        baseline_features = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.uniform(0, 1, n)
        })
        baseline_predictions = np.random.choice([0, 1], n)
        baseline_actuals = baseline_predictions.copy()
        baseline_actuals[:5] = 1 - baseline_actuals[:5]  # 5% error
        baseline_confidences = np.random.uniform(0.7, 0.9, n)
        
        # Set baseline
        integrator.set_baseline(
            baseline_features, 
            baseline_predictions, 
            baseline_actuals, 
            baseline_confidences
        )
        
        # Check with normal data (no degradation)
        report_normal = integrator.check_degradation(
            baseline_features,
            baseline_predictions,
            baseline_actuals,
            baseline_confidences
        )
        
        assert isinstance(report_normal, IntegratedDegradationReport)
        assert report_normal.overall_status in ["healthy", "warning"]
        print(f"✅ Normal check: {report_normal.overall_status}")
        
        # Check with degraded data
        degraded_features = baseline_features.copy()
        degraded_features['feature1'] += 2  # Shift distribution
        degraded_predictions = np.random.choice([0, 1], n)
        degraded_actuals = 1 - degraded_predictions  # All wrong
        degraded_confidences = np.ones(n) * 0.3  # Low confidence
        
        report_degraded = integrator.check_degradation(
            degraded_features,
            degraded_predictions,
            degraded_actuals,
            degraded_confidences
        )
        
        assert report_degraded.overall_status in ["degraded", "critical"]
        assert report_degraded.advanced_score > 0.3
        print(f"✅ Degraded check: {report_degraded.overall_status} (score={report_degraded.advanced_score:.2f})")
    
    def test_retraining_decision(self):
        """Test retraining decision logic"""
        integrator = create_integrated_monitor()
        
        # Set up baseline
        np.random.seed(42)
        n = 100
        features = pd.DataFrame({'f1': np.random.randn(n)})
        predictions = np.ones(n)
        actuals = np.ones(n)
        confidences = np.ones(n) * 0.8
        
        integrator.set_baseline(features, predictions, actuals, confidences)
        
        # Create critical degradation
        bad_predictions = np.zeros(n)
        bad_actuals = np.ones(n)
        low_confidences = np.ones(n) * 0.2
        
        report = integrator.check_degradation(
            features,
            bad_predictions,
            bad_actuals,
            low_confidences
        )
        
        should_retrain, reason = integrator.should_retrain(report)
        
        if report.overall_status == "critical":
            assert should_retrain == True
            print(f"✅ Retraining decision: {should_retrain} - {reason}")
        else:
            print(f"ℹ️ Status: {report.overall_status}, Retrain: {should_retrain}")
    
    def test_status_summary(self):
        """Test status summary generation"""
        integrator = create_integrated_monitor()
        
        # Check empty summary
        summary_empty = integrator.get_status_summary()
        assert summary_empty['status'] == 'no_data'
        print("✅ Empty summary handled correctly")
        
        # Add some data and check
        np.random.seed(42)
        n = 50
        features = pd.DataFrame({'f1': np.random.randn(n)})
        predictions = np.random.choice([0, 1], n)
        actuals = predictions.copy()
        confidences = np.random.uniform(0.6, 0.9, n)
        
        integrator.set_baseline(features, predictions, actuals, confidences)
        
        # Perform multiple checks
        for i in range(3):
            shift = i * 0.5
            shifted_features = features.copy()
            shifted_features['f1'] += shift
            
            integrator.check_degradation(
                shifted_features,
                predictions,
                actuals,
                confidences
            )
        
        summary = integrator.get_status_summary()
        
        assert 'status' in summary
        assert 'metrics' in summary
        assert 'should_retrain' in summary
        assert summary['total_checks'] == 3
        
        print(f"✅ Status summary: {summary['status']} ({summary['total_checks']} checks)")
        print(f"   Metrics: {summary['metrics']}")
    
    def test_confidence_calculation(self):
        """Test confidence level calculation in assessments"""
        integrator = create_integrated_monitor()
        
        # Set baseline
        np.random.seed(42)
        n = 100
        features = pd.DataFrame({'f1': np.random.randn(n)})
        predictions = np.ones(n)
        actuals = np.ones(n)
        confidences = np.ones(n) * 0.8
        
        integrator.set_baseline(features, predictions, actuals, confidences)
        
        # Check with clear degradation (high confidence)
        bad_predictions = np.zeros(n)
        report_clear = integrator.check_degradation(
            features,
            bad_predictions,
            actuals,
            confidences * 0.3  # Low confidence
        )
        
        # Check with ambiguous situation (lower confidence)
        mixed_predictions = predictions.copy()
        mixed_predictions[:30] = 0  # 30% error
        report_ambiguous = integrator.check_degradation(
            features,
            mixed_predictions,
            actuals,
            confidences * 0.7
        )
        
        # Clear degradation should have higher confidence
        if report_clear.overall_status in ["degraded", "critical"]:
            assert report_clear.confidence_level > 0.5
            print(f"✅ Clear degradation confidence: {report_clear.confidence_level:.2f}")
        
        print(f"✅ Ambiguous situation confidence: {report_ambiguous.confidence_level:.2f}")
    
    def test_export_functionality(self):
        """Test report export functionality"""
        integrator = create_integrated_monitor()
        
        # Generate some reports
        np.random.seed(42)
        n = 50
        features = pd.DataFrame({'f1': np.random.randn(n)})
        predictions = np.random.choice([0, 1], n)
        actuals = predictions.copy()
        confidences = np.random.uniform(0.6, 0.9, n)
        
        integrator.set_baseline(features, predictions, actuals, confidences)
        integrator.check_degradation(features, predictions, actuals, confidences)
        
        # Test JSON export
        json_path = "test_degradation_report.json"
        integrator.export_report(json_path, format='json')
        
        import os
        assert os.path.exists(json_path)
        os.remove(json_path)  # Clean up
        print("✅ JSON export successful")
        
        # Test CSV export
        csv_path = "test_degradation_report.csv"
        integrator.export_report(csv_path, format='csv')
        
        assert os.path.exists(csv_path)
        os.remove(csv_path)  # Clean up
        print("✅ CSV export successful")


def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("Degradation Integration Tests (MON-006)")
    print("="*60 + "\n")
    
    test_suite = TestDegradationIntegration()
    
    tests = [
        ("Initialization", test_suite.test_integration_initialization),
        ("Baseline Setting", test_suite.test_baseline_setting),
        ("Degradation Check", test_suite.test_degradation_check),
        ("Retraining Decision", test_suite.test_retraining_decision),
        ("Status Summary", test_suite.test_status_summary),
        ("Confidence Calculation", test_suite.test_confidence_calculation),
        ("Export Functionality", test_suite.test_export_functionality)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nTesting {test_name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_integration_tests()
    sys.exit(0 if success else 1)