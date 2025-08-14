"""
Tests for Phase 3 Model Performance Monitoring Components
Tests for tasks MON-001 to MON-008
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bot.ml.advanced_degradation_detector import (
    AdvancedDegradationDetector,
    DegradationType,
    DegradationAlert,
    DegradationMetrics
)


class TestAdvancedDegradationDetector:
    """Test suite for Advanced Degradation Detector"""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing"""
        return AdvancedDegradationDetector(
            metrics_window=100,
            drift_threshold=0.05,
            confidence_threshold=0.6,
            cusum_h=4.0,
            cusum_k=0.5
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        n_samples = 200
        
        # Create features with known distributions
        features = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.uniform(0, 1, n_samples),
            'feature3': np.random.exponential(1, n_samples)
        })
        
        # Create predictions and actuals
        predictions = np.random.choice([0, 1], n_samples)
        actuals = predictions.copy()
        # Add some errors
        error_indices = np.random.choice(n_samples, 20, replace=False)
        actuals[error_indices] = 1 - actuals[error_indices]
        
        # Create confidences
        confidences = np.random.uniform(0.5, 1.0, n_samples)
        
        return features, predictions, actuals, confidences
    
    def test_initialization(self, detector):
        """Test MON-001: Detector initialization"""
        assert detector.metrics_window == 100
        assert detector.drift_threshold == 0.05
        assert detector.confidence_threshold == 0.6
        assert detector.cusum_h == 4.0
        assert detector.cusum_k == 0.5
        assert len(detector.accuracy_history) == 0
        assert len(detector.alerts) == 0
    
    def test_update_baseline(self, detector, sample_data):
        """Test baseline update functionality"""
        features, predictions, actuals, confidences = sample_data
        
        # Update baseline
        detector.update_baseline(features[:100], predictions[:100], 
                                actuals[:100], confidences[:100])
        
        # Check baseline is set
        assert len(detector.baseline_distributions) == 3
        assert detector.target_accuracy is not None
        assert detector.cusum_pos == 0
        assert detector.cusum_neg == 0
    
    def test_ks_test_feature_drift(self, detector, sample_data):
        """Test MON-001: Kolmogorov-Smirnov test for feature drift"""
        features, predictions, actuals, confidences = sample_data
        
        # Set baseline
        detector.update_baseline(features[:100], predictions[:100], 
                                actuals[:100], confidences[:100])
        
        # Test with same distribution (no drift)
        ks_results = detector.detect_feature_drift(features[100:150])
        
        assert 'feature1' in ks_results
        assert 'feature2' in ks_results
        assert 'feature3' in ks_results
        
        # All p-values should be > threshold (no drift)
        for feature, (ks_stat, p_value) in ks_results.items():
            assert p_value > 0.01  # Some tolerance for randomness
            assert 0 <= ks_stat <= 1
        
        # Test with shifted distribution (drift)
        drifted_features = features[150:].copy()
        drifted_features['feature1'] += 2  # Shift mean by 2 std devs
        
        ks_results_drift = detector.detect_feature_drift(drifted_features)
        
        # Feature1 should show drift
        assert ks_results_drift['feature1'][1] < detector.drift_threshold
    
    def test_cusum_charts(self, detector):
        """Test MON-002: CUSUM charts for accuracy monitoring"""
        # Set target accuracy
        detector.target_accuracy = 0.9
        
        # Test with stable accuracy
        for _ in range(10):
            cusum_pos, cusum_neg = detector.update_cusum(0.9)
            assert cusum_pos == 0
            assert cusum_neg == 0
        
        # Test with decreasing accuracy
        for i in range(10):
            accuracy = 0.9 - (i * 0.02)  # Gradually decrease
            cusum_pos, cusum_neg = detector.update_cusum(accuracy)
            
            # Negative CUSUM should increase
            assert cusum_neg >= 0
            # Positive CUSUM should stay at 0
            assert cusum_pos == 0
        
        # Check that alert threshold is eventually exceeded
        assert detector.cusum_neg > 0
    
    def test_confidence_tracking(self, detector):
        """Test MON-003: Prediction confidence tracking"""
        # Generate confidence values
        high_confidences = np.random.uniform(0.8, 1.0, 150)
        low_confidences = np.random.uniform(0.4, 0.6, 150)
        
        # Track high confidences first
        for i in range(0, 150, 10):
            stats = detector.track_confidence(high_confidences[i:i+10])
        
        assert stats['mean'] > 0.7
        assert stats['below_threshold'] < 0.2
        
        # Track low confidences (should detect decay)
        for i in range(0, 150, 10):
            stats = detector.track_confidence(low_confidences[i:i+10])
        
        assert stats['mean'] < 0.6
        assert stats['below_threshold'] > 0.5
        
        # Should detect decay
        if 'decay_detected' in stats:
            assert stats['decay_detected'] == True
            assert stats['decay_rate'] > 0
    
    def test_error_pattern_analysis(self, detector):
        """Test MON-004: Error pattern analyzer"""
        # Create predictions with known error patterns
        n = 100
        predictions = np.ones(n)
        actuals = np.ones(n)
        
        # Add clustered errors
        actuals[10:20] = 0  # Cluster of errors
        actuals[50:55] = 0  # Another cluster
        
        # Analyze patterns
        features = pd.DataFrame({'f1': np.random.randn(n)})
        patterns = detector.analyze_error_patterns(predictions, actuals, features)
        
        assert 'error_rate' in patterns
        assert 'consecutive_errors' in patterns
        assert 'error_clustering' in patterns
        
        # Should detect clustering
        assert patterns['consecutive_errors'] == 10  # Longest cluster
        assert patterns['error_clustering'] > 0.3  # Some clustering
        assert patterns['error_rate'] == pytest.approx(0.15, rel=0.01)
    
    def test_comprehensive_degradation_check(self, detector, sample_data):
        """Test comprehensive degradation detection"""
        features, predictions, actuals, confidences = sample_data
        
        # Set baseline
        detector.update_baseline(features[:100], predictions[:100], 
                                actuals[:100], confidences[:100])
        
        # Check degradation with normal data
        metrics = detector.check_degradation(
            features[100:150],
            predictions[100:150],
            actuals[100:150],
            confidences[100:150]
        )
        
        assert isinstance(metrics, DegradationMetrics)
        assert metrics.status in DegradationType
        assert 0 <= metrics.degradation_score <= 1
        assert len(metrics.accuracy_trend) > 0
        assert len(metrics.feature_drift_scores) == 3
    
    def test_degradation_type_determination(self, detector, sample_data):
        """Test correct identification of degradation types"""
        features, predictions, actuals, confidences = sample_data
        
        # Set baseline
        detector.update_baseline(features[:100], predictions[:100], 
                                actuals[:100], confidences[:100])
        
        # Test feature drift detection
        drifted_features = features[100:150].copy()
        drifted_features += 3  # Large shift
        
        metrics = detector.check_degradation(
            drifted_features,
            predictions[100:150],
            actuals[100:150],
            confidences[100:150]
        )
        
        # Should detect feature drift
        assert metrics.status in [DegradationType.FEATURE_DRIFT, 
                                 DegradationType.CONCEPT_DRIFT]
    
    def test_alert_generation(self, detector, sample_data):
        """Test alert generation and management"""
        features, predictions, actuals, confidences = sample_data
        
        # Set baseline
        detector.update_baseline(features[:100], predictions[:100], 
                                actuals[:100], confidences[:100])
        
        # Create degraded scenario
        bad_predictions = np.random.choice([0, 1], 50)
        bad_actuals = 1 - bad_predictions  # All wrong
        low_confidences = np.ones(50) * 0.3  # Very low confidence
        
        # Check degradation
        metrics = detector.check_degradation(
            features[100:150],
            bad_predictions,
            bad_actuals,
            low_confidences
        )
        
        # Should generate alerts
        if metrics.degradation_score > 0.4:
            assert len(detector.alerts) > 0
            
            # Check alert properties
            alert = detector.alerts[-1]
            assert isinstance(alert, DegradationAlert)
            assert alert.severity in ["low", "medium", "high", "critical"]
            assert alert.recommended_action != ""
    
    def test_alert_cooldown(self, detector, sample_data):
        """Test alert cooldown to prevent spam"""
        features, predictions, actuals, confidences = sample_data
        
        # Set baseline
        detector.update_baseline(features[:100], predictions[:100], 
                                actuals[:100], confidences[:100])
        
        # Generate multiple degradation checks quickly
        initial_alert_count = len(detector.alerts)
        
        for _ in range(5):
            detector.check_degradation(
                features[100:150],
                np.zeros(50),  # Bad predictions
                np.ones(50),    # All wrong
                np.ones(50) * 0.3  # Low confidence
            )
        
        # Should not generate multiple alerts due to cooldown
        # (May generate 1-2 but not 5)
        assert len(detector.alerts) - initial_alert_count < 3
    
    def test_status_export(self, detector, sample_data):
        """Test status reporting and metric export"""
        features, predictions, actuals, confidences = sample_data
        
        # Set baseline and run detection
        detector.update_baseline(features[:100], predictions[:100], 
                                actuals[:100], confidences[:100])
        
        detector.check_degradation(
            features[100:150],
            predictions[100:150],
            actuals[100:150],
            confidences[100:150]
        )
        
        # Get status
        status = detector.get_status()
        
        assert status['operational'] == True
        assert status['metrics_collected'] > 0
        assert status['baseline_set'] == True
        assert 'cusum_status' in status
        assert status['features_monitored'] == 3
    
    def test_consecutive_error_counting(self, detector):
        """Test consecutive error counting logic"""
        # Test various error patterns
        errors1 = np.array([1, 1, 1, 0, 1, 1])  # Max 3 consecutive
        assert detector._count_consecutive_errors(errors1) == 3
        
        errors2 = np.array([0, 0, 0, 0, 0])  # No errors
        assert detector._count_consecutive_errors(errors2) == 0
        
        errors3 = np.array([1, 1, 1, 1, 1])  # All errors
        assert detector._count_consecutive_errors(errors3) == 5
        
        errors4 = np.array([1, 0, 1, 0, 1])  # Alternating
        assert detector._count_consecutive_errors(errors4) == 1
    
    def test_error_clustering_calculation(self, detector):
        """Test error clustering coefficient calculation"""
        # Random errors (low clustering)
        random_errors = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        clustering1 = detector._calculate_error_clustering(random_errors)
        assert clustering1 < 0.3
        
        # Clustered errors (high clustering)
        clustered_errors = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        clustering2 = detector._calculate_error_clustering(clustered_errors)
        assert clustering2 > 0.7
        
        # All errors (complete clustering)
        all_errors = np.array([1, 1, 1, 1, 1])
        clustering3 = detector._calculate_error_clustering(all_errors)
        assert clustering3 == 1.0
        
        # No errors (complete clustering)
        no_errors = np.array([0, 0, 0, 0, 0])
        clustering4 = detector._calculate_error_clustering(no_errors)
        assert clustering4 == 1.0


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""
    
    @pytest.fixture
    def detector(self):
        return AdvancedDegradationDetector()
    
    def test_gradual_degradation_scenario(self, detector):
        """Test detection of gradual model degradation"""
        np.random.seed(42)
        n_batches = 20
        batch_size = 50
        
        # Create baseline data
        baseline_features = pd.DataFrame({
            'f1': np.random.normal(0, 1, batch_size),
            'f2': np.random.normal(0, 1, batch_size)
        })
        baseline_pred = np.random.choice([0, 1], batch_size)
        baseline_actual = baseline_pred.copy()
        baseline_conf = np.random.uniform(0.7, 0.9, batch_size)
        
        detector.update_baseline(baseline_features, baseline_pred, 
                                baseline_actual, baseline_conf)
        
        # Simulate gradual degradation
        degradation_detected = False
        
        for i in range(n_batches):
            # Gradually shift features
            drift = i * 0.1
            features = pd.DataFrame({
                'f1': np.random.normal(drift, 1, batch_size),
                'f2': np.random.normal(0, 1 + drift, batch_size)
            })
            
            # Gradually decrease accuracy
            predictions = np.random.choice([0, 1], batch_size)
            actuals = predictions.copy()
            n_errors = min(batch_size // 2, int(i * 2))
            if n_errors > 0:
                error_idx = np.random.choice(batch_size, n_errors, replace=False)
                actuals[error_idx] = 1 - actuals[error_idx]
            
            # Gradually decrease confidence
            confidences = np.random.uniform(0.7 - i*0.02, 0.9 - i*0.02, batch_size)
            
            # Check for degradation
            metrics = detector.check_degradation(features, predictions, 
                                                actuals, confidences)
            
            if metrics.status != DegradationType.NONE:
                degradation_detected = True
                print(f"Degradation detected at batch {i}: {metrics.status}")
                break
        
        # Should detect degradation within reasonable time
        assert degradation_detected
        assert len(detector.alerts) > 0
    
    def test_sudden_drift_scenario(self, detector):
        """Test detection of sudden distribution shift"""
        np.random.seed(42)
        batch_size = 100
        
        # Normal operation
        for i in range(5):
            features = pd.DataFrame({
                'f1': np.random.normal(0, 1, batch_size),
                'f2': np.random.exponential(1, batch_size)
            })
            predictions = np.random.choice([0, 1], batch_size)
            actuals = predictions.copy()
            actuals[np.random.choice(batch_size, 10, replace=False)] = 1 - actuals[np.random.choice(batch_size, 10, replace=False)]
            confidences = np.random.uniform(0.6, 0.9, batch_size)
            
            if i == 0:
                detector.update_baseline(features, predictions, actuals, confidences)
            else:
                metrics = detector.check_degradation(features, predictions, 
                                                    actuals, confidences)
                assert metrics.status == DegradationType.NONE
        
        # Sudden drift
        drifted_features = pd.DataFrame({
            'f1': np.random.normal(5, 1, batch_size),  # Mean shifted by 5 std devs
            'f2': np.random.exponential(5, batch_size)  # Scale changed
        })
        
        metrics = detector.check_degradation(drifted_features, predictions, 
                                            actuals, confidences)
        
        # Should immediately detect drift
        assert metrics.status in [DegradationType.FEATURE_DRIFT, 
                                 DegradationType.CONCEPT_DRIFT]
        assert metrics.degradation_score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])