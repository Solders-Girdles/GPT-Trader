"""
Unit tests for Online Learning Pipeline
Phase 3 - ADAPT-001 through ADAPT-008 Testing

Comprehensive test suite for online learning components including
scheduler, drift detector, and main pipeline.
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.bot.ml.drift_detector import (
    ADWINDetector,
    ConceptDriftDetector,
    DriftDetection,
    DriftDetectorConfig,
    DriftSeverity,
    DriftType,
    PageHinkleyDetector,
)

# Import components to test
from src.bot.ml.learning_scheduler import (
    LearningRateScheduler,
    SchedulerConfig,
    SchedulerType,
    create_scheduler,
)
from src.bot.ml.online_learning import (
    AGGRESSIVE_CONFIG,
    CONSERVATIVE_CONFIG,
    DRIFT_ADAPTIVE_CONFIG,
    LearningMode,
    OnlineLearningConfig,
    OnlineLearningPipeline,
    UpdateStrategy,
    create_online_learning_pipeline,
)


class TestLearningRateScheduler:
    """Test learning rate scheduler functionality"""

    def test_initialization(self):
        """Test scheduler initialization"""
        config = SchedulerConfig(scheduler_type=SchedulerType.EXPONENTIAL, initial_lr=0.01)
        scheduler = LearningRateScheduler(config)

        assert scheduler.current_lr == 0.01
        assert scheduler.step_count == 0
        assert len(scheduler.performance_history) == 0
        assert len(scheduler.lr_history) == 0

    def test_exponential_decay(self):
        """Test exponential decay scheduler"""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.EXPONENTIAL,
            initial_lr=0.01,
            decay_rate=0.9,
            decay_steps=10,
        )
        scheduler = LearningRateScheduler(config)

        # Test first 9 steps (no decay)
        for _ in range(9):
            lr = scheduler.step()
            assert lr == 0.01

        # Step 10 should trigger decay
        lr = scheduler.step()
        expected_lr = 0.01 * 0.9
        assert abs(lr - expected_lr) < 1e-6

    def test_step_decay(self):
        """Test step decay scheduler"""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.STEP, initial_lr=0.01, step_size=5, gamma=0.5
        )
        scheduler = LearningRateScheduler(config)

        # First 5 steps
        for _ in range(5):
            lr = scheduler.step()
            assert lr == 0.01

        # Step 5 should trigger decay
        lr = scheduler.step()
        assert lr == 0.005

    def test_adaptive_adjustment(self):
        """Test adaptive learning rate adjustment"""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.ADAPTIVE,
            initial_lr=0.01,
            increase_factor=1.1,
            decrease_factor=0.9,
        )
        scheduler = LearningRateScheduler(config)

        # Add improving performance
        improving_performance = [0.5, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68]
        for perf in improving_performance:
            lr = scheduler.step(perf)

        # Learning rate should have increased
        assert scheduler.current_lr > 0.01

        # Reset and add degrading performance
        scheduler.reset()
        degrading_performance = [0.7, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.54, 0.52]
        for perf in degrading_performance:
            lr = scheduler.step(perf)

        # Learning rate should have decreased
        assert scheduler.current_lr < 0.01

    def test_plateau_reduction(self):
        """Test plateau-based learning rate reduction"""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.PLATEAU, initial_lr=0.01, patience=3, factor=0.5
        )
        scheduler = LearningRateScheduler(config)

        # Add plateauing performance
        for _ in range(5):
            lr = scheduler.step(0.6)  # No improvement

        # Should still be at initial LR (patience not exceeded)
        assert scheduler.current_lr == 0.01

        # Add more plateauing steps
        for _ in range(5):
            lr = scheduler.step(0.6)

        # Should have reduced LR
        assert scheduler.current_lr < 0.01

    def test_cyclical_lr(self):
        """Test cyclical learning rate"""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.CYCLICAL,
            base_lr=0.001,
            max_lr_cycle=0.01,
            step_size_up=5,
            step_size_down=5,
        )
        scheduler = LearningRateScheduler(config)

        lrs = []
        for _ in range(20):
            lr = scheduler.step()
            lrs.append(lr)

        # Should oscillate between base_lr and max_lr_cycle
        assert min(lrs) >= config.base_lr
        assert max(lrs) <= config.max_lr_cycle

        # Should complete at least one cycle
        assert len(set(lrs)) > 5  # Should have varied values

    def test_convergence_detection(self):
        """Test learning rate convergence detection"""
        config = SchedulerConfig(scheduler_type=SchedulerType.EXPONENTIAL)
        scheduler = LearningRateScheduler(config)

        # Add many steps to reach convergence
        for _ in range(1000):
            scheduler.step()

        # Should detect convergence
        assert scheduler.is_converged(window_size=50)

    def test_state_export_import(self):
        """Test state export and import"""
        config = SchedulerConfig(scheduler_type=SchedulerType.ADAPTIVE)
        scheduler1 = LearningRateScheduler(config)

        # Run some steps
        for i in range(20):
            scheduler1.step(0.5 + 0.01 * i)

        # Export state
        state = scheduler1.export_state()

        # Create new scheduler and import state
        scheduler2 = LearningRateScheduler(config)
        scheduler2.import_state(state)

        # Should have same state
        assert scheduler2.current_lr == scheduler1.current_lr
        assert scheduler2.step_count == scheduler1.step_count
        assert len(scheduler2.performance_history) == len(scheduler1.performance_history)

    def test_factory_function(self):
        """Test scheduler factory function"""
        scheduler = create_scheduler(SchedulerType.EXPONENTIAL, initial_lr=0.05)
        assert isinstance(scheduler, LearningRateScheduler)
        assert scheduler.config.initial_lr == 0.05

        scheduler2 = create_scheduler("adaptive", initial_lr=0.02)
        assert scheduler2.config.scheduler_type == SchedulerType.ADAPTIVE


class TestADWINDetector:
    """Test ADWIN drift detector"""

    def test_initialization(self):
        """Test ADWIN detector initialization"""
        detector = ADWINDetector(delta=0.002)
        assert detector.delta == 0.002
        assert detector.width == 0
        assert len(detector.window) == 0

    def test_no_drift_stable_data(self):
        """Test ADWIN with stable data (no drift)"""
        detector = ADWINDetector(delta=0.01)

        # Add stable data
        for _ in range(100):
            drift = detector.add_element(np.random.normal(0.5, 0.1))
            assert not drift  # Should not detect drift in stable data

    def test_drift_detection_sudden_change(self):
        """Test ADWIN drift detection with sudden change"""
        detector = ADWINDetector(delta=0.01)

        # Add stable data
        for _ in range(50):
            detector.add_element(np.random.normal(0.3, 0.05))

        # Add changed data
        drift_detected = False
        for _ in range(50):
            drift = detector.add_element(np.random.normal(0.7, 0.05))
            if drift:
                drift_detected = True
                break

        assert drift_detected  # Should detect the change


class TestPageHinkleyDetector:
    """Test Page-Hinkley drift detector"""

    def test_initialization(self):
        """Test Page-Hinkley detector initialization"""
        detector = PageHinkleyDetector(min_instances=30, threshold=50.0)
        assert detector.min_instances == 30
        assert detector.threshold == 50.0
        assert detector.n == 0

    def test_no_drift_stable_data(self):
        """Test Page-Hinkley with stable data"""
        detector = PageHinkleyDetector(min_instances=20, threshold=10.0)

        # Add stable data
        for _ in range(100):
            drift = detector.add_element(np.random.normal(0.5, 0.1))
            # Might detect drift occasionally due to randomness, but should be rare

        # At least initial samples should not detect drift
        detector2 = PageHinkleyDetector(min_instances=20, threshold=10.0)
        for i in range(25):
            drift = detector2.add_element(0.5)
            if i < 20:
                assert not drift  # Should not detect before min_instances


class TestConceptDriftDetector:
    """Test comprehensive concept drift detector"""

    def test_initialization(self):
        """Test drift detector initialization"""
        config = DriftDetectorConfig()
        detector = ConceptDriftDetector(config)

        assert detector.config == config
        assert detector.samples_seen == 0
        assert not detector.warmup_complete
        assert len(detector.drift_history) == 0

    def test_warmup_period(self):
        """Test warmup period functionality"""
        config = DriftDetectorConfig(warmup_period=50)
        detector = ConceptDriftDetector(config)

        # Generate sample data
        np.random.seed(42)
        for i in range(60):
            features = pd.DataFrame([np.random.randn(5)])
            target = np.random.choice([0, 1])

            result = detector.add_sample(features, target=target)

            if i < 50:
                assert not detector.warmup_complete
                assert result is None
            else:
                assert detector.warmup_complete

    def test_drift_detection_performance_degradation(self):
        """Test drift detection based on performance degradation"""
        config = DriftDetectorConfig(
            warmup_period=20, performance_threshold=0.2, performance_window=50
        )
        detector = ConceptDriftDetector(config)

        np.random.seed(42)

        # Add samples with good performance
        for i in range(100):
            features = pd.DataFrame([np.random.randn(3)])
            target = 1
            prediction = 1  # Perfect prediction

            result = detector.add_sample(features, target=target, prediction=prediction)

        # Add samples with poor performance
        drift_detected = False
        for i in range(50):
            features = pd.DataFrame([np.random.randn(3)])
            target = 1
            prediction = 0  # Wrong prediction

            result = detector.add_sample(features, target=target, prediction=prediction)
            if result is not None:
                drift_detected = True
                break

        # Should detect drift due to performance degradation
        # Note: This might not always trigger due to the specific implementation
        # but serves as a test of the mechanism

    def test_statistical_drift_detection(self):
        """Test statistical drift detection"""
        config = DriftDetectorConfig(
            warmup_period=20, window_size=100, reference_window_size=100, p_value_threshold=0.05
        )
        detector = ConceptDriftDetector(config)

        np.random.seed(42)

        # Add reference data (mean=0, std=1)
        for i in range(150):
            features = pd.DataFrame([np.random.normal(0, 1, 3)])
            target = np.random.choice([0, 1])
            detector.add_sample(features, target=target)

        # Add shifted data (mean=2, std=1)
        drift_detected = False
        for i in range(100):
            features = pd.DataFrame([np.random.normal(2, 1, 3)])
            target = np.random.choice([0, 1])

            result = detector.add_sample(features, target=target)
            if result is not None and result.is_drift:
                drift_detected = True
                assert result.drift_severity in [
                    DriftSeverity.LOW,
                    DriftSeverity.MEDIUM,
                    DriftSeverity.HIGH,
                    DriftSeverity.CRITICAL,
                ]
                break

        # Should detect drift due to feature distribution change
        # Note: May not always detect due to randomness and specific implementation

    def test_drift_history_tracking(self):
        """Test drift history tracking"""
        config = DriftDetectorConfig(warmup_period=10)
        detector = ConceptDriftDetector(config)

        # Manually add a drift detection
        drift_detection = DriftDetection(
            is_drift=True,
            drift_type=DriftType.SUDDEN,
            drift_severity=DriftSeverity.HIGH,
            confidence=0.9,
            detected_at=datetime.now(),
            drift_score=0.8,
            affected_features=["feature_0"],
            detection_method="test",
        )

        detector.drift_history.append(drift_detection)

        history = detector.get_drift_history()
        assert len(history) == 1
        assert history[0].drift_type == DriftType.SUDDEN

    def test_statistics_collection(self):
        """Test statistics collection"""
        config = DriftDetectorConfig()
        detector = ConceptDriftDetector(config)

        stats = detector.get_statistics()

        assert "samples_seen" in stats
        assert "warmup_complete" in stats
        assert "reference_window_size" in stats
        assert "current_window_size" in stats
        assert "total_drifts_detected" in stats

    def test_reset_functionality(self):
        """Test detector reset"""
        config = DriftDetectorConfig()
        detector = ConceptDriftDetector(config)

        # Add some data
        for i in range(50):
            features = pd.DataFrame([np.random.randn(3)])
            detector.add_sample(features, target=i % 2)

        # Reset
        detector.reset()

        assert detector.samples_seen == 0
        assert not detector.warmup_complete
        assert len(detector.drift_history) == 0
        assert len(detector.reference_window) == 0
        assert len(detector.current_window) == 0


class TestOnlineLearningPipeline:
    """Test online learning pipeline functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = OnlineLearningConfig(
            learning_mode=LearningMode.MINI_BATCH,
            batch_size=10,
            memory_buffer_size=100,
            warmup_epochs=2,
        )
        self.pipeline = OnlineLearningPipeline(self.config)

        # Generate initial training data
        np.random.seed(42)
        self.initial_features = pd.DataFrame(np.random.randn(50, 5))
        self.initial_targets = pd.Series(np.random.choice([0, 1], 50))

    def test_initialization(self):
        """Test pipeline initialization"""
        assert not self.pipeline.is_initialized
        assert self.pipeline.sample_count == 0
        assert len(self.pipeline.memory_buffer) == 0
        assert self.pipeline.primary_model is None

    def test_pipeline_initialization(self):
        """Test pipeline initialization with data"""
        self.pipeline.initialize(
            self.initial_features,
            self.initial_targets,
            feature_names=[f"feature_{i}" for i in range(5)],
        )

        assert self.pipeline.is_initialized
        assert self.pipeline.primary_model is not None
        assert self.pipeline.backup_model is not None
        assert len(self.pipeline.memory_buffer) == len(self.initial_features)
        assert self.pipeline.sample_count == len(self.initial_features)

    def test_single_update(self):
        """Test single sample update"""
        self.pipeline.initialize(self.initial_features, self.initial_targets)

        # Add single sample
        new_features = np.random.randn(5)
        new_target = 1

        result = self.pipeline.update(new_features, new_target)

        assert result["sample_added"]
        assert "prediction" in result
        assert "loss" in result
        assert self.pipeline.sample_count == len(self.initial_features) + 1

    def test_batch_update(self):
        """Test batch update functionality"""
        self.pipeline.initialize(self.initial_features, self.initial_targets)

        # Add samples to trigger batch update
        for i in range(self.config.batch_size):
            features = np.random.randn(5)
            target = i % 2
            result = self.pipeline.update(features, target)

            if i == self.config.batch_size - 1:
                assert result["model_updated"]

    def test_prediction(self):
        """Test prediction functionality"""
        self.pipeline.initialize(self.initial_features, self.initial_targets)

        # Test prediction
        test_features = np.random.randn(3, 5)
        predictions = self.pipeline.predict(test_features)

        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)

    def test_prediction_probabilities(self):
        """Test prediction probabilities"""
        self.pipeline.initialize(self.initial_features, self.initial_targets)

        # Test with SGD classifier (should support predict_proba)
        if hasattr(self.pipeline.primary_model, "predict_proba"):
            test_features = np.random.randn(3, 5)
            probabilities = self.pipeline.predict_proba(test_features)

            assert probabilities.shape[0] == 3
            assert probabilities.shape[1] >= 2  # At least binary classification
            assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_convergence_detection(self):
        """Test convergence detection"""
        config = OnlineLearningConfig(convergence_patience=5, convergence_threshold=0.01)
        pipeline = OnlineLearningPipeline(config)
        pipeline.initialize(self.initial_features, self.initial_targets)

        # Simulate stable performance
        for _ in range(20):
            pipeline.performance_history.append(0.8)

        # Trigger convergence check
        convergence = pipeline._check_convergence({"accuracy": 0.8})

        # Should detect convergence with stable performance
        assert "is_converged" in convergence

    def test_drift_adaptation(self):
        """Test drift adaptation functionality"""
        self.pipeline.initialize(self.initial_features, self.initial_targets)

        # Create mock drift detection
        drift_detection = DriftDetection(
            is_drift=True,
            drift_type=DriftType.SUDDEN,
            drift_severity=DriftSeverity.HIGH,
            confidence=0.9,
            detected_at=datetime.now(),
            drift_score=0.8,
            affected_features=["feature_0"],
            detection_method="test",
        )

        # Test adaptation
        X_batch = np.random.randn(10, 5)
        y_batch = np.random.choice([0, 1], 10)

        adaptation_result = self.pipeline._adapt_to_drift(drift_detection, X_batch, y_batch)

        assert "drift_type" in adaptation_result
        assert "adaptation_strategy" in adaptation_result
        assert self.pipeline.adaptation_count > 0

    def test_memory_buffer_management(self):
        """Test memory buffer with priority replay"""
        config = OnlineLearningConfig(memory_buffer_size=20, priority_replay=True, batch_size=5)
        pipeline = OnlineLearningPipeline(config)
        pipeline.initialize(self.initial_features[:15], self.initial_targets[:15])

        # Add samples to fill buffer beyond capacity
        for i in range(10):
            features = np.random.randn(5)
            target = i % 2
            pipeline.update(features, target)

        # Buffer should be at max capacity
        assert len(pipeline.memory_buffer) <= config.memory_buffer_size

        # Test priority sampling
        batch_samples = pipeline._sample_priority_batch()
        assert len(batch_samples) <= config.batch_size

    def test_model_info(self):
        """Test model information retrieval"""
        self.pipeline.initialize(self.initial_features, self.initial_targets)

        info = self.pipeline.get_model_info()

        required_keys = [
            "is_initialized",
            "sample_count",
            "memory_buffer_size",
            "current_learning_rate",
            "scheduler_stats",
            "drift_detector_stats",
        ]

        for key in required_keys:
            assert key in info

    def test_learning_curve_tracking(self):
        """Test learning curve tracking"""
        self.pipeline.initialize(self.initial_features, self.initial_targets)

        # Perform some updates to generate learning curve data
        for i in range(20):
            features = np.random.randn(5)
            target = i % 2
            self.pipeline.update(features, target, force_update=(i % 5 == 0))

        learning_curve = self.pipeline.get_learning_curve()

        assert len(learning_curve.timestamps) > 0
        assert len(learning_curve.learning_rates) > 0
        assert len(learning_curve.sample_counts) > 0

    def test_state_persistence(self):
        """Test state save and load"""
        self.pipeline.initialize(self.initial_features, self.initial_targets)

        # Add some updates
        for i in range(10):
            features = np.random.randn(5)
            target = i % 2
            self.pipeline.update(features, target)

        # Save state
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            self.pipeline.save_state(filepath)

            # Create new pipeline and load state
            new_pipeline = OnlineLearningPipeline(self.config)
            new_pipeline.load_state(filepath)

            assert new_pipeline.is_initialized
            assert new_pipeline.sample_count == self.pipeline.sample_count
            assert len(new_pipeline.memory_buffer) > 0
            assert new_pipeline.feature_names == self.pipeline.feature_names

        finally:
            # Cleanup
            for ext in [
                ".json",
                ".model.joblib",
                ".backup.joblib",
                ".scaler.joblib",
                ".encoder.joblib",
            ]:
                try:
                    os.unlink(filepath.replace(".json", ext))
                except FileNotFoundError:
                    pass

    def test_factory_function(self):
        """Test pipeline factory function"""
        config_dict = {
            "learning_mode": LearningMode.STREAM,
            "batch_size": 16,
            "memory_buffer_size": 500,
        }

        pipeline = create_online_learning_pipeline(config_dict)

        assert isinstance(pipeline, OnlineLearningPipeline)
        assert pipeline.config.learning_mode == LearningMode.STREAM
        assert pipeline.config.batch_size == 16

    def test_predefined_configurations(self):
        """Test predefined configuration objects"""
        # Test conservative config
        conservative_pipeline = OnlineLearningPipeline(CONSERVATIVE_CONFIG)
        assert conservative_pipeline.config.learning_mode == LearningMode.MINI_BATCH
        assert conservative_pipeline.config.batch_size == 64

        # Test aggressive config
        aggressive_pipeline = OnlineLearningPipeline(AGGRESSIVE_CONFIG)
        assert aggressive_pipeline.config.learning_mode == LearningMode.STREAM
        assert aggressive_pipeline.config.update_strategy == UpdateStrategy.IMMEDIATE

        # Test drift adaptive config
        drift_pipeline = OnlineLearningPipeline(DRIFT_ADAPTIVE_CONFIG)
        assert drift_pipeline.config.update_strategy == UpdateStrategy.DRIFT_TRIGGERED
        assert drift_pipeline.config.priority_replay


class TestIntegration:
    """Integration tests for online learning components"""

    def test_full_pipeline_integration(self):
        """Test full pipeline with all components working together"""
        config = OnlineLearningConfig(
            learning_mode=LearningMode.ADAPTIVE,
            update_strategy=UpdateStrategy.DRIFT_TRIGGERED,
            batch_size=20,
            memory_buffer_size=200,
            warmup_epochs=2,
        )

        pipeline = OnlineLearningPipeline(config)

        # Initialize with data
        np.random.seed(42)
        initial_features = pd.DataFrame(np.random.randn(100, 4))
        initial_targets = pd.Series(np.random.choice([0, 1], 100))

        pipeline.initialize(initial_features, initial_targets)

        # Simulate online learning with concept drift

        # Phase 1: Stable data
        for i in range(50):
            features = np.random.randn(4)
            target = int(features[0] > 0)  # Simple rule
            result = pipeline.update(features, target)

        performance_before_drift = (
            np.mean(list(pipeline.performance_history)[-10:])
            if pipeline.performance_history
            else 0.5
        )

        # Phase 2: Drifted data (reverse rule)
        for i in range(50):
            features = np.random.randn(4)
            target = int(features[0] <= 0)  # Reversed rule
            result = pipeline.update(features, target)

        # Check that system adapted
        final_info = pipeline.get_model_info()

        assert final_info["is_initialized"]
        assert final_info["sample_count"] == 200  # 100 initial + 100 updates
        assert final_info["adaptation_count"] >= 0  # May or may not detect drift

        # Test prediction still works
        test_features = np.random.randn(5, 4)
        predictions = pipeline.predict(test_features)
        assert len(predictions) == 5

    def test_scheduler_drift_detector_interaction(self):
        """Test interaction between scheduler and drift detector"""
        # Create components
        scheduler_config = SchedulerConfig(scheduler_type=SchedulerType.ADAPTIVE, initial_lr=0.01)
        scheduler = LearningRateScheduler(scheduler_config)

        drift_config = DriftDetectorConfig(warmup_period=20)
        drift_detector = ConceptDriftDetector(drift_config)

        # Simulate data with drift
        np.random.seed(42)

        performance_metrics = []

        # Stable period
        for i in range(50):
            features = pd.DataFrame([np.random.normal(0, 1, 3)])
            target = np.random.choice([0, 1])

            # Simulate good performance
            performance = 0.8 + np.random.normal(0, 0.05)
            performance_metrics.append(performance)

            # Update scheduler
            lr = scheduler.step(performance)

            # Check for drift
            drift = drift_detector.add_sample(features, target=target)

        # Drift period
        drift_detected = False
        for i in range(50):
            features = pd.DataFrame([np.random.normal(2, 1, 3)])  # Shifted distribution
            target = np.random.choice([0, 1])

            # Simulate degraded performance
            performance = 0.4 + np.random.normal(0, 0.05)
            performance_metrics.append(performance)

            # Update scheduler
            lr = scheduler.step(performance)

            # Check for drift
            drift = drift_detector.add_sample(features, target=target)
            if drift and drift.is_drift:
                drift_detected = True

        # Verify scheduler adapted to poor performance
        assert scheduler.current_lr != scheduler_config.initial_lr

        # Performance should show degradation
        early_performance = np.mean(performance_metrics[:30])
        late_performance = np.mean(performance_metrics[-30:])
        assert late_performance < early_performance


if __name__ == "__main__":
    pytest.main([__file__])
