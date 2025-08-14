#\!/usr/bin/env python3
"""
Integration Test for Automated Retraining System
Phase 3, Week 5-6: ADAPT-009 through ADAPT-016

Tests the complete automated retraining pipeline including:
- Performance monitoring and triggers
- Scheduling and orchestration  
- Model versioning and validation
- Cost optimization and safety features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_ml_pipeline():
    """Create mock ML pipeline for testing"""
    pipeline = Mock()
    
    # Mock successful training with varying performance
    def mock_train_and_validate(X, y, model_name="test_model"):
        # Simulate varying model performance
        base_accuracy = 0.60
        noise = np.random.normal(0, 0.05)  # 5% standard deviation
        accuracy = max(0.4, min(0.8, base_accuracy + noise))
        
        performance = Mock()
        performance.accuracy = accuracy
        performance.precision = accuracy + np.random.normal(0, 0.01)
        performance.recall = accuracy + np.random.normal(0, 0.01)
        performance.f1_score = accuracy + np.random.normal(0, 0.01)
        performance.roc_auc = accuracy + 0.1 + np.random.normal(0, 0.01)
        
        logger.info(f"Mock training completed: accuracy={accuracy:.3f}")
        return performance
    
    pipeline.train_and_validate_model.side_effect = mock_train_and_validate
    return pipeline


def create_sample_data(n_samples=1000):
    """Create sample training/test data"""
    n_features = 20
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    # Create non-linear relationship
    y = (X[:, 0] + X[:, 1] * X[:, 2] + np.random.randn(n_samples) * 0.3) > 0
    
    # Create DataFrame
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    data['target'] = y.astype(int)
    data['timestamp'] = pd.date_range(
        start='2024-01-01', 
        periods=n_samples, 
        freq='1min'
    )
    
    return data


def test_auto_retraining_system():
    """Test the automated retraining system"""
    logger.info("Testing AutoRetrainingSystem...")
    
    try:
        from bot.ml.auto_retraining import (
            AutoRetrainingSystem, RetrainingConfig, RetrainingTrigger
        )
        
        # Create configuration
        config = RetrainingConfig(
            min_accuracy_threshold=0.55,
            max_retrainings_per_day=5,
            cooldown_period_hours=0,  # No cooldown for testing
            require_manual_approval=False,  # Automated for testing
            max_daily_retraining_cost=100.0
        )
        
        # Create mock components
        ml_pipeline = create_mock_ml_pipeline()
        db_manager = Mock()
        
        # Create retraining system
        system = AutoRetrainingSystem(
            config=config,
            ml_pipeline=ml_pipeline,
            db_manager=db_manager
        )
        
        # Test basic functionality
        assert not system.is_running
        
        # Test manual retraining request
        request_id = system.request_manual_retraining(
            reason="Integration test",
            priority=5,
            requested_by="test_user"
        )
        
        assert request_id.startswith("manual_")
        assert len(system.retraining_queue) == 1
        
        # Test status reporting
        status = system.get_retraining_status()
        assert status["queue_length"] == 1
        assert status["active_retrainings"] == 0
        
        logger.info("✓ AutoRetrainingSystem tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ AutoRetrainingSystem test failed: {e}")
        return False


def test_retraining_scheduler():
    """Test the retraining scheduler"""
    logger.info("Testing RetrainingScheduler...")
    
    try:
        from bot.ml.retraining_scheduler import (
            RetrainingScheduler, ScheduleConfig, ScheduleType, TaskPriority
        )
        
        # Create scheduler
        scheduler = RetrainingScheduler()
        
        # Test basic functionality
        assert not scheduler.is_running
        
        # Create test callback
        callback_executed = False
        def test_callback():
            nonlocal callback_executed
            callback_executed = True
            return "callback_result"
        
        # Create schedule configuration
        config = ScheduleConfig(
            task_id="test_schedule",
            schedule_type=ScheduleType.INTERVAL,
            name="Test Schedule",
            description="Test interval schedule",
            interval_minutes=60,
            priority=TaskPriority.MEDIUM
        )
        
        # Add schedule
        success = scheduler.add_schedule(config, test_callback)
        assert success
        assert "test_schedule" in scheduler.schedules
        
        # Test immediate trigger
        success = scheduler.trigger_immediate("test_schedule")
        assert success
        
        # Test status reporting
        status = scheduler.get_schedule_status()
        assert status["total_schedules"] == 1
        
        logger.info("✓ RetrainingScheduler tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ RetrainingScheduler test failed: {e}")
        return False


def test_model_versioning():
    """Test the model versioning system"""
    logger.info("Testing ModelVersioning...")
    
    try:
        from bot.ml.model_versioning import (
            ModelVersioning, ModelMetadata, ModelStage, VersionType, ModelFormat
        )
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create versioning system
            versioning = ModelVersioning(
                base_path=temp_dir,
                enable_git_tracking=False,  # Disable for testing
                max_versions_per_model=5
            )
            
            # Create test model
            test_model = Mock()
            test_model.predict = Mock(return_value=np.array([1, 0, 1]))
            
            # Create metadata
            metadata = ModelMetadata(
                model_id="test_model",
                version="1.0.0",
                stage=ModelStage.DEVELOPMENT,
                model_type="RandomForest",
                model_format=ModelFormat.JOBLIB,
                model_size_bytes=1024,
                model_hash="test_hash",
                training_dataset_hash="dataset_hash",
                training_start_time=datetime.now() - timedelta(hours=1),
                training_end_time=datetime.now(),
                training_duration_seconds=3600,
                training_config={"n_estimators": 100},
                validation_accuracy=0.85,
                validation_precision=0.83,
                validation_recall=0.87,
                validation_f1=0.85,
                validation_roc_auc=0.90,
                feature_set_version="1.0",
                created_by="test_user"
            )
            
            # Create version
            version = versioning.create_version(
                model_id="test_model",
                model_object=test_model,
                metadata=metadata
            )
            
            assert version == "1.0.0"
            assert "test_model" in versioning.models
            
            # Test promotion
            success = versioning.promote_version(
                model_id="test_model",
                version=version,
                target_stage=ModelStage.TESTING,
                promoted_by="test_user"
            )
            
            assert success
            
            # Test listing
            models = versioning.list_models()
            assert "test_model" in models
            
            versions = versioning.list_versions("test_model")
            assert "1.0.0" in versions
            
            logger.info("✓ ModelVersioning tests passed")
            return True
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        logger.error(f"✗ ModelVersioning test failed: {e}")
        return False


def test_integration():
    """Test complete integration"""
    logger.info("Testing complete integration...")
    
    try:
        from bot.ml.auto_retraining import (
            AutoRetrainingSystem, RetrainingConfig, 
            PRODUCTION_RETRAINING_CONFIG
        )
        from bot.ml.retraining_scheduler import (
            RetrainingScheduler, create_interval_schedule, TaskPriority
        )
        from bot.ml.model_versioning import ModelVersioning
        
        # Create components
        ml_pipeline = create_mock_ml_pipeline()
        db_manager = Mock()
        
        # Create retraining system with production config
        system = AutoRetrainingSystem(
            config=PRODUCTION_RETRAINING_CONFIG,
            ml_pipeline=ml_pipeline,
            db_manager=db_manager
        )
        
        # Create scheduler
        scheduler = RetrainingScheduler()
        
        # Create versioning system
        import tempfile
        temp_dir = tempfile.mkdtemp()
        versioning = ModelVersioning(
            base_path=temp_dir,
            enable_git_tracking=False
        )
        
        # Test workflow
        # 1. Schedule regular retraining
        schedule_config = create_interval_schedule(
            task_id="regular_retraining",
            name="Regular Model Retraining",
            interval_minutes=30,  # Every 30 minutes for testing
            priority=TaskPriority.MEDIUM
        )
        
        def retraining_callback():
            return system.request_manual_retraining(
                reason="Scheduled retraining",
                requested_by="scheduler"
            )
        
        scheduler.add_schedule(schedule_config, retraining_callback)
        
        # 2. Simulate performance degradation
        for i in range(100):
            # Historical good performance
            performance = Mock(accuracy=0.65 + np.random.normal(0, 0.01))
            system.performance_history.append(performance)
        
        for i in range(50):
            # Recent poor performance  
            performance = Mock(accuracy=0.52 + np.random.normal(0, 0.01))
            system.performance_history.append(performance)
        
        # Trigger performance check
        system._check_performance_degradation()
        
        # Should have created retraining request
        assert len(system.retraining_queue) > 0
        
        # 3. Test cost tracking
        initial_status = system.get_retraining_status()
        cost_summary = system.get_cost_summary()
        
        assert "daily_cost" in initial_status
        assert "total_lifetime_cost" in cost_summary
        
        logger.info("✓ Integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


def test_performance_benchmarks():
    """Test performance characteristics"""
    logger.info("Testing performance benchmarks...")
    
    try:
        from bot.ml.auto_retraining import AutoRetrainingSystem, RetrainingConfig
        
        # Create system
        config = RetrainingConfig(require_manual_approval=False)
        ml_pipeline = create_mock_ml_pipeline()
        db_manager = Mock()
        
        system = AutoRetrainingSystem(
            config=config,
            ml_pipeline=ml_pipeline,
            db_manager=db_manager
        )
        
        # Test decision latency
        start_time = time.time()
        
        # Add multiple requests
        for i in range(10):
            system.request_manual_retraining(
                reason=f"Test request {i}",
                priority=5
            )
        
        decision_time = time.time() - start_time
        
        # Should be very fast (< 100ms for 10 requests)
        assert decision_time < 0.1, f"Decision latency too high: {decision_time:.3f}s"
        
        # Test cost estimation performance
        start_time = time.time()
        
        for i in range(100):
            request = Mock()
            request.trigger = Mock()
            request.trigger.value = "manual"
            cost = system._estimate_retraining_cost(request)
            assert cost.estimated_total > 0
        
        estimation_time = time.time() - start_time
        
        # Should handle 100 cost estimations quickly
        assert estimation_time < 1.0, f"Cost estimation too slow: {estimation_time:.3f}s"
        
        logger.info(f"✓ Performance benchmarks passed:")
        logger.info(f"  - Decision latency: {decision_time*1000:.1f}ms for 10 requests")
        logger.info(f"  - Cost estimation: {estimation_time*1000:.1f}ms for 100 estimates")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance benchmark failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("Starting Automated Retraining System Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("AutoRetrainingSystem", test_auto_retraining_system),
        ("RetrainingScheduler", test_retraining_scheduler),
        ("ModelVersioning", test_model_versioning),
        ("Integration", test_integration),
        ("Performance", test_performance_benchmarks)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} tests...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:<20} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed\! Automated retraining system is ready.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Review implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
EOF < /dev/null