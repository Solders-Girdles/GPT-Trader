"""
Tests for Automated Retraining System
Phase 3, Week 5-6: ADAPT-009 through ADAPT-016
"""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Import the modules we're testing
from bot.ml.auto_retraining import (
    AutoRetrainingSystem,
    RetrainingConfig,
    RetrainingCost,
    RetrainingRequest,
    RetrainingResult,
    RetrainingStatus,
    RetrainingTrigger,
)
from bot.ml.model_versioning import (
    ModelFormat,
    ModelMetadata,
    ModelStage,
    ModelVersioning,
    VersionType,
)
from bot.ml.retraining_scheduler import (
    RetrainingScheduler,
    ScheduleConfig,
    ScheduleType,
)


class TestAutoRetrainingSystem:
    """Test cases for AutoRetrainingSystem"""

    @pytest.fixture
    def mock_ml_pipeline(self):
        """Mock ML pipeline"""
        pipeline = Mock()
        pipeline.train_and_validate_model.return_value = Mock(
            accuracy=0.65, precision=0.63, recall=0.67, f1_score=0.65, roc_auc=0.70
        )
        return pipeline

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager"""
        return Mock()

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return RetrainingConfig(
            min_accuracy_threshold=0.55,
            max_retrainings_per_day=2,
            cooldown_period_hours=1,  # Short for testing
            require_manual_approval=False,  # Disable for automated testing
            max_daily_retraining_cost=50.0,
        )

    @pytest.fixture
    def retraining_system(self, config, mock_ml_pipeline, mock_db_manager):
        """Create retraining system for testing"""
        return AutoRetrainingSystem(
            config=config, ml_pipeline=mock_ml_pipeline, db_manager=mock_db_manager
        )

    def test_initialization(self, retraining_system):
        """Test system initialization"""
        assert not retraining_system.is_running
        assert len(retraining_system.retraining_queue) == 0
        assert len(retraining_system.current_retrainings) == 0

    def test_start_stop(self, retraining_system):
        """Test starting and stopping the system"""
        # Start system
        retraining_system.start()
        assert retraining_system.is_running

        # Stop system
        retraining_system.stop()
        assert not retraining_system.is_running

    def test_manual_retraining_request(self, retraining_system):
        """Test manual retraining request"""
        request_id = retraining_system.request_manual_retraining(
            reason="Test manual retraining", priority=8, requested_by="test_user"
        )

        assert request_id.startswith("manual_")
        assert len(retraining_system.retraining_queue) == 1

        request = retraining_system.retraining_queue[0]
        assert request.trigger == RetrainingTrigger.MANUAL
        assert request.priority == 8
        assert request.reason == "Test manual retraining"

    def test_performance_degradation_trigger(self, retraining_system):
        """Test performance degradation trigger"""
        # Add performance history that shows degradation
        for i in range(100):
            # Historical good performance
            performance = Mock(accuracy=0.65 + np.random.normal(0, 0.01))
            retraining_system.performance_history.append(performance)

        for i in range(50):
            # Recent poor performance
            performance = Mock(accuracy=0.50 + np.random.normal(0, 0.01))
            retraining_system.performance_history.append(performance)

        # Trigger check
        retraining_system._check_performance_degradation()

        # Should have created a retraining request
        assert len(retraining_system.retraining_queue) > 0
        request = retraining_system.retraining_queue[0]
        assert request.trigger == RetrainingTrigger.PERFORMANCE_DEGRADATION

    def test_emergency_retraining_trigger(self, retraining_system):
        """Test emergency retraining trigger"""
        # Add performance history with sudden drop
        for i in range(10):
            if i < 5:
                performance = Mock(accuracy=0.65)
            else:
                performance = Mock(accuracy=0.40)  # Sudden drop
            retraining_system.performance_history.append(performance)

        # Trigger check
        retraining_system._check_emergency_conditions()

        # Should have created an emergency request
        assert len(retraining_system.retraining_queue) > 0
        request = retraining_system.retraining_queue[0]
        assert request.trigger == RetrainingTrigger.EMERGENCY
        assert request.priority == 10
        assert not request.approval_required  # Emergency bypasses approval

    def test_cost_estimation(self, retraining_system):
        """Test cost estimation"""
        request = RetrainingRequest(
            trigger=RetrainingTrigger.MANUAL,
            priority=5,
            requested_at=datetime.now(),
            requested_by="test",
            model_id="test_model",
            reason="test",
        )

        cost = retraining_system._estimate_retraining_cost(request)

        assert isinstance(cost, RetrainingCost)
        assert cost.computational_cost > 0
        assert cost.time_cost > 0
        assert cost.estimated_total > 0
        assert cost.roi_estimate > 0

    def test_cost_limits(self, retraining_system):
        """Test cost limit enforcement"""
        # Create expensive request
        expensive_cost = RetrainingCost(
            computational_cost=100.0,
            time_cost=10.0,
            opportunity_cost=50.0,
            resource_usage={},
            estimated_total=150.0,
        )

        # Should exceed daily limits
        assert not retraining_system._is_within_cost_limits(expensive_cost)

        # Create affordable request
        affordable_cost = RetrainingCost(
            computational_cost=5.0,
            time_cost=1.0,
            opportunity_cost=2.0,
            resource_usage={},
            estimated_total=7.0,
        )

        # Should be within limits
        assert retraining_system._is_within_cost_limits(affordable_cost)

    def test_cooldown_period(self, retraining_system):
        """Test cooldown period enforcement"""
        # Set recent retraining time
        retraining_system.last_retraining = datetime.now() - timedelta(minutes=30)

        # Should be in cooldown (config has 1 hour cooldown)
        assert retraining_system._is_in_cooldown()

        # Set old retraining time
        retraining_system.last_retraining = datetime.now() - timedelta(hours=2)

        # Should not be in cooldown
        assert not retraining_system._is_in_cooldown()

    def test_daily_limits(self, retraining_system):
        """Test daily retraining limits"""
        # Add completed retrainings for today
        today = datetime.now().date()
        for i in range(2):  # Config allows 2 per day
            result = RetrainingResult(
                request_id=f"test_{i}",
                status=RetrainingStatus.COMPLETED,
                started_at=datetime.combine(today, datetime.min.time()),
                old_model_id="test",
            )
            retraining_system.retraining_history.append(result)

        # Should exceed daily limits
        assert retraining_system._exceeds_daily_limits()

    def test_status_reporting(self, retraining_system):
        """Test status reporting"""
        status = retraining_system.get_retraining_status()

        assert "is_running" in status
        assert "queue_length" in status
        assert "active_retrainings" in status
        assert "daily_cost" in status
        assert "monthly_cost" in status

    def test_cost_summary(self, retraining_system):
        """Test cost summary"""
        # Add some completed retrainings with costs
        for i in range(3):
            cost = RetrainingCost(
                computational_cost=5.0,
                time_cost=1.0,
                opportunity_cost=0.0,
                resource_usage={},
                estimated_total=5.0,
                actual_total=5.0,
            )

            result = RetrainingResult(
                request_id=f"test_{i}",
                status=RetrainingStatus.COMPLETED,
                started_at=datetime.now(),
                old_model_id="test",
                actual_cost=cost,
            )
            retraining_system.retraining_history.append(result)

        summary = retraining_system.get_cost_summary()

        assert "total_lifetime_cost" in summary
        assert "cost_per_retraining" in summary
        assert "successful_retrainings" in summary
        assert summary["cost_per_retraining"] == 5.0


class TestRetrainingScheduler:
    """Test cases for RetrainingScheduler"""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler for testing"""
        return RetrainingScheduler()

    @pytest.fixture
    def test_callback(self):
        """Test callback function"""

        def callback():
            return "callback_executed"

        return callback

    def test_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert not scheduler.is_running
        assert len(scheduler.schedules) == 0
        assert len(scheduler.task_queue) == 0

    def test_start_stop(self, scheduler):
        """Test starting and stopping scheduler"""
        scheduler.start()
        assert scheduler.is_running

        scheduler.stop()
        assert not scheduler.is_running

    def test_add_cron_schedule(self, scheduler, test_callback):
        """Test adding cron schedule"""
        config = ScheduleConfig(
            task_id="test_cron",
            schedule_type=ScheduleType.CRON,
            name="Test Cron",
            description="Test cron schedule",
            cron_expression="0 2 * * *",  # 2 AM daily
        )

        success = scheduler.add_schedule(config, test_callback)
        assert success
        assert "test_cron" in scheduler.schedules
        assert len(scheduler.task_queue) == 1

    def test_add_interval_schedule(self, scheduler, test_callback):
        """Test adding interval schedule"""
        config = ScheduleConfig(
            task_id="test_interval",
            schedule_type=ScheduleType.INTERVAL,
            name="Test Interval",
            description="Test interval schedule",
            interval_minutes=60,
        )

        success = scheduler.add_schedule(config, test_callback)
        assert success
        assert "test_interval" in scheduler.schedules

    def test_remove_schedule(self, scheduler, test_callback):
        """Test removing schedule"""
        config = ScheduleConfig(
            task_id="test_remove",
            schedule_type=ScheduleType.INTERVAL,
            name="Test Remove",
            description="Test removal",
            interval_minutes=30,
        )

        scheduler.add_schedule(config, test_callback)
        assert "test_remove" in scheduler.schedules

        success = scheduler.remove_schedule("test_remove")
        assert success
        assert "test_remove" not in scheduler.schedules

    def test_trigger_immediate(self, scheduler, test_callback):
        """Test immediate task triggering"""
        config = ScheduleConfig(
            task_id="test_immediate",
            schedule_type=ScheduleType.INTERVAL,
            name="Test Immediate",
            description="Test immediate execution",
            interval_minutes=60,
        )

        scheduler.add_schedule(config, test_callback)

        success = scheduler.trigger_immediate("test_immediate")
        assert success

        # Should have added immediate task to queue
        assert len(scheduler.task_queue) == 2  # Original + immediate

    def test_schedule_validation(self, scheduler, test_callback):
        """Test schedule validation"""
        # Invalid cron expression
        invalid_config = ScheduleConfig(
            task_id="invalid_cron",
            schedule_type=ScheduleType.CRON,
            name="Invalid Cron",
            description="Invalid cron expression",
            cron_expression="invalid_cron",
        )

        success = scheduler.add_schedule(invalid_config, test_callback)
        assert not success

        # Missing interval
        invalid_interval = ScheduleConfig(
            task_id="invalid_interval",
            schedule_type=ScheduleType.INTERVAL,
            name="Invalid Interval",
            description="Missing interval",
        )

        success = scheduler.add_schedule(invalid_interval, test_callback)
        assert not success

    def test_get_status(self, scheduler):
        """Test status reporting"""
        status = scheduler.get_schedule_status()

        assert "is_running" in status
        assert "total_schedules" in status
        assert "queued_tasks" in status
        assert "running_tasks" in status


class TestModelVersioning:
    """Test cases for ModelVersioning"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def versioning(self, temp_dir):
        """Create model versioning system for testing"""
        return ModelVersioning(
            base_path=temp_dir,
            enable_git_tracking=False,  # Disable Git for testing
            max_versions_per_model=5,
        )

    @pytest.fixture
    def test_model(self):
        """Create test model"""
        # Simple mock model
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 0, 1]))
        return model

    @pytest.fixture
    def test_metadata(self):
        """Create test metadata"""
        return ModelMetadata(
            model_id="test_model",
            version="1.0.0",
            stage=ModelStage.DEVELOPMENT,
            model_type="RandomForest",
            model_format=ModelFormat.JOBLIB,
            model_size_bytes=1024,
            model_hash="abc123",
            training_dataset_hash="def456",
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
            created_by="test_user",
        )

    def test_initialization(self, versioning, temp_dir):
        """Test versioning system initialization"""
        assert versioning.base_path == Path(temp_dir)
        assert (Path(temp_dir) / "metadata").exists()
        assert (Path(temp_dir) / "models").exists()
        assert (Path(temp_dir) / "archive").exists()

    def test_create_version(self, versioning, test_model, test_metadata):
        """Test creating new model version"""
        version = versioning.create_version(
            model_id="test_model", model_object=test_model, metadata=test_metadata
        )

        assert version == "1.0.0"
        assert "test_model" in versioning.models
        assert len(versioning.models["test_model"]) == 1
        assert versioning.current_versions["test_model"] == "1.0.0"

    def test_version_promotion(self, versioning, test_model, test_metadata):
        """Test version promotion"""
        # Create version
        version = versioning.create_version(
            model_id="test_model", model_object=test_model, metadata=test_metadata
        )

        # Promote to testing
        success = versioning.promote_version(
            model_id="test_model",
            version=version,
            target_stage=ModelStage.TESTING,
            promoted_by="test_user",
        )

        assert success
        metadata = versioning.get_version_metadata("test_model", version)
        assert metadata.stage == ModelStage.TESTING
        assert metadata.promoted_by == "test_user"

    def test_model_loading(self, versioning, test_model, test_metadata):
        """Test loading model"""
        # Create version
        version = versioning.create_version(
            model_id="test_model", model_object=test_model, metadata=test_metadata
        )

        # Load model
        loaded_model = versioning.load_model("test_model", version)
        assert loaded_model is not None

    def test_version_comparison(self, versioning, test_model):
        """Test version comparison"""
        # Create two versions with different performance
        metadata1 = ModelMetadata(
            model_id="test_model",
            version="1.0.0",
            stage=ModelStage.DEVELOPMENT,
            model_type="RandomForest",
            model_format=ModelFormat.JOBLIB,
            model_size_bytes=1024,
            model_hash="abc123",
            training_dataset_hash="def456",
            training_start_time=datetime.now(),
            training_end_time=datetime.now(),
            training_duration_seconds=3600,
            training_config={},
            validation_accuracy=0.80,
            validation_precision=0.78,
            validation_recall=0.82,
            validation_f1=0.80,
            validation_roc_auc=0.85,
            feature_set_version="1.0",
            created_by="test_user",
        )

        metadata2 = ModelMetadata(
            model_id="test_model",
            version="2.0.0",
            stage=ModelStage.DEVELOPMENT,
            model_type="RandomForest",
            model_format=ModelFormat.JOBLIB,
            model_size_bytes=1024,
            model_hash="xyz789",
            training_dataset_hash="def456",
            training_start_time=datetime.now(),
            training_end_time=datetime.now(),
            training_duration_seconds=3600,
            training_config={},
            validation_accuracy=0.85,
            validation_precision=0.83,
            validation_recall=0.87,
            validation_f1=0.85,
            validation_roc_auc=0.90,
            feature_set_version="1.0",
            created_by="test_user",
        )

        versioning.create_version("test_model", test_model, metadata1)
        versioning.create_version("test_model", test_model, metadata2, VersionType.MAJOR)

        # Compare versions
        comparison = versioning.compare_versions("test_model", "1.0.0", "2.0.0")

        assert comparison.accuracy_improvement == 0.05
        assert comparison.precision_improvement == 0.05
        assert comparison.recall_improvement == 0.05
        assert comparison.f1_improvement == 0.05

    def test_rollback(self, versioning, test_model, test_metadata):
        """Test model rollback"""
        # Create and promote to production
        version1 = versioning.create_version(
            model_id="test_model", model_object=test_model, metadata=test_metadata
        )

        # Promote through stages to production
        versioning.promote_version("test_model", version1, ModelStage.TESTING, "test")
        versioning.promote_version("test_model", version1, ModelStage.STAGING, "test")
        versioning.promote_version("test_model", version1, ModelStage.SHADOW, "test")
        versioning.promote_version("test_model", version1, ModelStage.CANDIDATE, "test")
        versioning.promote_version("test_model", version1, ModelStage.PRODUCTION, "test")

        # Create second version
        test_metadata.version = "2.0.0"
        version2 = versioning.create_version(
            model_id="test_model",
            model_object=test_model,
            metadata=test_metadata,
            version_type=VersionType.MAJOR,
        )

        # Promote to production
        versioning.promote_version("test_model", version2, ModelStage.TESTING, "test")
        versioning.promote_version("test_model", version2, ModelStage.STAGING, "test")
        versioning.promote_version("test_model", version2, ModelStage.SHADOW, "test")
        versioning.promote_version("test_model", version2, ModelStage.CANDIDATE, "test")
        versioning.promote_version("test_model", version2, ModelStage.PRODUCTION, "test")

        # Rollback to version1
        success = versioning.rollback_to_version(
            model_id="test_model",
            target_version=version1,
            rollback_by="test_user",
            reason="Performance degradation",
        )

        assert success
        assert versioning.get_production_version("test_model") == version1

    def test_archival_and_cleanup(self, versioning, test_model, test_metadata):
        """Test version archival and cleanup"""
        # Create multiple versions
        versions = []
        for i in range(7):  # More than max_versions_per_model (5)
            test_metadata.version = f"{i+1}.0.0"
            test_metadata.created_at = datetime.now() - timedelta(days=i * 10)
            version = versioning.create_version(
                model_id="test_model",
                model_object=test_model,
                metadata=test_metadata,
                custom_version=f"{i+1}.0.0",
            )
            versions.append(version)

        # Cleanup old versions
        cleanup_stats = versioning.cleanup_old_versions("test_model")

        assert cleanup_stats["archived"] > 0

    def test_list_operations(self, versioning, test_model, test_metadata):
        """Test listing operations"""
        # Create versions
        versioning.create_version("test_model", test_model, test_metadata)

        test_metadata.model_id = "another_model"
        versioning.create_version("another_model", test_model, test_metadata)

        # Test listing
        models = versioning.list_models()
        assert "test_model" in models
        assert "another_model" in models

        versions = versioning.list_versions("test_model")
        assert len(versions) == 1

        latest = versioning.get_latest_version("test_model")
        assert latest is not None


@pytest.fixture
def sample_training_data():
    """Create sample training data"""
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1) > 0

    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    data["target"] = y.astype(int)
    data["timestamp"] = pd.date_range(start="2024-01-01", periods=n_samples, freq="1min")

    return data


def test_integration_auto_retraining_with_scheduler():
    """Integration test combining auto-retraining with scheduler"""
    config = RetrainingConfig(
        require_manual_approval=False,
        cooldown_period_hours=0,  # No cooldown for testing
        max_retrainings_per_day=10,
    )

    mock_pipeline = Mock()
    mock_pipeline.train_and_validate_model.return_value = Mock(
        accuracy=0.65, precision=0.63, recall=0.67, f1_score=0.65, roc_auc=0.70
    )

    # Create system
    system = AutoRetrainingSystem(config=config, ml_pipeline=mock_pipeline, db_manager=Mock())

    # Create scheduler
    scheduler = RetrainingScheduler()

    # Add scheduled retraining
    schedule_config = ScheduleConfig(
        task_id="test_retraining",
        schedule_type=ScheduleType.MANUAL,  # For immediate execution
        name="Test Retraining",
        description="Test scheduled retraining",
    )

    def retraining_callback():
        return system.request_manual_retraining("Scheduled retraining")

    scheduler.add_schedule(schedule_config, retraining_callback)

    # Trigger immediate execution
    success = scheduler.trigger_immediate("test_retraining")
    assert success


if __name__ == "__main__":
    pytest.main([__file__])
