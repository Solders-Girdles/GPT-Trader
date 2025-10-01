"""Tests for failure detection module"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from bot_v2.state.recovery.detection import FailureDetector
from bot_v2.state.recovery.models import FailureType


@pytest.fixture
def mock_state_manager():
    """Mock state manager"""
    manager = Mock()
    manager.redis_client = Mock()
    manager.pg_conn = MagicMock()
    manager.s3_client = Mock()
    manager.config = Mock()
    manager.config.s3_bucket = "test-bucket"
    return manager


@pytest.fixture
def mock_checkpoint_handler():
    """Mock checkpoint handler"""
    return Mock()


@pytest.fixture
def detector(mock_state_manager, mock_checkpoint_handler):
    """Create FailureDetector instance"""
    return FailureDetector(mock_state_manager, mock_checkpoint_handler)


class TestFailureDetector:
    """Test suite for FailureDetector"""

    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, detector):
        """Test successful Redis health check"""
        detector.state_manager.redis_client.ping.return_value = True

        result = await detector.test_redis_health()

        assert result is True
        detector.state_manager.redis_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_health_check_failure(self, detector):
        """Test Redis health check failure"""
        detector.state_manager.redis_client.ping.side_effect = Exception("Connection failed")

        result = await detector.test_redis_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_redis_health_check_no_client(self, detector):
        """Test Redis health check with no client"""
        detector.state_manager.redis_client = None

        result = await detector.test_redis_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_postgres_health_check_success(self, detector):
        """Test successful PostgreSQL health check"""
        mock_cursor = Mock()
        detector.state_manager.pg_conn.cursor.return_value.__enter__.return_value = mock_cursor

        result = await detector.test_postgres_health()

        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_postgres_health_check_failure(self, detector):
        """Test PostgreSQL health check failure"""
        detector.state_manager.pg_conn.cursor.side_effect = Exception("Connection failed")

        result = await detector.test_postgres_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_s3_health_check_success(self, detector):
        """Test successful S3 health check"""
        detector.state_manager.s3_client.head_bucket.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

        result = await detector.test_s3_health()

        assert result is True
        detector.state_manager.s3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @pytest.mark.asyncio
    async def test_s3_health_check_failure(self, detector):
        """Test S3 health check failure"""
        detector.state_manager.s3_client.head_bucket.side_effect = Exception("Bucket not found")

        result = await detector.test_s3_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_trading_engine_health_success(self, detector):
        """Test trading engine health check success"""
        detector.state_manager.get_state = AsyncMock(return_value="running")

        result = await detector.test_trading_engine_health()

        assert result is True
        detector.state_manager.get_state.assert_called_once_with("system:trading_engine_status")

    @pytest.mark.asyncio
    async def test_trading_engine_health_crashed(self, detector):
        """Test trading engine health check with crashed status"""
        detector.state_manager.get_state = AsyncMock(return_value="crashed")

        result = await detector.test_trading_engine_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_trading_engine_health_none(self, detector):
        """Test trading engine health check with None status"""
        detector.state_manager.get_state = AsyncMock(return_value=None)

        result = await detector.test_trading_engine_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_data_corruption_detection_no_corruption(self, detector):
        """Test data corruption detection with valid checksums"""
        import hashlib
        import json

        data = {"value": 123, "timestamp": "2025-01-01"}
        checksum = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        data["_checksum"] = checksum

        detector.state_manager.get_state = AsyncMock(return_value=data)

        result = await detector.detect_data_corruption()

        assert result is False

    @pytest.mark.asyncio
    async def test_data_corruption_detection_with_corruption(self, detector):
        """Test data corruption detection with invalid checksums"""
        data = {"value": 123, "_checksum": "invalid_checksum"}
        detector.state_manager.get_state = AsyncMock(return_value=data)

        result = await detector.detect_data_corruption()

        assert result is True

    @pytest.mark.asyncio
    async def test_data_corruption_detection_no_checksum(self, detector):
        """Test data corruption detection without checksum"""
        data = {"value": 123}
        detector.state_manager.get_state = AsyncMock(return_value=data)

        result = await detector.detect_data_corruption()

        assert result is False

    @pytest.mark.asyncio
    @patch('bot_v2.state.recovery.detection.psutil')
    async def test_memory_usage_check(self, mock_psutil, detector):
        """Test memory usage check"""
        mock_psutil.virtual_memory.return_value.percent = 85.5

        result = await detector.check_memory_usage()

        assert result == 85.5

    @pytest.mark.asyncio
    async def test_memory_usage_check_no_psutil(self, detector):
        """Test memory usage check without psutil"""
        with patch('bot_v2.state.recovery.detection.psutil', side_effect=ImportError):
            result = await detector.check_memory_usage()

        assert result == 0

    @pytest.mark.asyncio
    @patch('bot_v2.state.recovery.detection.psutil')
    async def test_disk_usage_check(self, mock_psutil, detector):
        """Test disk usage check"""
        mock_psutil.disk_usage.return_value.percent = 92.3

        result = await detector.check_disk_usage()

        assert result == 92.3
        mock_psutil.disk_usage.assert_called_once_with("/")

    @pytest.mark.asyncio
    async def test_detect_failures_all_healthy(self, detector):
        """Test failure detection with all systems healthy"""
        detector.test_redis_health = AsyncMock(return_value=True)
        detector.test_postgres_health = AsyncMock(return_value=True)
        detector.test_s3_health = AsyncMock(return_value=True)
        detector.test_trading_engine_health = AsyncMock(return_value=True)
        detector.detect_data_corruption = AsyncMock(return_value=False)
        detector.check_memory_usage = AsyncMock(return_value=50)
        detector.check_disk_usage = AsyncMock(return_value=60)

        failures = await detector.detect_failures()

        assert failures == []

    @pytest.mark.asyncio
    async def test_detect_failures_redis_down(self, detector):
        """Test failure detection with Redis down"""
        detector.test_redis_health = AsyncMock(return_value=False)
        detector.test_postgres_health = AsyncMock(return_value=True)
        detector.test_s3_health = AsyncMock(return_value=True)
        detector.test_trading_engine_health = AsyncMock(return_value=True)
        detector.detect_data_corruption = AsyncMock(return_value=False)
        detector.check_memory_usage = AsyncMock(return_value=50)
        detector.check_disk_usage = AsyncMock(return_value=60)

        failures = await detector.detect_failures()

        assert FailureType.REDIS_DOWN in failures

    @pytest.mark.asyncio
    async def test_detect_failures_multiple_issues(self, detector):
        """Test failure detection with multiple issues"""
        detector.test_redis_health = AsyncMock(return_value=False)
        detector.test_postgres_health = AsyncMock(return_value=False)
        detector.test_s3_health = AsyncMock(return_value=True)
        detector.test_trading_engine_health = AsyncMock(return_value=False)
        detector.detect_data_corruption = AsyncMock(return_value=True)
        detector.check_memory_usage = AsyncMock(return_value=95)
        detector.check_disk_usage = AsyncMock(return_value=98)

        failures = await detector.detect_failures()

        assert FailureType.REDIS_DOWN in failures
        assert FailureType.POSTGRES_DOWN in failures
        assert FailureType.TRADING_ENGINE_CRASH in failures
        assert FailureType.DATA_CORRUPTION in failures
        assert FailureType.MEMORY_OVERFLOW in failures
        assert FailureType.DISK_FULL in failures
        assert len(failures) == 6

    @pytest.mark.asyncio
    async def test_detect_failures_memory_threshold(self, detector):
        """Test memory overflow detection at threshold"""
        detector.test_redis_health = AsyncMock(return_value=True)
        detector.test_postgres_health = AsyncMock(return_value=True)
        detector.test_s3_health = AsyncMock(return_value=True)
        detector.test_trading_engine_health = AsyncMock(return_value=True)
        detector.detect_data_corruption = AsyncMock(return_value=False)
        detector.check_memory_usage = AsyncMock(return_value=91)
        detector.check_disk_usage = AsyncMock(return_value=60)

        failures = await detector.detect_failures()

        assert FailureType.MEMORY_OVERFLOW in failures

    @pytest.mark.asyncio
    async def test_detect_failures_disk_threshold(self, detector):
        """Test disk full detection at threshold"""
        detector.test_redis_health = AsyncMock(return_value=True)
        detector.test_postgres_health = AsyncMock(return_value=True)
        detector.test_s3_health = AsyncMock(return_value=True)
        detector.test_trading_engine_health = AsyncMock(return_value=True)
        detector.detect_data_corruption = AsyncMock(return_value=False)
        detector.check_memory_usage = AsyncMock(return_value=50)
        detector.check_disk_usage = AsyncMock(return_value=96)

        failures = await detector.detect_failures()

        assert FailureType.DISK_FULL in failures
