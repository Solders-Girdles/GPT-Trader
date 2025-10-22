"""Tests for Two-Person Rule"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from bot_v2.monitoring.two_person_rule import (
    ApprovalRequest,
    ApprovalStatus,
    ChangeType,
    ConfigChange,
    TwoPersonRule,
)
from bot_v2.security.auth_handler import Role, User


@pytest.fixture
def requester():
    """Sample requester user"""
    return User(
        id="user-001",
        username="trader1",
        email="trader1@example.com",
        role=Role.TRADER,
    )


@pytest.fixture
def approver():
    """Sample approver user"""
    return User(
        id="user-002",
        username="admin1",
        email="admin1@example.com",
        role=Role.ADMIN,
    )


@pytest.fixture
def event_store():
    """Mock event store"""
    store = Mock()
    store.append_metric = Mock()
    return store


@pytest.fixture
def two_person_rule(event_store):
    """Two-person rule instance"""
    return TwoPersonRule(event_store=event_store)


@pytest.fixture
def sample_changes():
    """Sample configuration changes"""
    return [
        ConfigChange(
            change_type=ChangeType.LEVERAGE,
            field_name="max_leverage",
            old_value=3,
            new_value=5,
            description="Increase max leverage to 5x",
        ),
        ConfigChange(
            change_type=ChangeType.RISK_LIMIT,
            field_name="daily_loss_limit",
            old_value=100,
            new_value=200,
            description="Increase daily loss limit",
        ),
    ]


def test_create_approval_request(two_person_rule, requester, sample_changes):
    """Test creating approval request"""
    request = two_person_rule.create_approval_request(
        requester,
        sample_changes,
        metadata={"reason": "Production configuration update"},
    )

    assert request.requester_id == requester.id
    assert request.requester_name == requester.username
    assert request.status == ApprovalStatus.PENDING
    assert len(request.changes) == 2
    assert request.requires_approval
    assert not request.is_expired


def test_approve_request(two_person_rule, requester, approver, sample_changes):
    """Test approving request"""
    request = two_person_rule.create_approval_request(requester, sample_changes)

    success, error = two_person_rule.approve_request(request.request_id, approver)

    assert success
    assert error is None

    updated_request = two_person_rule.get_request(request.request_id)
    assert updated_request.status == ApprovalStatus.APPROVED
    assert updated_request.approver_id == approver.id
    assert updated_request.approver_name == approver.username
    assert updated_request.approved_at is not None


def test_approve_request_same_user(two_person_rule, requester, sample_changes):
    """Test rejection when approver is same as requester"""
    request = two_person_rule.create_approval_request(requester, sample_changes)

    success, error = two_person_rule.approve_request(request.request_id, requester)

    assert not success
    assert "two-person rule" in error.lower()


def test_approve_expired_request(two_person_rule, requester, approver, sample_changes):
    """Test approving expired request"""
    # Create rule with short timeout
    short_timeout_rule = TwoPersonRule(approval_timeout_hours=0)
    request = short_timeout_rule.create_approval_request(requester, sample_changes)

    # Wait for expiration (simulate by manipulating expires_at)
    request.expires_at = datetime.now(UTC) - timedelta(hours=1)

    success, error = short_timeout_rule.approve_request(request.request_id, approver)

    assert not success
    assert "expired" in error.lower()


def test_reject_request(two_person_rule, requester, approver, sample_changes):
    """Test rejecting request"""
    request = two_person_rule.create_approval_request(requester, sample_changes)

    success, error = two_person_rule.reject_request(
        request.request_id,
        approver,
        "Risk parameters too aggressive",
    )

    assert success
    assert error is None

    # Request should be moved to history
    pending_requests = two_person_rule.get_pending_requests()
    assert request.request_id not in [r.request_id for r in pending_requests]

    # Check history
    history = two_person_rule.get_request_history()
    assert any(r.request_id == request.request_id for r in history)

    # Verify rejection details
    rejected = next(r for r in history if r.request_id == request.request_id)
    assert rejected.status == ApprovalStatus.REJECTED
    assert rejected.rejection_reason == "Risk parameters too aggressive"


def test_mark_applied(two_person_rule, requester, approver, sample_changes):
    """Test marking approved request as applied"""
    request = two_person_rule.create_approval_request(requester, sample_changes)
    two_person_rule.approve_request(request.request_id, approver)

    success, error = two_person_rule.mark_applied(request.request_id)

    assert success
    assert error is None

    # Request should be moved to history
    pending_requests = two_person_rule.get_pending_requests()
    assert request.request_id not in [r.request_id for r in pending_requests]

    # Check history
    history = two_person_rule.get_request_history()
    applied = next(r for r in history if r.request_id == request.request_id)
    assert applied.status == ApprovalStatus.APPLIED
    assert applied.applied_at is not None


def test_mark_applied_not_approved(two_person_rule, requester, sample_changes):
    """Test marking unapproved request as applied"""
    request = two_person_rule.create_approval_request(requester, sample_changes)

    success, error = two_person_rule.mark_applied(request.request_id)

    assert not success
    assert "approved" in error.lower()


def test_get_pending_requests(two_person_rule, requester, sample_changes):
    """Test getting pending requests"""
    # Create multiple requests
    request1 = two_person_rule.create_approval_request(requester, sample_changes)
    request2 = two_person_rule.create_approval_request(
        requester,
        [sample_changes[0]],
    )

    pending = two_person_rule.get_pending_requests()

    assert len(pending) == 2
    request_ids = {r.request_id for r in pending}
    assert request1.request_id in request_ids
    assert request2.request_id in request_ids


def test_requires_approval(two_person_rule):
    """Test checking which fields require approval"""
    changes = {
        "max_leverage": 5,
        "daily_loss_limit": 200,
        "some_other_field": "value",
    }

    required_fields = two_person_rule.requires_approval(changes)

    assert "max_leverage" in required_fields
    assert "daily_loss_limit" in required_fields
    assert "some_other_field" not in required_fields


def test_log_config_delta(two_person_rule, event_store):
    """Test logging configuration delta"""
    changes = {
        "max_leverage": (3, 5),
        "daily_loss_limit": (100, 200),
    }

    two_person_rule.log_config_delta(
        "risk_limit_update",
        changes,
        user_id="user-001",
        metadata={"reason": "Production update"},
    )

    assert event_store.append_metric.called
    call_args = event_store.append_metric.call_args
    assert call_args[0][0] == "config_guardian"

    event_data = call_args[0][1]
    assert event_data["event_type"] == "config_delta"
    assert event_data["change_type"] == "risk_limit_update"
    assert "max_leverage" in event_data["changes"]


def test_request_expiration():
    """Test request expiration"""
    now = datetime.now(UTC)
    expires_at = now + timedelta(hours=24)

    request = ApprovalRequest(
        request_id="test-001",
        requester_id="user-001",
        requester_name="trader1",
        changes=[],
        status=ApprovalStatus.PENDING,
        created_at=now,
        expires_at=expires_at,
    )

    assert not request.is_expired
    assert request.requires_approval

    # Simulate expiration
    request.expires_at = now - timedelta(hours=1)
    assert request.is_expired
    assert request.requires_approval  # Still true until status changes


def test_cleanup_expired_requests(two_person_rule, requester, sample_changes):
    """Test automatic cleanup of expired requests"""
    # Create request
    request = two_person_rule.create_approval_request(requester, sample_changes)

    # Manually expire it
    pending_request = two_person_rule.get_request(request.request_id)
    pending_request.expires_at = datetime.now(UTC) - timedelta(hours=1)

    # Trigger cleanup by getting pending requests
    pending = two_person_rule.get_pending_requests()

    # Request should not be in pending list
    assert request.request_id not in [r.request_id for r in pending]

    # Should be in history as expired
    history = two_person_rule.get_request_history()
    expired = next((r for r in history if r.request_id == request.request_id), None)
    assert expired is not None
    assert expired.status == ApprovalStatus.EXPIRED


def test_config_change_to_dict():
    """Test ConfigChange serialization"""
    change = ConfigChange(
        change_type=ChangeType.LEVERAGE,
        field_name="max_leverage",
        old_value=3,
        new_value=5,
        description="Test change",
    )

    data = change.to_dict()

    assert data["change_type"] == "leverage"
    assert data["field_name"] == "max_leverage"
    assert data["old_value"] == "3"
    assert data["new_value"] == "5"
    assert data["description"] == "Test change"


def test_approval_request_to_dict(requester, sample_changes):
    """Test ApprovalRequest serialization"""
    now = datetime.now(UTC)
    request = ApprovalRequest(
        request_id="test-001",
        requester_id=requester.id,
        requester_name=requester.username,
        changes=sample_changes,
        status=ApprovalStatus.PENDING,
        created_at=now,
        expires_at=now + timedelta(hours=24),
    )

    data = request.to_dict()

    assert data["request_id"] == "test-001"
    assert data["requester_id"] == requester.id
    assert data["status"] == "pending"
    assert len(data["changes"]) == 2
    assert data["approved_at"] is None


def test_critical_fields_coverage():
    """Test that all critical fields are defined"""
    critical_fields = TwoPersonRule.CRITICAL_FIELDS

    # Ensure important fields are covered
    assert "max_leverage" in critical_fields
    assert "max_position_size" in critical_fields
    assert "daily_loss_limit" in critical_fields
    assert "liquidation_buffer" in critical_fields
    assert "symbols" in critical_fields
    assert "profile" in critical_fields
