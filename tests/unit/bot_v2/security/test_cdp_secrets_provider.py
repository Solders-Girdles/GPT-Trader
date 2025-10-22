"""Tests for CDP Secrets Provider"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from bot_v2.security.cdp_secrets_provider import (
    CDPCredentials,
    CDPJWTToken,
    CDPSecretsProvider,
)


@pytest.fixture
def mock_secrets_manager():
    """Mock SecretsManager"""
    manager = Mock()
    manager.store_secret = Mock(return_value=True)
    manager.get_secret = Mock(return_value=None)
    manager.list_secrets = Mock(return_value=[])
    return manager


@pytest.fixture
def provider(mock_secrets_manager):
    """CDP Secrets Provider instance"""
    return CDPSecretsProvider(secrets_manager=mock_secrets_manager)


@pytest.fixture
def sample_credentials():
    """Sample CDP credentials"""
    return {
        "api_key_name": "organizations/test-org/apiKeys/test-key",
        "private_key_pem": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        "allowed_ips": ["192.168.1.1", "10.0.0.0/24"],
    }


def test_store_cdp_credentials(provider, mock_secrets_manager, sample_credentials):
    """Test storing CDP credentials"""
    success = provider.store_cdp_credentials(
        "coinbase_production",
        sample_credentials["api_key_name"],
        sample_credentials["private_key_pem"],
        allowed_ips=sample_credentials["allowed_ips"],
    )

    assert success
    assert mock_secrets_manager.store_secret.called
    call_args = mock_secrets_manager.store_secret.call_args
    assert call_args[0][0] == "cdp_credentials/coinbase_production"
    assert call_args[0][1]["api_key_name"] == sample_credentials["api_key_name"]


def test_get_cdp_credentials(provider, mock_secrets_manager, sample_credentials):
    """Test retrieving CDP credentials"""
    now = datetime.now(UTC)
    mock_secrets_manager.get_secret.return_value = {
        "api_key_name": sample_credentials["api_key_name"],
        "private_key_pem": sample_credentials["private_key_pem"],
        "created_at": now.isoformat(),
        "last_rotated_at": None,
        "rotation_policy_days": 90,
        "allowed_ips": sample_credentials["allowed_ips"],
    }

    credentials = provider.get_cdp_credentials("coinbase_production")

    assert credentials is not None
    assert credentials.api_key_name == sample_credentials["api_key_name"]
    assert credentials.private_key_pem == sample_credentials["private_key_pem"]
    assert credentials.allowed_ips == sample_credentials["allowed_ips"]


@patch("bot_v2.security.cdp_secrets_provider.create_cdp_jwt_auth")
def test_generate_short_lived_jwt(mock_create_auth, provider, mock_secrets_manager, sample_credentials):
    """Test generating short-lived JWT"""
    now = datetime.now(UTC)
    mock_secrets_manager.get_secret.return_value = {
        "api_key_name": sample_credentials["api_key_name"],
        "private_key_pem": sample_credentials["private_key_pem"],
        "created_at": now.isoformat(),
        "last_rotated_at": None,
        "rotation_policy_days": 90,
        "allowed_ips": ["192.168.1.1"],
    }

    # Mock auth instance
    mock_auth = Mock()
    mock_auth.generate_jwt.return_value = "test-jwt-token"
    mock_create_auth.return_value = mock_auth

    token = provider.generate_short_lived_jwt(
        "coinbase_production",
        "GET",
        "/api/v3/brokerage/accounts",
        client_ip="192.168.1.1",
    )

    assert token is not None
    assert token.token == "test-jwt-token"
    assert token.method == "GET"
    assert token.path == "/api/v3/brokerage/accounts"
    assert not token.is_expired
    assert token.seconds_until_expiry <= 120


def test_generate_jwt_ip_rejected(provider, mock_secrets_manager, sample_credentials):
    """Test JWT generation rejected by IP allowlist"""
    now = datetime.now(UTC)
    mock_secrets_manager.get_secret.return_value = {
        "api_key_name": sample_credentials["api_key_name"],
        "private_key_pem": sample_credentials["private_key_pem"],
        "created_at": now.isoformat(),
        "last_rotated_at": None,
        "rotation_policy_days": 90,
        "allowed_ips": ["192.168.1.1"],
    }

    token = provider.generate_short_lived_jwt(
        "coinbase_production",
        "GET",
        "/api/v3/brokerage/accounts",
        client_ip="10.0.0.1",  # Not in allowlist
    )

    assert token is None


def test_validate_ip_allowlist_exact_match(provider):
    """Test IP validation with exact match"""
    assert provider._validate_ip_allowlist("192.168.1.1", ["192.168.1.1", "10.0.0.1"])


def test_validate_ip_allowlist_cidr(provider):
    """Test IP validation with CIDR notation"""
    assert provider._validate_ip_allowlist("192.168.1.50", ["192.168.1.0/24"])
    assert not provider._validate_ip_allowlist("192.168.2.1", ["192.168.1.0/24"])


def test_rotate_credentials(provider, mock_secrets_manager, sample_credentials):
    """Test credential rotation"""
    now = datetime.now(UTC)
    mock_secrets_manager.get_secret.return_value = {
        "api_key_name": sample_credentials["api_key_name"],
        "private_key_pem": sample_credentials["private_key_pem"],
        "created_at": now.isoformat(),
        "last_rotated_at": None,
        "rotation_policy_days": 90,
        "allowed_ips": sample_credentials["allowed_ips"],
    }

    new_key = "organizations/test-org/apiKeys/new-key"
    new_pem = "-----BEGIN EC PRIVATE KEY-----\nnew\n-----END EC PRIVATE KEY-----"

    success = provider.rotate_credentials("coinbase_production", new_key, new_pem)

    assert success
    assert mock_secrets_manager.store_secret.called
    call_args = mock_secrets_manager.store_secret.call_args
    assert call_args[0][1]["api_key_name"] == new_key
    assert call_args[0][1]["last_rotated_at"] is not None


def test_needs_rotation(provider):
    """Test rotation check"""
    now = datetime.now(UTC)

    # Recent credential - no rotation needed
    recent_cred = CDPCredentials(
        api_key_name="test",
        private_key_pem="test",
        created_at=now - timedelta(days=30),
        last_rotated_at=None,
        rotation_policy_days=90,
    )
    assert not provider._needs_rotation(recent_cred)

    # Old credential - rotation needed
    old_cred = CDPCredentials(
        api_key_name="test",
        private_key_pem="test",
        created_at=now - timedelta(days=100),
        last_rotated_at=None,
        rotation_policy_days=90,
    )
    assert provider._needs_rotation(old_cred)


def test_update_ip_allowlist(provider, mock_secrets_manager, sample_credentials):
    """Test updating IP allowlist"""
    now = datetime.now(UTC)
    mock_secrets_manager.get_secret.return_value = {
        "api_key_name": sample_credentials["api_key_name"],
        "private_key_pem": sample_credentials["private_key_pem"],
        "created_at": now.isoformat(),
        "last_rotated_at": None,
        "rotation_policy_days": 90,
        "allowed_ips": ["192.168.1.1"],
    }

    new_ips = ["192.168.1.1", "192.168.1.2", "10.0.0.0/24"]
    success = provider.update_ip_allowlist("coinbase_production", new_ips)

    assert success
    assert mock_secrets_manager.store_secret.called
    call_args = mock_secrets_manager.store_secret.call_args
    assert call_args[0][1]["allowed_ips"] == new_ips


def test_jwt_token_expiration():
    """Test JWT token expiration tracking"""
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=120)

    token = CDPJWTToken(
        token="test-token",
        created_at=now,
        expires_at=expires_at,
        method="GET",
        path="/test",
    )

    assert not token.is_expired
    assert 119 <= token.seconds_until_expiry <= 120

    # Simulate expired token
    expired_token = CDPJWTToken(
        token="test-token",
        created_at=now - timedelta(seconds=130),
        expires_at=now - timedelta(seconds=10),
        method="GET",
        path="/test",
    )

    assert expired_token.is_expired
    assert expired_token.seconds_until_expiry == 0
