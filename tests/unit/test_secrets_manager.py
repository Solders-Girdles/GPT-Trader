"""Unit tests for the SecretManager."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from bot.security.secrets_manager import (
    SecretManager,
    SecretConfig,
    ConfigurationError,
    get_secret_manager,
)


class TestSecretManager:
    """Test suite for SecretManager."""

    def test_initialization(self):
        """Test SecretManager initialization."""
        manager = SecretManager()
        assert manager.config is not None
        assert not manager._validated
        assert manager._secrets_cache == {}

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = SecretConfig(
            required_secrets=["TEST_KEY"],
            optional_secrets=["OPT_KEY"],
            defaults={"OPT_KEY": "default_value"},
        )
        manager = SecretManager(config=config)
        assert manager.config.required_secrets == ["TEST_KEY"]
        assert manager.config.optional_secrets == ["OPT_KEY"]
        assert manager.config.defaults["OPT_KEY"] == "default_value"

    @patch.dict(
        os.environ, {"ALPACA_API_KEY_ID": "test_key", "ALPACA_API_SECRET_KEY": "test_secret"}
    )
    def test_validate_startup_secrets_success(self):
        """Test successful validation of required secrets."""
        manager = SecretManager()
        results = manager.validate_startup_secrets(raise_on_missing=False)

        assert results["ALPACA_API_KEY_ID"] is True
        assert results["ALPACA_API_SECRET_KEY"] is True
        assert manager._validated is True

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_startup_secrets_missing(self):
        """Test validation with missing required secrets."""
        manager = SecretManager()

        # Should not raise when raise_on_missing=False
        results = manager.validate_startup_secrets(raise_on_missing=False)
        assert results["ALPACA_API_KEY_ID"] is False
        assert results["ALPACA_API_SECRET_KEY"] is False

        # Should raise when raise_on_missing=True
        manager._validated = False
        with pytest.raises(ConfigurationError) as exc_info:
            manager.validate_startup_secrets(raise_on_missing=True)

        assert "Missing required secrets" in str(exc_info.value)
        assert "ALPACA_API_KEY_ID" in str(exc_info.value)

    @patch.dict(os.environ, {"TEST_SECRET": "test_value"})
    def test_get_secret_from_env(self):
        """Test retrieving secret from environment."""
        manager = SecretManager()
        value = manager.get_secret("TEST_SECRET")
        assert value == "test_value"
        assert manager._secrets_cache["TEST_SECRET"] == "test_value"

    def test_get_secret_with_default(self):
        """Test retrieving secret with default value."""
        manager = SecretManager()
        value = manager.get_secret("NONEXISTENT_SECRET", default="default_value")
        assert value == "default_value"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_secret_from_config_defaults(self):
        """Test retrieving secret from config defaults."""
        manager = SecretManager()
        value = manager.get_secret("LOG_LEVEL")
        assert value == "INFO"  # From config defaults

    @patch.dict(
        os.environ,
        {
            "ALPACA_API_KEY_ID": "test_key_id",
            "ALPACA_API_SECRET_KEY": "test_secret_key",
            "ALPACA_PAPER_BASE_URL": "https://test.url",
        },
    )
    def test_get_alpaca_credentials(self):
        """Test retrieving Alpaca credentials."""
        manager = SecretManager()
        creds = manager.get_alpaca_credentials()

        assert creds["api_key_id"] == "test_key_id"
        assert creds["api_secret_key"] == "test_secret_key"
        assert creds["base_url"] == "https://test.url"

    def test_clear_cache(self):
        """Test clearing the secrets cache."""
        manager = SecretManager()
        manager._secrets_cache = {"TEST": "value"}
        manager._validated = True

        manager.clear_cache()

        assert manager._secrets_cache == {}
        assert manager._validated is False

    def test_create_env_template(self):
        """Test creating .env template content."""
        template = SecretManager.create_env_template()
        assert "ALPACA_API_KEY_ID=" in template
        assert "ALPACA_API_SECRET_KEY=" in template
        assert "NEVER commit" in template

    def test_setup_secure_environment(self, tmp_path):
        """Test setting up secure environment."""
        # Create a temporary project directory
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create existing .gitignore
        gitignore_path = project_dir / ".gitignore"
        gitignore_path.write_text("# Existing content\n*.pyc\n")

        # Run setup
        SecretManager.setup_secure_environment(project_dir)

        # Check .env.template was created
        template_path = project_dir / ".env.template"
        assert template_path.exists()
        assert "ALPACA_API_KEY_ID=" in template_path.read_text()

        # Check .gitignore was updated
        gitignore_content = gitignore_path.read_text()
        assert ".env" in gitignore_content
        assert "!.env.template" in gitignore_content
        assert "secrets/" in gitignore_content

    def test_singleton_instance(self):
        """Test that get_secret_manager returns singleton."""
        manager1 = get_secret_manager()
        manager2 = get_secret_manager()
        assert manager1 is manager2

    @patch.dict(os.environ, {"ALPACA_API_KEY_ID": "", "ALPACA_API_SECRET_KEY": "  "})
    def test_validate_empty_secrets(self):
        """Test validation treats empty/whitespace strings as missing."""
        manager = SecretManager()
        results = manager.validate_startup_secrets(raise_on_missing=False)

        assert results["ALPACA_API_KEY_ID"] is False  # Empty string
        assert results["ALPACA_API_SECRET_KEY"] is False  # Whitespace only

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_optional_secret_override(self):
        """Test optional secret can override default."""
        manager = SecretManager()
        value = manager.get_secret("LOG_LEVEL")
        assert value == "DEBUG"  # Environment overrides default


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_error_message(self):
        """Test error message formatting."""
        error = ConfigurationError("Test error message")
        assert str(error) == "Test error message"

    def test_error_inheritance(self):
        """Test ConfigurationError inherits from Exception."""
        error = ConfigurationError("Test")
        assert isinstance(error, Exception)
