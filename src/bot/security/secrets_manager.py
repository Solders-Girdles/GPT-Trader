"""Secure secrets management for GPT-Trader.

This module provides secure handling of API keys and sensitive configuration
without storing them in plain text files.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SecretConfig:
    """Configuration for required secrets."""

    required_secrets: list[str] = field(
        default_factory=lambda: [
            "ALPACA_API_KEY_ID",
            "ALPACA_API_SECRET_KEY",
        ]
    )

    optional_secrets: list[str] = field(
        default_factory=lambda: [
            "ALPACA_PAPER_BASE_URL",
            "LOG_LEVEL",
        ]
    )

    defaults: dict[str, str] = field(
        default_factory=lambda: {
            "ALPACA_PAPER_BASE_URL": "https://paper-api.alpaca.markets",
            "LOG_LEVEL": "INFO",
        }
    )


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class SecretManager:
    """Manages secure access to secrets and sensitive configuration.

    This class provides:
    - Validation of required environment variables at startup
    - Secure storage recommendations
    - Default values for optional configuration
    - Centralized secret access with logging
    """

    def __init__(self, config: SecretConfig | None = None) -> None:
        """Initialize the SecretManager.

        Args:
            config: Optional configuration for secrets. Uses defaults if not provided.
        """
        self.config = config or SecretConfig()
        self._secrets_cache: dict[str, str] = {}
        self._validated = False

    def validate_startup_secrets(self, raise_on_missing: bool = True) -> dict[str, bool]:
        """Validate that all required secrets are available at startup.

        Args:
            raise_on_missing: If True, raises ConfigurationError when required secrets are missing.

        Returns:
            Dictionary mapping secret names to their availability status.

        Raises:
            ConfigurationError: If required secrets are missing and raise_on_missing is True.
        """
        validation_results = {}
        missing_required = []

        # Check required secrets
        for secret_name in self.config.required_secrets:
            value = os.getenv(secret_name)
            if value and value.strip():
                validation_results[secret_name] = True
                logger.debug(f"Found required secret: {secret_name}")
            else:
                validation_results[secret_name] = False
                missing_required.append(secret_name)
                logger.warning(f"Missing required secret: {secret_name}")

        # Check optional secrets
        for secret_name in self.config.optional_secrets:
            value = os.getenv(secret_name)
            if value and value.strip():
                validation_results[secret_name] = True
                logger.debug(f"Found optional secret: {secret_name}")
            else:
                validation_results[secret_name] = False
                logger.info(f"Optional secret not set (will use default): {secret_name}")

        if missing_required and raise_on_missing:
            error_msg = (
                f"Missing required secrets: {', '.join(missing_required)}\n"
                f"Please set these environment variables before running the application.\n"
                f"You can use one of the following methods:\n"
                f"  1. Export in your shell: export ALPACA_API_KEY_ID='your-key'\n"
                f"  2. Use a .env.local file (not tracked by git)\n"
                f"  3. Use a secure secrets manager (recommended for production)"
            )
            raise ConfigurationError(error_msg)

        self._validated = True
        return validation_results

    def get_secret(self, secret_name: str, default: str | None = None) -> str | None:
        """Get a secret value securely.

        Args:
            secret_name: Name of the secret to retrieve.
            default: Default value if secret is not found.

        Returns:
            The secret value or default if not found.
        """
        # Check cache first
        if secret_name in self._secrets_cache:
            return self._secrets_cache[secret_name]

        # Try environment variable
        value = os.getenv(secret_name)

        # Use configured default if available
        if not value and secret_name in self.config.defaults:
            value = self.config.defaults[secret_name]

        # Use provided default as last resort
        if not value:
            value = default

        # Cache the value
        if value:
            self._secrets_cache[secret_name] = value

        return value

    def get_alpaca_credentials(self) -> dict[str, str]:
        """Get Alpaca API credentials.

        Returns:
            Dictionary with Alpaca API credentials.

        Raises:
            ConfigurationError: If credentials are not available.
        """
        if not self._validated:
            self.validate_startup_secrets()

        return {
            "api_key_id": self.get_secret("ALPACA_API_KEY_ID"),
            "api_secret_key": self.get_secret("ALPACA_API_SECRET_KEY"),
            "base_url": self.get_secret("ALPACA_PAPER_BASE_URL"),
        }

    def clear_cache(self) -> None:
        """Clear the internal secrets cache.

        Use this when you need to reload secrets from environment.
        """
        self._secrets_cache.clear()
        self._validated = False
        logger.info("Cleared secrets cache")

    @staticmethod
    def create_env_template() -> str:
        """Create a template .env file content.

        Returns:
            String content for .env.template file.
        """
        template = """# GPT-Trader Environment Configuration Template
# Copy this file to .env.local and fill in your actual values
# NEVER commit .env.local or any file with real secrets to git

# Required: Alpaca API Credentials
ALPACA_API_KEY_ID=your-api-key-id-here
ALPACA_API_SECRET_KEY=your-api-secret-key-here

# Optional: API Configuration
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets

# Optional: Logging Configuration
LOG_LEVEL=INFO
"""
        return template

    @staticmethod
    def setup_secure_environment(project_root: Path) -> None:
        """Set up secure environment configuration.

        This method:
        1. Creates .env.template if it doesn't exist
        2. Updates .gitignore to exclude sensitive files
        3. Provides instructions for secure setup

        Args:
            project_root: Path to the project root directory.
        """
        # Create .env.template
        template_path = project_root / ".env.template"
        if not template_path.exists():
            template_path.write_text(SecretManager.create_env_template())
            logger.info(f"Created .env.template at {template_path}")

        # Update .gitignore
        gitignore_path = project_root / ".gitignore"
        gitignore_content = gitignore_path.read_text() if gitignore_path.exists() else ""

        security_section = """
# Security - NEVER remove these entries
.env
.env.*
!.env.template
*.pem
*.key
*.crt
*.p12
secrets/
credentials/
"""

        if ".env.template" not in gitignore_content:
            with open(gitignore_path, "a") as f:
                f.write(security_section)
            logger.info("Updated .gitignore with security exclusions")

        print("\n" + "=" * 60)
        print("SECURE ENVIRONMENT SETUP COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Copy .env.template to .env.local")
        print("2. Fill in your actual API credentials in .env.local")
        print("3. Load .env.local in your shell or IDE")
        print("\nSecurity reminders:")
        print("- NEVER commit .env.local or real secrets to git")
        print("- Use environment variables or secure vaults in production")
        print("- Rotate API keys regularly")
        print("=" * 60 + "\n")


# Singleton instance for application-wide use
_secret_manager: SecretManager | None = None


def get_secret_manager() -> SecretManager:
    """Get the singleton SecretManager instance.

    Returns:
        The global SecretManager instance.
    """
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager
