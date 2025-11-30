"""
Security infrastructure for GPT-Trader.

This module provides security utilities for credential management and API authentication.

Authentication
--------------
API credentials are managed through the ``SimpleAuth`` class in the brokerages module.
Credentials are loaded from environment variables or configuration files.

Environment Variables
---------------------
- ``COINBASE_API_KEY``: Coinbase Advanced Trade API key
- ``COINBASE_API_SECRET``: Coinbase API secret (base64 encoded)
- ``COINBASE_PASSPHRASE``: Optional passphrase for legacy API

Security Best Practices
-----------------------
1. Never commit credentials to version control
2. Use ``.env`` files for local development (excluded via ``.gitignore``)
3. Use secure secret managers in production (AWS Secrets Manager, HashiCorp Vault)
4. Rotate API keys periodically
5. Use read-only API keys for backtesting/research

See Also
--------
- ``gpt_trader.features.brokerages.coinbase.auth``: Coinbase-specific authentication
- ``gpt_trader.orchestration.configuration``: Configuration loading
"""

# Authentication is handled via SimpleAuth in the brokerages module
