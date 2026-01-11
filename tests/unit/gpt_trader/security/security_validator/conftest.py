"""Shared fixtures for security_validator tests."""

pytest_plugins = [
    "tests.unit.gpt_trader.security.security_validator.core_fixtures",
    "tests.unit.gpt_trader.security.security_validator.time_fixtures",
    "tests.unit.gpt_trader.security.security_validator.sample_fixtures",
]
