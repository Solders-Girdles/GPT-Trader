"""Shared types for configuration."""

from enum import Enum


class Profile(Enum):
    """Configuration profiles."""

    DEV = "dev"
    DEMO = "demo"
    PROD = "prod"
    CANARY = "canary"
    SPOT = "spot"
