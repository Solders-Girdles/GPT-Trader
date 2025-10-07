"""
Tier Promotion Policy for State Management

Encapsulates tier promotion and demotion decisions based on
access patterns and storage tier characteristics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot_v2.state.repositories import (
        PostgresStateRepository,
        RedisStateRepository,
        S3StateRepository,
    )
    from bot_v2.state.state_manager import StateCategory

logger = logging.getLogger(__name__)


class TierPromotionPolicy:
    """
    Manages tier promotion and demotion decisions for state data.

    Encapsulates the logic for moving data between HOT, WARM, and COLD
    tiers based on access patterns and manual interventions.
    """

    def __init__(
        self,
        redis_repo: RedisStateRepository | None,
        postgres_repo: PostgresStateRepository | None,
        s3_repo: S3StateRepository | None,
    ) -> None:
        """
        Initialize promotion policy.

        Args:
            redis_repo: Repository for HOT tier (Redis)
            postgres_repo: Repository for WARM tier (PostgreSQL)
            s3_repo: Repository for COLD tier (S3)
        """
        self._redis_repo = redis_repo
        self._postgres_repo = postgres_repo
        self._s3_repo = s3_repo

    def should_auto_promote(self, from_tier: StateCategory, auto_promote: bool) -> bool:
        """
        Determine if state should be automatically promoted on access.

        Args:
            from_tier: Current storage tier
            auto_promote: Auto-promotion flag from caller

        Returns:
            True if state should be promoted, False otherwise
        """
        from bot_v2.state.state_manager import StateCategory

        if not auto_promote:
            return False

        # Only auto-promote from WARM and COLD tiers
        return from_tier in (StateCategory.WARM, StateCategory.COLD)

    def get_promotion_target(self, from_tier: StateCategory) -> StateCategory:
        """
        Get target tier for promotion.

        Args:
            from_tier: Current storage tier

        Returns:
            Target tier for promotion
        """
        from bot_v2.state.state_manager import StateCategory

        # COLD → WARM
        if from_tier == StateCategory.COLD:
            return StateCategory.WARM

        # WARM → HOT
        if from_tier == StateCategory.WARM:
            return StateCategory.HOT

        # Already at HOT tier
        return StateCategory.HOT

    async def promote_value(
        self, key: str, value: str, from_tier: StateCategory, metadata: dict[str, Any]
    ) -> bool:
        """
        Promote value to higher tier.

        Args:
            key: State key
            value: Serialized state value
            from_tier: Current storage tier
            metadata: Metadata for storage (checksum, ttl_seconds, etc.)

        Returns:
            True if promotion succeeded, False otherwise
        """
        from bot_v2.state.state_manager import StateCategory

        target_tier = self.get_promotion_target(from_tier)

        # Promote to target tier
        if target_tier == StateCategory.HOT and self._redis_repo:
            return await self._redis_repo.store(key, value, metadata)
        elif target_tier == StateCategory.WARM and self._postgres_repo:
            return await self._postgres_repo.store(key, value, metadata)

        return False

    async def demote_to_cold(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """
        Demote value to cold tier, removing from higher tiers.

        Args:
            key: State key
            value: Serialized state value
            metadata: Metadata for storage

        Returns:
            True if demotion succeeded, False otherwise
        """
        # Delete from hot tier
        if self._redis_repo:
            await self._redis_repo.delete(key)

        # Delete from warm tier
        if self._postgres_repo:
            await self._postgres_repo.delete(key)

        # Store in cold tier
        if self._s3_repo:
            return await self._s3_repo.store(key, value, metadata)

        return False

    async def promote_to_hot(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """
        Manually promote value to hot tier.

        Args:
            key: State key
            value: Serialized state value
            metadata: Metadata for storage

        Returns:
            True if promotion succeeded, False otherwise
        """
        if self._redis_repo:
            return await self._redis_repo.store(key, value, metadata)
        return False
