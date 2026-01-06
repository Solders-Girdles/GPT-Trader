"""Tests for credential validation caching functionality."""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_trader.tui.services.preferences_service import PreferencesService


class TestCredentialCache:
    """Tests for credential caching in PreferencesService."""

    @pytest.fixture
    def temp_prefs_path(self) -> Path:
        """Create a temporary preferences file path."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    def prefs_service(self, temp_prefs_path: Path) -> PreferencesService:
        """Create a PreferencesService with a temp file."""
        return PreferencesService(preferences_path=temp_prefs_path)

    def test_set_and_get_credential_cache(self, prefs_service: PreferencesService) -> None:
        """Test setting and retrieving credential cache."""
        fingerprint = "orgsfake...keyfake"
        validation_modes = {"paper": True, "live": True}

        prefs_service.set_credential_cache(fingerprint, validation_modes)
        cache = prefs_service.get_credential_cache()

        assert cache["fingerprint"] == fingerprint
        assert cache["validation_modes"] == validation_modes
        assert cache["validated_at"] is not None
        assert cache["validated_at"] > 0

    def test_invalidate_credential_cache(self, prefs_service: PreferencesService) -> None:
        """Test invalidating credential cache."""
        prefs_service.set_credential_cache("test...fingerp", {"paper": True})
        prefs_service.invalidate_credential_cache()

        cache = prefs_service.get_credential_cache()
        assert cache["fingerprint"] is None
        assert cache["validated_at"] is None
        assert cache["validation_modes"] == {}

    def test_cache_valid_same_fingerprint(self, prefs_service: PreferencesService) -> None:
        """Test cache is valid when fingerprint matches and within age limit."""
        fingerprint = "orgsfake...keyfake"
        prefs_service.set_credential_cache(fingerprint, {"paper": True})

        is_valid = prefs_service.is_credential_cache_valid(fingerprint, "paper")
        assert is_valid is True

    def test_cache_invalid_fingerprint_mismatch(self, prefs_service: PreferencesService) -> None:
        """Test cache is invalid when API key fingerprint changes."""
        prefs_service.set_credential_cache("original...finger", {"paper": True})

        is_valid = prefs_service.is_credential_cache_valid("different...finger", "paper")
        assert is_valid is False

    def test_cache_invalid_mode_not_validated(self, prefs_service: PreferencesService) -> None:
        """Test cache is invalid for a mode that wasn't validated."""
        fingerprint = "orgsfake...keyfake"
        prefs_service.set_credential_cache(fingerprint, {"paper": True})

        # read_only was not validated
        is_valid = prefs_service.is_credential_cache_valid(fingerprint, "read_only")
        assert is_valid is False

    def test_cache_invalid_expired(self, prefs_service: PreferencesService) -> None:
        """Test cache is invalid after expiration time."""
        fingerprint = "orgsfake...keyfake"
        prefs_service.set_credential_cache(fingerprint, {"paper": True})

        # Manually set validated_at to 25 hours ago (expired)
        prefs_service.preferences.credential_validated_at = time.time() - (25 * 3600)

        is_valid = prefs_service.is_credential_cache_valid(fingerprint, "paper")
        assert is_valid is False

    def test_cache_valid_within_age_limit(self, prefs_service: PreferencesService) -> None:
        """Test cache is valid when within custom age limit."""
        fingerprint = "orgsfake...keyfake"
        prefs_service.set_credential_cache(fingerprint, {"paper": True})

        # Set to 23 hours ago (should still be valid with 24h limit)
        prefs_service.preferences.credential_validated_at = time.time() - (23 * 3600)

        is_valid = prefs_service.is_credential_cache_valid(fingerprint, "paper", max_age_hours=24.0)
        assert is_valid is True

    def test_cache_persists_across_sessions(self, temp_prefs_path: Path) -> None:
        """Test cache is saved to file and persists across sessions."""
        fingerprint = "orgsfake...keyfake"
        validation_modes = {"paper": True}

        # Create first service instance and save cache
        prefs1 = PreferencesService(preferences_path=temp_prefs_path)
        prefs1.set_credential_cache(fingerprint, validation_modes)

        # Create second service instance (simulating restart)
        prefs2 = PreferencesService(preferences_path=temp_prefs_path)

        # Cache should be loaded from file
        cache = prefs2.get_credential_cache()
        assert cache["fingerprint"] == fingerprint
        assert cache["validation_modes"] == validation_modes

    def test_update_existing_cache_modes(self, prefs_service: PreferencesService) -> None:
        """Test updating cache adds new modes while preserving existing."""
        fingerprint = "orgsfake...keyfake"

        # First validation for paper mode
        prefs_service.set_credential_cache(fingerprint, {"paper": True})

        # Second validation adds live mode
        cache = prefs_service.get_credential_cache()
        existing_modes = cache.get("validation_modes", {})
        existing_modes["live"] = True
        prefs_service.set_credential_cache(fingerprint, existing_modes)

        # Both modes should be cached
        final_cache = prefs_service.get_credential_cache()
        assert final_cache["validation_modes"]["paper"] is True
        assert final_cache["validation_modes"]["live"] is True


class TestCredentialFingerprint:
    """Tests for credential fingerprint generation."""

    def test_fingerprint_generation(self) -> None:
        """Test fingerprint is generated from API key."""
        from gpt_trader.tui.services.credential_validator import CredentialValidator

        with patch.dict(
            "os.environ",
            {"COINBASE_CDP_API_KEY": "organizations/abc12345/apiKeys/xyz98765"},
        ):
            validator = CredentialValidator()
            fp = validator.compute_credential_fingerprint()

            assert fp is not None
            assert fp.startswith("organiza")  # First 8 chars
            assert fp.endswith("xyz98765")  # Last 8 chars
            assert "..." in fp

    def test_fingerprint_none_when_no_key(self) -> None:
        """Test fingerprint is None when no API key configured."""
        from gpt_trader.tui.services.credential_validator import CredentialValidator

        with patch.dict(
            "os.environ",
            {"COINBASE_CDP_API_KEY": "", "COINBASE_PROD_CDP_API_KEY": ""},
            clear=True,
        ):
            validator = CredentialValidator()
            fp = validator.compute_credential_fingerprint()
            assert fp is None

    def test_fingerprint_none_for_short_key(self) -> None:
        """Test fingerprint is None for keys shorter than 16 chars."""
        from gpt_trader.tui.services.credential_validator import CredentialValidator

        with patch.dict("os.environ", {"COINBASE_CDP_API_KEY": "short"}):
            validator = CredentialValidator()
            fp = validator.compute_credential_fingerprint()
            assert fp is None
