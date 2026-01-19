"""Tests for ModeService preference persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpt_trader.config import path_registry
from gpt_trader.tui.services.mode_service import ModeService


class TestModePreferencePersistence:
    """Test mode preference persistence functionality."""

    def test_load_mode_preference_returns_saved_mode(self, tmp_path: Path):
        """Test loading a previously saved mode."""
        prefs_file = tmp_path / "tui_preferences.json"
        prefs_file.write_text(json.dumps({"mode": "paper", "theme": "dark"}))

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.load_mode_preference()

        assert result == "paper"

    def test_load_mode_preference_returns_none_when_missing(self, tmp_path: Path):
        """Test returns None when no mode saved."""
        prefs_file = tmp_path / "tui_preferences.json"
        prefs_file.write_text(json.dumps({"theme": "dark"}))  # No mode key

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.load_mode_preference()

        assert result is None

    def test_load_mode_preference_returns_none_for_nonexistent_file(self, tmp_path: Path):
        """Test returns None when preferences file doesn't exist."""
        prefs_file = tmp_path / "nonexistent.json"

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.load_mode_preference()

        assert result is None

    def test_load_mode_preference_ignores_invalid_mode(self, tmp_path: Path):
        """Test ignores saved mode if it's not a valid mode."""
        prefs_file = tmp_path / "tui_preferences.json"
        prefs_file.write_text(json.dumps({"mode": "invalid_mode"}))

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.load_mode_preference()

        assert result is None

    def test_save_mode_preference_creates_file(self, tmp_path: Path):
        """Test saving mode creates preferences file."""
        prefs_file = tmp_path / "config" / "tui_preferences.json"
        # Parent directory doesn't exist yet

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.save_mode_preference("demo")

        assert result is True
        assert prefs_file.exists()
        saved = json.loads(prefs_file.read_text())
        assert saved["mode"] == "demo"

    def test_save_mode_preference_preserves_other_keys(self, tmp_path: Path):
        """Test saving mode preserves other preferences."""
        prefs_file = tmp_path / "tui_preferences.json"
        prefs_file.write_text(json.dumps({"theme": "light", "other": "value"}))

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.save_mode_preference("live")

        assert result is True
        saved = json.loads(prefs_file.read_text())
        assert saved["mode"] == "live"
        assert saved["theme"] == "light"
        assert saved["other"] == "value"

    def test_save_mode_preference_rejects_invalid_mode(self, tmp_path: Path):
        """Test saving invalid mode returns False."""
        prefs_file = tmp_path / "tui_preferences.json"

        mock_app = MagicMock()
        service = ModeService(mock_app, preferences_path=prefs_file)

        result = service.save_mode_preference("invalid_mode")

        assert result is False
        assert not prefs_file.exists()

    def test_save_mode_preference_accepts_all_valid_modes(self, tmp_path: Path):
        """Test all valid modes can be saved."""
        valid_modes = ["demo", "paper", "read_only", "live"]

        for mode in valid_modes:
            prefs_file = tmp_path / f"prefs_{mode}.json"

            mock_app = MagicMock()
            service = ModeService(mock_app, preferences_path=prefs_file)

            result = service.save_mode_preference(mode)

            assert result is True, f"Failed to save mode: {mode}"
            saved = json.loads(prefs_file.read_text())
            assert saved["mode"] == mode

    @pytest.mark.uses_real_preferences
    def test_init_sets_default_preferences_path(self):
        """Test init uses default preferences path when not specified."""
        mock_app = MagicMock()
        service = ModeService(mock_app)

        assert service.preferences_path == (path_registry.RUNTIME_DATA_DIR / "tui_preferences.json")
