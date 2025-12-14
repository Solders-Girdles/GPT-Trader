"""Tests for ThemeService."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from gpt_trader.tui.services.theme_service import ThemeService
from gpt_trader.tui.theme import ThemeMode


class TestThemeService:
    """Test ThemeService functionality."""

    def test_init_creates_theme_manager(self):
        """Test that initialization creates a theme manager."""
        mock_app = MagicMock()
        service = ThemeService(mock_app)

        assert service.app == mock_app
        assert service.theme_manager is not None

    def test_load_preference_returns_default_when_no_file(self):
        """Test load_preference returns dark mode when no file exists."""
        mock_app = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_path = Path(tmpdir) / "nonexistent.json"
            service = ThemeService(mock_app, preferences_path=prefs_path)

            result = service.load_preference()

            assert result == ThemeMode.DARK

    def test_load_preference_reads_from_file(self):
        """Test load_preference reads theme from file."""
        mock_app = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_path = Path(tmpdir) / "prefs.json"
            prefs_path.write_text(json.dumps({"theme": "light"}))

            service = ThemeService(mock_app, preferences_path=prefs_path)
            result = service.load_preference()

            assert result == ThemeMode.LIGHT

    def test_save_preference_creates_file(self):
        """Test save_preference creates preferences file."""
        mock_app = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_path = Path(tmpdir) / "subdir" / "prefs.json"
            service = ThemeService(mock_app, preferences_path=prefs_path)

            result = service.save_preference(ThemeMode.LIGHT)

            assert result is True
            assert prefs_path.exists()
            data = json.loads(prefs_path.read_text())
            assert data["theme"] == "light"

    def test_save_preference_preserves_other_prefs(self):
        """Test save_preference preserves other preferences."""
        mock_app = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_path = Path(tmpdir) / "prefs.json"
            prefs_path.write_text(json.dumps({"theme": "dark", "other": "value"}))

            service = ThemeService(mock_app, preferences_path=prefs_path)
            service.save_preference(ThemeMode.LIGHT)

            data = json.loads(prefs_path.read_text())
            assert data["theme"] == "light"
            assert data["other"] == "value"

    def test_toggle_theme_notifies_user(self):
        """Test toggle_theme notifies the user."""
        mock_app = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_path = Path(tmpdir) / "prefs.json"
            service = ThemeService(mock_app, preferences_path=prefs_path)

            service.toggle_theme()

            mock_app.notify.assert_called_once()
            call_args = mock_app.notify.call_args
            assert "theme" in call_args[0][0].lower()

    def test_toggle_theme_posts_event(self):
        """Test toggle_theme posts ThemeChanged event."""
        mock_app = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_path = Path(tmpdir) / "prefs.json"
            service = ThemeService(mock_app, preferences_path=prefs_path)

            service.toggle_theme()

            mock_app.post_message.assert_called_once()

    def test_current_theme_property(self):
        """Test current_theme returns the current theme."""
        mock_app = MagicMock()
        service = ThemeService(mock_app)

        theme = service.current_theme

        assert theme is not None

    def test_current_mode_property(self):
        """Test current_mode returns the current mode."""
        mock_app = MagicMock()
        service = ThemeService(mock_app)

        mode = service.current_mode

        assert mode in (ThemeMode.DARK, ThemeMode.LIGHT)
