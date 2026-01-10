"""
Unified preferences service for TUI settings persistence.

Manages all user preferences including theme, mode, log level,
and other session settings in a single JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpt_trader.tui.preferences_paths import resolve_preferences_path
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from textual.app import App

logger = get_logger(__name__, component="tui")

# Valid values for constrained fields
VALID_MODES = {"demo", "paper", "read_only", "live"}
VALID_THEMES = {"dark", "light"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


@dataclass
class UserPreferences:
    """User preferences data structure.

    Attributes:
        theme: Color theme ("dark" or "light")
        mode: Last used trading mode
        log_level: Preferred log verbosity level
        last_profile: Last used configuration profile name
        last_symbols: Last used trading symbols
        show_mode_selection: Whether to show mode selection on startup
        compact_view: Whether to use compact dashboard view
        credential_fingerprint: Hash of API key for cache validation
        credential_validated_at: Unix timestamp of last validation
        credential_validation_modes: Cached validation results per mode
    """

    theme: str = "dark"
    mode: str | None = None
    log_level: str = "INFO"
    last_profile: str | None = None
    last_symbols: list[str] = field(default_factory=list)
    show_mode_selection: bool = True
    compact_view: bool = False
    # Credential validation cache
    credential_fingerprint: str | None = None
    credential_validated_at: float | None = None
    credential_validation_modes: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "theme": self.theme,
            "mode": self.mode,
            "log_level": self.log_level,
            "last_profile": self.last_profile,
            "last_symbols": self.last_symbols,
            "show_mode_selection": self.show_mode_selection,
            "compact_view": self.compact_view,
            "credential_fingerprint": self.credential_fingerprint,
            "credential_validated_at": self.credential_validated_at,
            "credential_validation_modes": self.credential_validation_modes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserPreferences:
        """Create from dictionary, with validation."""
        theme = data.get("theme", "dark")
        if theme not in VALID_THEMES:
            theme = "dark"

        mode = data.get("mode")
        if mode is not None and mode not in VALID_MODES:
            mode = None

        log_level = data.get("log_level", "INFO")
        if log_level not in VALID_LOG_LEVELS:
            log_level = "INFO"

        last_symbols = data.get("last_symbols", [])
        if not isinstance(last_symbols, list):
            last_symbols = []

        # Parse credential cache
        validation_modes = data.get("credential_validation_modes", {})
        if not isinstance(validation_modes, dict):
            validation_modes = {}

        return cls(
            theme=theme,
            mode=mode,
            log_level=log_level,
            last_profile=data.get("last_profile"),
            last_symbols=last_symbols,
            show_mode_selection=data.get("show_mode_selection", True),
            compact_view=data.get("compact_view", False),
            credential_fingerprint=data.get("credential_fingerprint"),
            credential_validated_at=data.get("credential_validated_at"),
            credential_validation_modes=validation_modes,
        )


class PreferencesService:
    """Unified service for managing all TUI preferences.

    Provides a single point of access for reading and writing
    user preferences, with validation and atomic updates.

    Attributes:
        app: Reference to the parent Textual app (optional).
        preferences_path: Path to the preferences file.
        preferences: Current loaded preferences.
    """

    def __init__(
        self,
        app: App | None = None,
        preferences_path: Path | None = None,
    ) -> None:
        """Initialize the preferences service.

        Args:
            app: Optional parent Textual app.
            preferences_path: Optional custom path for preferences file.
        """
        self.app = app
        self.preferences_path = resolve_preferences_path(preferences_path)
        self._preferences: UserPreferences | None = None

    @property
    def preferences(self) -> UserPreferences:
        """Get current preferences, loading from file if needed."""
        if self._preferences is None:
            self._preferences = self._load()
        return self._preferences

    def _load(self) -> UserPreferences:
        """Load preferences from file.

        Returns:
            UserPreferences instance with loaded or default values.
        """
        try:
            if self.preferences_path.exists():
                with open(self.preferences_path) as f:
                    data = json.load(f)
                    prefs = UserPreferences.from_dict(data)
                    logger.debug("Loaded preferences from %s", self.preferences_path)
                    return prefs
        except Exception as e:
            logger.debug("Could not load preferences from %s: %s", self.preferences_path, e)

        return UserPreferences()

    def _save(self) -> bool:
        """Save current preferences to file.

        Returns:
            True if saved successfully, False otherwise.
        """
        if self._preferences is None:
            return False

        try:
            self.preferences_path.parent.mkdir(parents=True, exist_ok=True)

            # Preserve unknown keys from any existing file (useful for forward/backward
            # compatibility if older/newer versions store additional fields).
            data: dict[str, Any] = {}
            if self.preferences_path.exists():
                try:
                    with open(self.preferences_path) as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            data = loaded
                except Exception:
                    data = {}

            data.update(self._preferences.to_dict())

            with open(self.preferences_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved preferences to {self.preferences_path}")
            return True
        except Exception as e:
            logger.warning(f"Could not save preferences: {e}")
            return False

    def reload(self) -> UserPreferences:
        """Force reload preferences from file.

        Returns:
            Freshly loaded UserPreferences.
        """
        self._preferences = self._load()
        return self._preferences

    # Theme methods
    def get_theme(self) -> str:
        """Get current theme preference."""
        return self.preferences.theme

    def set_theme(self, theme: str) -> bool:
        """Set theme preference.

        Args:
            theme: Theme name ("dark" or "light")

        Returns:
            True if saved successfully.
        """
        if theme not in VALID_THEMES:
            logger.warning(f"Invalid theme: {theme}")
            return False

        self.preferences.theme = theme
        return self._save()

    # Mode methods
    def get_mode(self) -> str | None:
        """Get last used mode."""
        return self.preferences.mode

    def set_mode(self, mode: str) -> bool:
        """Set last used mode.

        Args:
            mode: Mode name (demo, paper, read_only, live)

        Returns:
            True if saved successfully.
        """
        if mode not in VALID_MODES:
            logger.warning(f"Invalid mode: {mode}")
            return False

        self.preferences.mode = mode
        return self._save()

    # Log level methods
    def get_log_level(self) -> str:
        """Get preferred log level."""
        return self.preferences.log_level

    def set_log_level(self, level: str) -> bool:
        """Set preferred log level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)

        Returns:
            True if saved successfully.
        """
        if level not in VALID_LOG_LEVELS:
            logger.warning(f"Invalid log level: {level}")
            return False

        self.preferences.log_level = level
        return self._save()

    # Profile methods
    def get_last_profile(self) -> str | None:
        """Get last used profile name."""
        return self.preferences.last_profile

    def set_last_profile(self, profile: str | None) -> bool:
        """Set last used profile.

        Args:
            profile: Profile name or None

        Returns:
            True if saved successfully.
        """
        self.preferences.last_profile = profile
        return self._save()

    # Symbols/Watchlist methods
    def get_last_symbols(self) -> list[str]:
        """Get last used trading symbols (watchlist)."""
        return self.preferences.last_symbols.copy()

    def set_last_symbols(self, symbols: list[str]) -> bool:
        """Set last used trading symbols (watchlist).

        Args:
            symbols: List of symbol strings

        Returns:
            True if saved successfully.
        """
        self.preferences.last_symbols = list(symbols)
        return self._save()

    def add_watchlist_symbol(self, symbol: str) -> bool:
        """Add a symbol to the watchlist if not already present.

        Args:
            symbol: Symbol to add (e.g., "BTC-USD")

        Returns:
            True if added successfully (or already present).
        """
        symbol = symbol.upper().strip()
        if not symbol:
            return False

        symbols = self.get_last_symbols()
        if symbol not in symbols:
            symbols.append(symbol)
            return self.set_last_symbols(symbols)
        return True  # Already present

    def remove_watchlist_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the watchlist.

        Args:
            symbol: Symbol to remove

        Returns:
            True if removed successfully (or not present).
        """
        symbol = symbol.upper().strip()
        symbols = self.get_last_symbols()
        if symbol in symbols:
            symbols.remove(symbol)
            return self.set_last_symbols(symbols)
        return True  # Not present

    def clear_watchlist(self) -> bool:
        """Clear all symbols from the watchlist.

        Returns:
            True if cleared successfully.
        """
        return self.set_last_symbols([])

    def reorder_watchlist(self, symbols: list[str]) -> bool:
        """Reorder the watchlist (preserves order).

        Args:
            symbols: List of symbols in desired order

        Returns:
            True if saved successfully.
        """
        # Normalize and dedupe while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for s in symbols:
            s = s.upper().strip()
            if s and s not in seen:
                seen.add(s)
                ordered.append(s)
        return self.set_last_symbols(ordered)

    # View preference methods
    def get_show_mode_selection(self) -> bool:
        """Get whether to show mode selection on startup."""
        return self.preferences.show_mode_selection

    def set_show_mode_selection(self, show: bool) -> bool:
        """Set whether to show mode selection on startup.

        Args:
            show: Whether to show mode selection

        Returns:
            True if saved successfully.
        """
        self.preferences.show_mode_selection = show
        return self._save()

    def get_compact_view(self) -> bool:
        """Get whether to use compact view."""
        return self.preferences.compact_view

    def set_compact_view(self, compact: bool) -> bool:
        """Set whether to use compact view.

        Args:
            compact: Whether to use compact view

        Returns:
            True if saved successfully.
        """
        self.preferences.compact_view = compact
        return self._save()

    # Convenience method for updating multiple preferences
    def update(self, **kwargs: Any) -> bool:
        """Update multiple preferences at once.

        Args:
            **kwargs: Preference key-value pairs to update

        Returns:
            True if all updates and save succeeded.
        """
        prefs = self.preferences
        updated = False

        for key, value in kwargs.items():
            if hasattr(prefs, key):
                # Validate constrained fields
                if key == "theme" and value not in VALID_THEMES:
                    continue
                if key == "mode" and value is not None and value not in VALID_MODES:
                    continue
                if key == "log_level" and value not in VALID_LOG_LEVELS:
                    continue

                setattr(prefs, key, value)
                updated = True

        if updated:
            return self._save()
        return True

    # =========================================================================
    # Credential Cache Methods
    # =========================================================================

    def get_credential_cache(self) -> dict[str, Any]:
        """Get cached credential validation info.

        Returns:
            Dictionary with fingerprint, validated_at, and validation_modes.
        """
        return {
            "fingerprint": self.preferences.credential_fingerprint,
            "validated_at": self.preferences.credential_validated_at,
            "validation_modes": self.preferences.credential_validation_modes.copy(),
        }

    def set_credential_cache(
        self,
        fingerprint: str,
        validation_modes: dict[str, bool],
    ) -> bool:
        """Store credential validation cache.

        Args:
            fingerprint: Hash/fingerprint of the API key.
            validation_modes: Dict mapping mode names to validation success.

        Returns:
            True if saved successfully.
        """
        import time

        self.preferences.credential_fingerprint = fingerprint
        self.preferences.credential_validated_at = time.time()
        self.preferences.credential_validation_modes = validation_modes.copy()
        logger.debug(f"Cached credential validation: {list(validation_modes.keys())}")
        return self._save()

    def invalidate_credential_cache(self) -> bool:
        """Clear the credential validation cache.

        Returns:
            True if saved successfully.
        """
        self.preferences.credential_fingerprint = None
        self.preferences.credential_validated_at = None
        self.preferences.credential_validation_modes = {}
        logger.debug("Invalidated credential cache")
        return self._save()

    def is_credential_cache_valid(
        self,
        current_fingerprint: str,
        mode: str,
        max_age_hours: float = 24.0,
    ) -> bool:
        """Check if cached credentials are valid for the specified mode.

        Args:
            current_fingerprint: Fingerprint of current API key.
            mode: Trading mode to check validation for.
            max_age_hours: Maximum age of cache in hours (default 24).

        Returns:
            True if cache is valid and credentials were validated for this mode.
        """
        import time

        prefs = self.preferences

        # Check fingerprint matches (credentials haven't changed)
        if prefs.credential_fingerprint != current_fingerprint:
            logger.debug("Credential cache invalid: fingerprint mismatch")
            return False

        # Check cache age
        if prefs.credential_validated_at is None:
            logger.debug("Credential cache invalid: no validation timestamp")
            return False

        age_seconds = time.time() - prefs.credential_validated_at
        max_age_seconds = max_age_hours * 3600

        if age_seconds > max_age_seconds:
            logger.debug(f"Credential cache expired: {age_seconds/3600:.1f}h > {max_age_hours}h")
            return False

        # Check if this mode was validated
        if not prefs.credential_validation_modes.get(mode, False):
            logger.debug(f"Credential cache invalid: mode '{mode}' not validated")
            return False

        logger.debug(f"Credential cache valid for '{mode}' " f"(age: {age_seconds/3600:.1f}h)")
        return True


# Singleton instance for global access
_preferences_service: PreferencesService | None = None


def get_preferences_service(
    preferences_path: Path | None = None,
) -> PreferencesService:
    """Get the global preferences service instance.

    Args:
        preferences_path: Optional custom path (only used on first call)

    Returns:
        The global PreferencesService instance.
    """
    global _preferences_service
    if _preferences_service is None:
        _preferences_service = PreferencesService(preferences_path=preferences_path)
    return _preferences_service
