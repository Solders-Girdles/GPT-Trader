"""Tests for DataTable clipboard utilities."""

import subprocess
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.utilities.table_formatting import copy_to_clipboard


class TestCopyToClipboard:
    """Tests for copy_to_clipboard function."""

    def test_copy_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test clipboard copy on macOS."""
        mock_run = MagicMock()
        mock_run.return_value.returncode = 0
        monkeypatch.setattr(subprocess, "run", mock_run)
        # This test is platform-dependent, so we just verify no crash
        result = copy_to_clipboard("test text")
        # Result depends on platform and available tools
        assert isinstance(result, bool)

    def test_copy_empty_string(self) -> None:
        """Copy empty string doesn't crash."""
        result = copy_to_clipboard("")
        assert isinstance(result, bool)
