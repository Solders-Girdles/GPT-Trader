"""Tests for DataTable ID formatting utilities."""

from gpt_trader.tui.utilities.table_formatting import truncate_id


class TestTruncateId:
    """Tests for truncate_id function."""

    def test_long_id_truncated(self) -> None:
        """Long ID is truncated to last N characters."""
        result = truncate_id("abc123def456ghi789", length=8)
        assert result == "56ghi789"

    def test_short_id_unchanged(self) -> None:
        """Short ID (shorter than length) is unchanged."""
        result = truncate_id("short", length=8)
        assert result == "short"

    def test_exact_length_unchanged(self) -> None:
        """ID exactly at length is unchanged."""
        result = truncate_id("12345678", length=8)
        assert result == "12345678"

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        result = truncate_id("")
        assert result == ""

    def test_none_returns_empty(self) -> None:
        """None returns empty string."""
        result = truncate_id(None)  # type: ignore[arg-type]
        assert result == ""

    def test_default_length(self) -> None:
        """Default length is 8."""
        result = truncate_id("0123456789abcdef")
        assert result == "89abcdef"
        assert len(result) == 8
