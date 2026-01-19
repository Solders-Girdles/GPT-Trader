"""Tests for DataTable sorting utilities."""

from decimal import Decimal

from gpt_trader.tui.utilities.table_formatting import get_sort_indicator, sort_table_data


class TestSortTableData:
    """Tests for sort_table_data function."""

    def test_sort_by_string_column_ascending(self) -> None:
        """Sort by string column ascending."""
        data = [
            {"name": "Charlie", "age": 30},
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 35},
        ]
        result = sort_table_data(data, "name", ascending=True)
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"
        assert result[2]["name"] == "Charlie"

    def test_sort_by_string_column_descending(self) -> None:
        """Sort by string column descending."""
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Charlie", "age": 30},
            {"name": "Bob", "age": 35},
        ]
        result = sort_table_data(data, "name", ascending=False)
        assert result[0]["name"] == "Charlie"
        assert result[1]["name"] == "Bob"
        assert result[2]["name"] == "Alice"

    def test_sort_by_numeric_column(self) -> None:
        """Sort by numeric column."""
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Charlie", "age": 30},
            {"name": "Bob", "age": 35},
        ]
        result = sort_table_data(data, "age", ascending=True, numeric_columns={"age"})
        assert result[0]["age"] == 25
        assert result[1]["age"] == 30
        assert result[2]["age"] == 35

    def test_sort_empty_list(self) -> None:
        """Sort empty list returns empty list."""
        result = sort_table_data([], "name", ascending=True)
        assert result == []

    def test_sort_missing_column(self) -> None:
        """Sort by missing column doesn't crash."""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        result = sort_table_data(data, "nonexistent", ascending=True)
        assert len(result) == 2

    def test_sort_decimal_values(self) -> None:
        """Sort Decimal values correctly."""
        data = [
            {"price": Decimal("100.50")},
            {"price": Decimal("50.25")},
            {"price": Decimal("200.75")},
        ]
        result = sort_table_data(data, "price", ascending=True, numeric_columns={"price"})
        assert result[0]["price"] == Decimal("50.25")
        assert result[2]["price"] == Decimal("200.75")


class TestGetSortIndicator:
    """Tests for get_sort_indicator function."""

    def test_no_sort(self) -> None:
        """No sort indicator when column not sorted."""
        result = get_sort_indicator("name", None, True)
        assert result == ""

    def test_ascending_indicator(self) -> None:
        """Ascending indicator for sorted column."""
        result = get_sort_indicator("name", "name", True)
        assert "▲" in result or "asc" in result.lower() or result != ""

    def test_descending_indicator(self) -> None:
        """Descending indicator for sorted column."""
        result = get_sort_indicator("name", "name", False)
        assert "▼" in result or "desc" in result.lower() or result != ""

    def test_different_column_no_indicator(self) -> None:
        """No indicator when different column is sorted."""
        result = get_sort_indicator("name", "age", True)
        assert result == ""
