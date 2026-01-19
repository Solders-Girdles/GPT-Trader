from __future__ import annotations

from unittest.mock import MagicMock

from textual.widgets import DataTable


def create_mock_datatable() -> MagicMock:
    """Create a properly configured mock DataTable with row-key support."""
    mock_table = MagicMock(spec=DataTable)
    mock_table.rows = MagicMock()
    mock_table.rows.keys.return_value = set()
    mock_table.row_count = 0
    mock_table.columns = MagicMock()
    mock_table.columns.keys.return_value = []
    return mock_table
