"""Tests for ConfigStore persistence functionality."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bot_v2.persistence.config_store import ConfigStore


class TestConfigStore:
    """Test the ConfigStore class."""

    def test_config_store_init_default_path(self) -> None:
        """Test ConfigStore initialization with default path."""
        store = ConfigStore()

        # Should create a path ending with bots.json in a results/managed directory
        path_str = str(store.path)
        assert path_str.endswith("bots.json")
        assert "results" in path_str
        assert "managed" in path_str

    def test_config_store_init_custom_path(self) -> None:
        """Test ConfigStore initialization with custom path."""
        custom_root = Path("/tmp/test_config")
        store = ConfigStore(root=custom_root)

        expected_path = custom_root / "bots.json"
        assert store.path == expected_path

    @patch("bot_v2.persistence.config_store.JsonFileStore")
    @patch("pathlib.Path.stat")
    def test_config_store_init_empty_file(self, mock_stat: Mock, mock_json_store: Mock) -> None:
        """Test ConfigStore initialization creates default structure for empty file."""
        # Mock empty file
        mock_stat.return_value.st_size = 0
        mock_store_instance = Mock()
        mock_json_store.return_value = mock_store_instance

        store = ConfigStore()

        # Should write default structure
        mock_store_instance.write_json.assert_called_once_with({"bots": []})

    @patch("bot_v2.persistence.config_store.JsonFileStore")
    @patch("pathlib.Path.stat")
    def test_config_store_init_non_empty_file(self, mock_stat: Mock, mock_json_store: Mock) -> None:
        """Test ConfigStore initialization doesn't overwrite existing data."""
        # Mock non-empty file
        mock_stat.return_value.st_size = 100
        mock_store_instance = Mock()
        mock_json_store.return_value = mock_store_instance

        store = ConfigStore()

        # Should not write anything for non-empty file
        mock_store_instance.write_json.assert_not_called()

    def test_load_bots_default_empty(self) -> None:
        """Test load_bots returns empty list for missing/invalid data."""
        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = None
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            bots = store.load_bots()

            assert bots == []

    def test_load_bots_with_valid_data(self) -> None:
        """Test load_bots returns valid bot list."""
        test_data = {"bots": [{"bot_id": "test1", "name": "Test Bot 1"}]}

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = test_data
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            bots = store.load_bots()

            assert bots == [{"bot_id": "test1", "name": "Test Bot 1"}]

    def test_load_bots_with_invalid_structure(self) -> None:
        """Test load_bots handles various invalid data structures."""
        test_cases = [
            None,
            {},
            {"invalid_key": []},
            {"bots": "not_a_list"},
            {"bots": None},
            "not_a_dict",
            [],
        ]

        for test_data in test_cases:
            with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
                mock_store_instance = Mock()
                mock_store_instance.read_json.return_value = test_data
                mock_json_store.return_value = mock_store_instance

                store = ConfigStore()
                bots = store.load_bots()

                assert bots == []

    def test_save_bots(self) -> None:
        """Test save_bots writes data correctly."""
        test_bots = [{"bot_id": "test1", "name": "Test Bot 1"}]

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.save_bots(test_bots)

            mock_store_instance.write_json.assert_called_once_with({"bots": test_bots})

    def test_add_bot_new(self) -> None:
        """Test add_bot adds new bot configuration."""
        existing_bots = [{"bot_id": "existing1", "name": "Existing Bot"}]
        new_bot = {"bot_id": "new1", "name": "New Bot"}

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = {"bots": existing_bots}
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.add_bot(new_bot)

            expected_bots = existing_bots + [new_bot]
            mock_store_instance.write_json.assert_called_once_with({"bots": expected_bots})

    def test_add_bot_replace_existing(self) -> None:
        """Test add_bot replaces existing bot with same ID."""
        existing_bots = [
            {"bot_id": "bot1", "name": "Old Name"},
            {"bot_id": "bot2", "name": "Another Bot"},
        ]
        updated_bot = {"bot_id": "bot1", "name": "Updated Name"}

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = {"bots": existing_bots}
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.add_bot(updated_bot)

            expected_bots = [
                {"bot_id": "bot2", "name": "Another Bot"},
                {"bot_id": "bot1", "name": "Updated Name"},
            ]
            mock_store_instance.write_json.assert_called_once_with({"bots": expected_bots})

    def test_add_bot_without_id(self) -> None:
        """Test add_bot handles bot without ID."""
        existing_bots = [{"bot_id": "existing1", "name": "Existing Bot"}]
        new_bot = {"name": "Bot without ID"}

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = {"bots": existing_bots}
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.add_bot(new_bot)

            # Should append since no ID to match
            expected_bots = existing_bots + [new_bot]
            mock_store_instance.write_json.assert_called_once_with({"bots": expected_bots})

    def test_remove_bot_existing(self) -> None:
        """Test remove_bot removes existing bot."""
        existing_bots = [
            {"bot_id": "bot1", "name": "Bot 1"},
            {"bot_id": "bot2", "name": "Bot 2"},
            {"bot_id": "bot3", "name": "Bot 3"},
        ]

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = {"bots": existing_bots}
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.remove_bot("bot2")

            expected_bots = [
                {"bot_id": "bot1", "name": "Bot 1"},
                {"bot_id": "bot3", "name": "Bot 3"},
            ]
            mock_store_instance.write_json.assert_called_once_with({"bots": expected_bots})

    def test_remove_bot_nonexistent(self) -> None:
        """Test remove_bot handles non-existent bot ID."""
        existing_bots = [
            {"bot_id": "bot1", "name": "Bot 1"},
            {"bot_id": "bot2", "name": "Bot 2"},
        ]

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = {"bots": existing_bots}
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.remove_bot("nonexistent")

            # Should save unchanged list
            mock_store_instance.write_json.assert_called_once_with({"bots": existing_bots})

    def test_update_bot_existing(self) -> None:
        """Test update_bot updates existing bot."""
        existing_bots = [
            {"bot_id": "bot1", "name": "Bot 1", "status": "active"},
            {"bot_id": "bot2", "name": "Bot 2", "status": "inactive"},
        ]
        updates = {"status": "paused", "version": "2.0"}

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = {"bots": existing_bots}
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.update_bot("bot1", updates)

            expected_bots = [
                {"bot_id": "bot1", "name": "Bot 1", "status": "paused", "version": "2.0"},
                {"bot_id": "bot2", "name": "Bot 2", "status": "inactive"},
            ]
            mock_store_instance.write_json.assert_called_once_with({"bots": expected_bots})

    def test_update_bot_nonexistent(self) -> None:
        """Test update_bot handles non-existent bot ID."""
        existing_bots = [
            {"bot_id": "bot1", "name": "Bot 1"},
            {"bot_id": "bot2", "name": "Bot 2"},
        ]
        updates = {"status": "paused"}

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = {"bots": existing_bots}
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.update_bot("nonexistent", updates)

            # Should save unchanged list
            mock_store_instance.write_json.assert_called_once_with({"bots": existing_bots})

    def test_update_bot_partial_update(self) -> None:
        """Test update_bot only updates specified fields."""
        existing_bots = [
            {"bot_id": "bot1", "name": "Bot 1", "status": "active", "version": "1.0"},
        ]
        updates = {"status": "paused"}

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = {"bots": existing_bots}
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.update_bot("bot1", updates)

            expected_bots = [
                {"bot_id": "bot1", "name": "Bot 1", "status": "paused", "version": "1.0"},
            ]
            mock_store_instance.write_json.assert_called_once_with({"bots": expected_bots})

    def test_integration_workflow(self) -> None:
        """Test complete workflow of bot management."""
        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()

            # Mock read_json to return current state that gets updated
            current_data = {"bots": []}

            def mock_read_json(default=None):
                return current_data

            def mock_write_json(data):
                current_data.update(data)

            mock_store_instance.read_json.side_effect = mock_read_json
            mock_store_instance.write_json.side_effect = mock_write_json
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()

            # Add first bot
            bot1 = {"bot_id": "bot1", "name": "First Bot", "status": "active"}
            store.add_bot(bot1)

            # Add second bot
            bot2 = {"bot_id": "bot2", "name": "Second Bot", "status": "active"}
            store.add_bot(bot2)

            # Update first bot
            store.update_bot("bot1", {"status": "paused"})

            # Remove second bot
            store.remove_bot("bot2")

            # Check final state from the mock
            expected_bots = [
                {"bot_id": "bot1", "name": "First Bot", "status": "paused"},
            ]
            assert current_data == {"bots": expected_bots}

    def test_load_bots_returns_copy(self) -> None:
        """Test that load_bots returns a copy, not reference to internal data."""
        test_data = {"bots": [{"bot_id": "test1"}]}

        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.read_json.return_value = test_data
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            bots1 = store.load_bots()
            bots2 = store.load_bots()

            # Should be equal but not the same object
            assert bots1 == bots2
            assert bots1 is not bots2

    def test_save_bots_with_empty_list(self) -> None:
        """Test save_bots handles empty list."""
        with patch("bot_v2.persistence.config_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = ConfigStore()
            store.save_bots([])

            mock_store_instance.write_json.assert_called_once_with({"bots": []})
