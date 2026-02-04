"""Tests for scripts/ci/check_test_hygiene.py."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

import scripts.ci.check_test_hygiene as check_test_hygiene

if TYPE_CHECKING:
    from pathlib import Path


def _write_test_file(path: Path, content: str = "", lines: int | None = None) -> None:
    """Write a test file with given content or generate lines."""
    if lines is not None:
        content = "\n".join(f"# line {i}" for i in range(lines))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestValidCases:
    """Tests that verify valid/clean examples pass successfully."""

    def test_valid_unit_test_in_gpt_trader_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A properly placed unit test should produce no problems."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, "def test_something(): pass\n")

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 0

    def test_valid_unit_test_in_scripts_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unit test in scripts directory should pass."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "scripts"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, "def test_something(): pass\n")

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 0

    def test_valid_integration_test_with_marker_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration test with proper marker should pass."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "integration"
        test_file = test_dir / "test_example.py"
        _write_test_file(
            test_file,
            "import pytest\n\n@pytest.mark.integration\ndef test_something(): pass\n",
        )

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 0


class TestLayoutViolations:
    """Tests for test file placement violations."""

    def test_unit_test_outside_allowed_prefix_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unit test outside allowed prefixes should be flagged."""
        monkeypatch.chdir(tmp_path)
        # Create test in tests/unit/wrong_place/ (not gpt_trader/scripts/support)
        test_dir = tmp_path / "tests" / "unit" / "wrong_place"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, "def test_something(): pass\n")

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1


class TestMarkerViolations:
    """Tests for missing or misplaced pytest markers."""

    def test_integration_test_missing_marker_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration test without marker should be flagged."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "integration"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, "def test_something(): pass\n")

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1

    def test_contract_test_missing_marker_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contract test without marker should be flagged."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "contract"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, "def test_something(): pass\n")

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1

    def test_real_api_test_missing_marker_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real API test without marker should be flagged."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "real_api"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, "def test_something(): pass\n")

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1

    def test_unit_test_with_integration_marker_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unit test with integration marker should be flagged for reclassification."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(
            test_file,
            "import pytest\n\n@pytest.mark.integration\ndef test_something(): pass\n",
        )

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1

    def test_integration_test_with_real_api_marker_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration test with real_api marker should be flagged to move."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "integration"
        test_file = test_dir / "test_example.py"
        _write_test_file(
            test_file,
            "import pytest\n\n@pytest.mark.integration\n@pytest.mark.real_api\ndef test_something(): pass\n",
        )

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1


class TestSizeViolations:
    """Tests for file size threshold violations."""

    def test_test_exceeding_threshold_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test file over threshold should be flagged."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        # Generate file with more than THRESHOLD (400) lines
        _write_test_file(test_file, lines=450)

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1

    def test_test_under_threshold_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test file under threshold should pass."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, lines=100)

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 0


class TestSleepViolations:
    """Tests for time.sleep() usage violations."""

    def test_time_sleep_without_fake_clock_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test using time.sleep without fake_clock should be flagged."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(
            test_file,
            "import time\n\ndef test_something():\n    time.sleep(1)\n",
        )

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1

    def test_time_sleep_inside_string_literal_not_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sample code containing 'time.sleep(' in a string should not be flagged."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(
            test_file,
            """def test_something():\n    sample = \"time.sleep(1)\"\n    assert sample\n""",
        )

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 0

    def test_time_sleep_with_fake_clock_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test using time.sleep with fake_clock fixture should pass."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(
            test_file,
            "import time\n\ndef test_something(fake_clock):\n    time.sleep(1)\n",
        )

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 0


class TestPatchViolations:
    """Tests for patch() usage violations (prefer monkeypatch)."""

    def test_patch_call_flagged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test using patch() should be flagged."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(
            test_file,
            "from unittest.mock import patch\n\n@patch('module.func')\ndef test_something(mock_func): pass\n",
        )

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1

    def test_patch_object_call_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test using patch.object() should also be flagged."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(
            test_file,
            "from unittest.mock import patch\n\n@patch.object(SomeClass, 'method')\ndef test_something(mock): pass\n",
        )

        result = check_test_hygiene.scan([str(test_dir)])

        assert result == 1


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_is_patch_callable_detects_patch_name(self) -> None:
        """_is_patch_callable should detect bare 'patch' name."""
        import ast

        node = ast.Name(id="patch")
        assert check_test_hygiene._is_patch_callable(node) is True

    def test_is_patch_callable_detects_patch_attribute(self) -> None:
        """_is_patch_callable should detect 'mock.patch' attribute."""
        import ast

        # mock.patch
        node = ast.Attribute(
            value=ast.Name(id="mock"),
            attr="patch",
        )
        assert check_test_hygiene._is_patch_callable(node) is True

    def test_is_patch_callable_detects_patch_object(self) -> None:
        """_is_patch_callable should detect 'patch.object' chain."""
        import ast

        # patch.object
        node = ast.Attribute(
            value=ast.Name(id="patch"),
            attr="object",
        )
        assert check_test_hygiene._is_patch_callable(node) is True

    def test_first_patch_call_line_returns_none_for_no_patch(self) -> None:
        """_first_patch_call_line should return None when no patch present."""
        code = "def test_foo(): pass\n"
        assert check_test_hygiene._first_patch_call_line(code) is None

    def test_first_patch_call_line_returns_line_number(self) -> None:
        """_first_patch_call_line should return line number of first patch call."""
        code = "from unittest.mock import patch\n\npatch('foo')\n"
        result = check_test_hygiene._first_patch_call_line(code)
        assert result == 3

    def test_first_patch_call_line_handles_syntax_error(self) -> None:
        """_first_patch_call_line should return None on syntax error."""
        code = "def broken( patch\n"
        assert check_test_hygiene._first_patch_call_line(code) is None


class TestMainEntryPoint:
    """Tests for main() function and CLI interface."""

    def test_main_returns_zero_for_valid_tests(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() should return 0 when all tests pass hygiene checks."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "unit" / "gpt_trader"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, "def test_something(): pass\n")

        result = check_test_hygiene.main([str(test_dir)])

        assert result == 0

    def test_main_returns_one_for_violations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() should return 1 when hygiene violations found."""
        monkeypatch.chdir(tmp_path)
        test_dir = tmp_path / "tests" / "integration"
        test_file = test_dir / "test_example.py"
        _write_test_file(test_file, "def test_something(): pass\n")  # Missing marker

        result = check_test_hygiene.main([str(test_dir)])

        assert result == 1
