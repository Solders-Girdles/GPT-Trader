from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.agents.change_impact as change_impact
import scripts.agents.generate_test_inventory as generate_test_inventory

FRESH_MAP_ENTRY = {"gpt_trader.sample_module": ["tests/unit/test_sample.py"]}


def _point_at_tmp_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route change_impact's map path and the generator's scan at a tmp repo."""
    map_path = tmp_path / "var" / "agents" / "testing" / "source_test_map.json"
    tests_dir = tmp_path / "tests" / "unit"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text(
        "from gpt_trader.sample_module import sample_thing\n"
        "\n"
        "\n"
        "def test_sample() -> None:\n"
        "    assert sample_thing\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(change_impact, "SOURCE_TEST_MAP_PATH", map_path)
    # The shared helper scans PROJECT_ROOT / "tests" and records relative paths.
    monkeypatch.setattr(generate_test_inventory, "PROJECT_ROOT", tmp_path)
    return map_path


def test_load_source_test_map_refreshes_stale_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """agent-impact must not silently use a stale gitignored map."""
    map_path = _point_at_tmp_repo(monkeypatch, tmp_path)
    stale_payload = {"source_to_tests": {"gpt_trader.deleted_module": ["tests/unit/test_gone.py"]}}
    map_path.parent.mkdir(parents=True)
    map_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    source_test_map = change_impact.load_source_test_map()

    assert source_test_map is not None
    assert source_test_map["source_to_tests"] == FRESH_MAP_ENTRY
    on_disk = json.loads(map_path.read_text(encoding="utf-8"))
    assert on_disk == source_test_map


def test_load_source_test_map_creates_missing_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_path = _point_at_tmp_repo(monkeypatch, tmp_path)
    assert not map_path.exists()

    source_test_map = change_impact.load_source_test_map()

    assert source_test_map is not None
    assert source_test_map["source_to_tests"] == FRESH_MAP_ENTRY
    assert map_path.exists()


def test_load_source_test_map_falls_back_to_disk_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    map_path = _point_at_tmp_repo(monkeypatch, tmp_path)
    existing_payload = {"source_to_tests": {"gpt_trader.existing": ["tests/unit/test_old.py"]}}
    map_path.parent.mkdir(parents=True)
    map_path.write_text(json.dumps(existing_payload), encoding="utf-8")

    def _boom(_test_dir: Path) -> dict[str, object]:
        raise RuntimeError("scan failed")

    monkeypatch.setattr(generate_test_inventory, "scan_test_files", _boom)

    source_test_map = change_impact.load_source_test_map()

    assert source_test_map == existing_payload
    captured = capsys.readouterr()
    assert "failed to regenerate source test map" in captured.err
    assert "possibly stale" in captured.err


def test_load_source_test_map_warns_and_returns_none_without_any_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _point_at_tmp_repo(monkeypatch, tmp_path)

    def _boom(_test_dir: Path) -> dict[str, object]:
        raise RuntimeError("scan failed")

    monkeypatch.setattr(generate_test_inventory, "scan_test_files", _boom)

    source_test_map = change_impact.load_source_test_map()

    assert source_test_map is None
    assert "import-map test suggestions skipped" in capsys.readouterr().err
