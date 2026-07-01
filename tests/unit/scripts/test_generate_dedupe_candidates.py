from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.agents.generate_dedupe_candidates as generate_dedupe_candidates
import scripts.agents.generate_test_inventory as generate_test_inventory


def _point_at_tmp_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the dedupe script's map path and test scan at a tmp repo layout."""
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
    monkeypatch.setattr(generate_dedupe_candidates, "SOURCE_TEST_MAP_PATH", map_path)
    monkeypatch.setattr(generate_dedupe_candidates, "TESTS_DIR", tmp_path / "tests")
    # scan_test_files records paths relative to the generator's PROJECT_ROOT.
    monkeypatch.setattr(generate_test_inventory, "PROJECT_ROOT", tmp_path)
    return map_path


def test_load_source_test_map_regenerates_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    map_path = _point_at_tmp_repo(monkeypatch, tmp_path)
    assert not map_path.exists()

    source_test_map = generate_dedupe_candidates.load_source_test_map()

    assert map_path.exists()
    assert source_test_map["source_to_tests"] == {
        "gpt_trader.sample_module": ["tests/unit/test_sample.py"]
    }
    on_disk = json.loads(map_path.read_text(encoding="utf-8"))
    assert on_disk == source_test_map
    assert "regenerating" in capsys.readouterr().out


def test_load_source_test_map_reads_existing_without_regenerating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    map_path = _point_at_tmp_repo(monkeypatch, tmp_path)
    payload = {"source_to_tests": {"gpt_trader.existing": ["tests/unit/test_existing.py"]}}
    map_path.parent.mkdir(parents=True)
    map_path.write_text(json.dumps(payload), encoding="utf-8")

    source_test_map = generate_dedupe_candidates.load_source_test_map()

    assert source_test_map == payload
    assert "regenerating" not in capsys.readouterr().out


def test_load_source_test_map_exits_when_regeneration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _point_at_tmp_repo(monkeypatch, tmp_path)

    def _boom(_test_dir: Path) -> dict[str, object]:
        raise RuntimeError("scan failed")

    monkeypatch.setattr(generate_test_inventory, "scan_test_files", _boom)

    with pytest.raises(SystemExit) as exc_info:
        generate_dedupe_candidates.load_source_test_map()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "failed to regenerate source test map" in captured.err
    assert "agent-regenerate --only testing" in captured.err
