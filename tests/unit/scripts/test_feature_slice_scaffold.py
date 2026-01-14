from __future__ import annotations

from scripts.maintenance import feature_slice_scaffold


def test_dry_run_prints_actions_without_writes(tmp_path, capsys) -> None:
    result = feature_slice_scaffold.scaffold_slice(
        "alpha",
        root=tmp_path,
        with_tests=True,
        with_readme=True,
        dry_run=True,
    )

    captured = capsys.readouterr()

    assert result == 0
    assert "DRY-RUN" in captured.out
    assert not (tmp_path / "src" / "gpt_trader" / "features" / "alpha").exists()


def test_apply_creates_scaffold(tmp_path) -> None:
    result = feature_slice_scaffold.scaffold_slice(
        "beta",
        root=tmp_path,
        with_tests=True,
        with_readme=True,
    )

    assert result == 0

    slice_dir = tmp_path / "src" / "gpt_trader" / "features" / "beta"
    tests_dir = tmp_path / "tests" / "unit" / "gpt_trader" / "features" / "beta"

    assert slice_dir.is_dir()
    assert (
        (slice_dir / "__init__.py")
        .read_text(encoding="utf-8")
        .startswith('"""Beta feature slice."""')
    )
    assert (slice_dir / "README.md").exists()
    assert tests_dir.is_dir()
    assert (tests_dir / "test_beta_slice.py").exists()


def test_refuses_overwrite(tmp_path, capsys) -> None:
    existing_dir = tmp_path / "src" / "gpt_trader" / "features" / "gamma"
    existing_dir.mkdir(parents=True)
    (existing_dir / "__init__.py").write_text("existing", encoding="utf-8")

    result = feature_slice_scaffold.scaffold_slice("gamma", root=tmp_path)
    captured = capsys.readouterr()

    assert result == 1
    assert "Refusing to overwrite" in captured.err
    assert not (tmp_path / "tests" / "unit" / "gpt_trader" / "features" / "gamma").exists()
