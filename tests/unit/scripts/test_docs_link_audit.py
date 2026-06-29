from pathlib import Path

from scripts.maintenance import docs_link_audit


def test_iter_markdown_files_ignores_review_artifacts_tmp(tmp_path: Path) -> None:
    docs_file = tmp_path / "docs" / "README.md"
    scratch_file = tmp_path / "review_artifacts" / "tmp" / "scratch.md"
    durable_artifact_file = tmp_path / "review_artifacts" / "summary.md"

    docs_file.parent.mkdir()
    docs_file.write_text("# docs\n")
    scratch_file.parent.mkdir(parents=True)
    scratch_file.write_text("[broken](missing.md)\n")
    durable_artifact_file.write_text("# review\n")

    markdown_files = {
        path.relative_to(tmp_path) for path in docs_link_audit.iter_markdown_files(tmp_path)
    }

    assert Path("docs/README.md") in markdown_files
    assert Path("review_artifacts/summary.md") in markdown_files
    assert Path("review_artifacts/tmp/scratch.md") not in markdown_files
