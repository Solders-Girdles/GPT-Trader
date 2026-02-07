from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.agents import regenerate_all


def test_run_generator_clears_output_dir_before_generation(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path
    output_dir = "schemas"
    target_dir = output_root / output_dir
    target_dir.mkdir(parents=True)
    orphan_path = target_dir / "obsolete.json"
    orphan_path.write_text("{}\n", encoding="utf-8")

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        generated_path = target_dir / "generated.json"
        generated_path.write_text('{"fresh":true}\n', encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(regenerate_all.subprocess, "run", fake_run)

    result = regenerate_all.run_generator(
        "generate_config_schemas.py",
        output_dir,
        output_root,
    )

    assert result.success is True
    assert not orphan_path.exists()
    assert (target_dir / "generated.json").exists()
