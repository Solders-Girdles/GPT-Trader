from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.agents import dependency_graph


def _write_python(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sample_source_tree(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    _write_python(src / "gpt_trader" / "__init__.py")
    _write_python(src / "gpt_trader" / "app" / "__init__.py")
    _write_python(src / "gpt_trader" / "app" / "container.py", "class ApplicationContainer: ...\n")
    _write_python(src / "gpt_trader" / "features" / "__init__.py")
    _write_python(src / "gpt_trader" / "features" / "alpha" / "__init__.py")
    _write_python(
        src / "gpt_trader" / "features" / "alpha" / "service.py",
        "from gpt_trader.app.container import ApplicationContainer\n",
    )
    return src


def test_main_queries_fully_qualified_dependencies(tmp_path, monkeypatch, capsys) -> None:
    src = _sample_source_tree(tmp_path)
    monkeypatch.setattr(dependency_graph, "SRC_DIR", src)
    monkeypatch.setattr(dependency_graph, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dependency_graph.py",
            "--dependencies-of",
            "gpt_trader.features.alpha.service",
        ],
    )

    result = dependency_graph.main()

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["module"] == "gpt_trader.features.alpha.service"
    assert output["direct_dependencies"] == ["gpt_trader.app.container"]


def test_main_queries_fully_qualified_dependents(tmp_path, monkeypatch, capsys) -> None:
    src = _sample_source_tree(tmp_path)
    monkeypatch.setattr(dependency_graph, "SRC_DIR", src)
    monkeypatch.setattr(dependency_graph, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dependency_graph.py",
            "--depends-on",
            "gpt_trader.app.container",
        ],
    )

    result = dependency_graph.main()

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["module"] == "gpt_trader.app.container"
    assert output["direct_dependents"] == ["gpt_trader.features.alpha.service"]
