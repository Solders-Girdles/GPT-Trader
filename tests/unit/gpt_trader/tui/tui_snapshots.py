from __future__ import annotations

import asyncio
import inspect
import pickle  # noqa: S403 - Required for pytest-textual-snapshot report format
import re
from collections.abc import Awaitable, Callable, Iterable
from importlib.util import module_from_spec, spec_from_file_location  # naming: allow
from pathlib import Path
from typing import Any

import pytest
import pytest_textual_snapshot as pts
from _pytest.fixtures import FixtureRequest
from rich.console import Console
from syrupy import SnapshotAssertion
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from textual.app import App
from textual.pilot import Pilot

# Patterns to normalize for stable snapshots
_TERMINAL_HASH_RE = re.compile(r"terminal-\d+-")
# Normalize timing values like "(157ms)" to "(XXms)" for API connectivity checks
_TIMING_RE = re.compile(r"\((\d+)ms\)")
# Normalize UTC time display like "19:22&#160;UTC" to "XX:XX&#160;UTC"
_UTC_TIME_RE = re.compile(r"\d{2}:\d{2}(&#160;| )UTC")
# Normalize local timestamps like "01-15 12:30:45"
_LOCAL_DATETIME_RE = re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b")
# Normalize local time display like "12:30:45"
_LOCAL_TIME_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
# Normalize braille spinner characters to a fixed state (spinner animation frames)
_BRAILLE_SPINNER_RE = re.compile(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")


def _normalize_svg(text: str) -> str:
    """Apply all normalizations to SVG snapshot content."""
    endswith_newline = text.endswith("\n")
    text = _TERMINAL_HASH_RE.sub("terminal-", text)
    text = _TIMING_RE.sub("(XXms)", text)
    text = _UTC_TIME_RE.sub("XX:XX UTC", text)
    text = _LOCAL_DATETIME_RE.sub("XX-XX XX:XX:XX", text)
    text = _LOCAL_TIME_RE.sub("XX:XX:XX", text)
    text = _BRAILLE_SPINNER_RE.sub("⠋", text)  # Normalize to first spinner frame
    lines = [line.rstrip() for line in text.splitlines()]
    lines = ["" if line.strip() == "" else line for line in lines]
    normalized = "\n".join(lines)
    if endswith_newline:
        normalized += "\n"
    return normalized


class NormalizedSVGImageExtension(SingleFileSnapshotExtension):
    """SVG extension that normalizes Textual's hashed terminal CSS class names."""

    _file_extension = "svg"
    _write_mode = WriteMode.TEXT

    def serialize(self, data, *, exclude=None, include=None, matcher=None):
        """Normalize volatile content before serializing."""
        text = super().serialize(data, exclude=exclude, include=include, matcher=matcher)
        return _normalize_svg(text)

    def read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ):
        data = super().read_snapshot_data_from_location(
            snapshot_location=snapshot_location,
            snapshot_name=snapshot_name,
            session_id=session_id,
        )
        if isinstance(data, str):
            return _normalize_svg(data)
        return data


def _import_app_from_path(app_path: str) -> App[Any]:
    """Import an App class from a file path using public stdlib APIs.

    This replaces textual._import_app.import_app with standard library equivalents.
    """
    path = Path(app_path)
    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {app_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the App subclass in the module
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, App) and obj is not App:
            return obj()

    raise ImportError(f"No App subclass found in {app_path}")


async def _capture_screenshot(
    app: App[Any],
    press: Iterable[str],
    terminal_size: tuple[int, int],
    run_before: Callable[[Pilot], Awaitable[None] | None] | None,
) -> str:
    """Capture an SVG screenshot using public Textual APIs.

    This replaces textual._doc.take_svg_screenshot with run_test() + export_screenshot().
    """
    async with app.run_test(size=terminal_size) as pilot:
        # Run any setup code
        if run_before is not None:
            result = run_before(pilot)
            if asyncio.iscoroutine(result):
                await result

        # Apply key presses
        for key in press:
            if key == "_":
                await pilot.pause()
            else:
                await pilot.press(key)

        # Small pause to let UI settle
        await pilot.pause()

        # Capture screenshot using public API
        return app.export_screenshot()


@pytest.fixture
def snap_compare(
    snapshot: SnapshotAssertion, request: FixtureRequest
) -> Callable[[str | Path | App[Any]], bool]:
    """
    Snapshot comparison fixture with normalization for Textual's hashed CSS classes.

    This overrides pytest-textual-snapshot's snap_compare to strip the session-specific
    terminal-<digits>- prefix from SVG output, ensuring snapshots remain stable across runs.

    Uses only public Textual APIs (no private _doc or _import_app imports).
    """
    # Use our normalized extension
    snapshot = snapshot.use_extension(NormalizedSVGImageExtension)

    def compare(
        app: str | Path | App[Any],
        press: Iterable[str] = (),
        terminal_size: tuple[int, int] = (80, 24),
        run_before: Callable[[Pilot], Awaitable[None] | None] | None = None,
    ) -> bool:
        """Compare a screenshot with normalized terminal class names."""
        node = request.node

        if isinstance(app, App):
            app_instance = app
            app_path = ""
        else:
            path = Path(app)
            if path.is_absolute():
                app_path = str(path.resolve())
            else:
                node_path = node.path.parent
                resolved = (node_path / app).resolve()
                app_path = str(resolved)
            app_instance = _import_app_from_path(app_path)

        # Capture screenshot using public APIs
        actual_screenshot = asyncio.run(
            _capture_screenshot(app_instance, press, terminal_size, run_before)
        )

        # Normalize the actual screenshot
        actual_screenshot = _normalize_svg(actual_screenshot)

        console = Console(legacy_windows=False, force_terminal=True)
        p_app = pts.PseudoApp(pts.PseudoConsole(console.legacy_windows, console.size))

        result = snapshot == actual_screenshot

        # Store data for report generation (matches plugin behavior)
        execution_index = (
            snapshot._custom_index and snapshot._execution_name_index.get(snapshot._custom_index)
        ) or snapshot.num_executions - 1
        assertion_result = snapshot.executions.get(execution_index)

        snapshot_exists = (
            execution_index in snapshot.executions
            and assertion_result
            and assertion_result.final_data is not None
        )

        expected_svg_text = str(snapshot)
        full_path, line_number, name = node.reportinfo()

        data = (
            result,
            expected_svg_text,
            actual_screenshot,
            p_app,
            full_path,
            line_number,
            name,
            inspect.getdoc(node.function) or "",
            app_path,
            snapshot_exists,
        )
        data_path = pts.node_to_report_path(request.node)
        data_path.write_bytes(pickle.dumps(data))  # noqa: S301

        return result

    return compare
