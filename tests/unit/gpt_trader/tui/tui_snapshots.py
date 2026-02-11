from __future__ import annotations

import asyncio
import difflib
import inspect
import itertools
import pickle  # noqa: S403 - Required for pytest-textual-snapshot report format
import re
from collections.abc import Awaitable, Callable, Iterable
from importlib.util import module_from_spec, spec_from_file_location  # naming: allow
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

try:
    import pytest_textual_snapshot as pts
except ImportError:  # pragma: no cover - shim for environments without the plugin
    from tests.unit.gpt_trader.tui import pytest_textual_snapshot_stub as pts
from _pytest.fixtures import FixtureRequest
from rich.console import Console
from textual.app import App
from textual.pilot import Pilot

try:
    from syrupy import SnapshotAssertion
    from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

    _SYRUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - not available everywhere
    SnapshotAssertion = None  # type: ignore[assignment]
    SingleFileSnapshotExtension = None
    WriteMode = None
    _SYRUPY_AVAILABLE = False

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


if (
    SingleFileSnapshotExtension is not None
):  # pragma: no cover - only available when syrupy is installed

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

else:

    class NormalizedSVGImageExtension:
        """Simple stand-in when syrupy is unavailable."""

        def serialize(self, data, *, exclude=None, include=None, matcher=None):
            return _normalize_svg(data)

        def read_snapshot_data_from_location(
            self, *, snapshot_location: str, snapshot_name: str, session_id: str
        ):
            return ""


def _sanitize_snapshot_component(value: str | None) -> str:
    if not value:
        return "snapshot"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return sanitized or "snapshot"


def _snapshot_file_path(request: FixtureRequest) -> Path:
    module_path = Path(str(request.node.fspath))
    snapshots_dir = module_path.parent / "__snapshots__" / module_path.stem
    nodeid_parts = str(request.node.nodeid).split("::")
    function_name = _sanitize_snapshot_component(nodeid_parts[-1])
    class_name = _sanitize_snapshot_component(nodeid_parts[-2]) if len(nodeid_parts) > 1 else None
    file_name = f"{class_name}.{function_name}.raw" if class_name else f"{function_name}.raw"
    return snapshots_dir / file_name


class LocalSnapshotAssertion:
    def __init__(self, request: FixtureRequest) -> None:
        self._request = request
        self._expected: str | None = None
        self._custom_index = None
        self._execution_name_index: dict[str, int] = {}
        self.executions: dict[int, SimpleNamespace] = {}
        self.num_executions = 0

    def use_extension(self, extension: type[Any]) -> "LocalSnapshotAssertion":
        return self

    def _load_expected(self) -> str:
        path = _snapshot_file_path(self._request)
        normalized = _normalize_svg(path.read_text())
        self._expected = normalized
        self.num_executions = 1
        self.executions = {0: SimpleNamespace(final_data=normalized)}
        self._custom_index = None
        self._execution_name_index = {}
        return normalized

    def __eq__(self, other: Any) -> bool:
        actual = _normalize_svg(other)
        expected = self._expected or self._load_expected()
        return expected == actual

    def __str__(self) -> str:
        return self._expected or self._load_expected()


def _get_snapshot_assertion(request: FixtureRequest) -> Any:
    if _SYRUPY_AVAILABLE:
        try:
            return request.getfixturevalue("snapshot")
        except pytest.FixtureLookupError:
            pass
    return LocalSnapshotAssertion(request)


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
def snap_compare(request: FixtureRequest) -> Callable[[str | Path | App[Any]], bool]:
    """
    Snapshot comparison fixture with normalization for Textual's hashed CSS classes.

    This overrides pytest-textual-snapshot's snap_compare to strip the session-specific
    terminal-<digits>- prefix from SVG output, ensuring snapshots remain stable across runs.

    Uses only public Textual APIs (no private _doc or _import_app imports).
    """
    snapshot = _get_snapshot_assertion(request)
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
        expected_svg_text = str(snapshot)

        if not result:
            expected_lines = expected_svg_text.splitlines()
            actual_lines = actual_screenshot.splitlines()
            diff_iter = difflib.unified_diff(
                expected_lines,
                actual_lines,
                fromfile="expected",
                tofile="actual",
                lineterm="",
                n=2,
            )
            diff_excerpt = "\n".join(itertools.islice(diff_iter, 120))
            if diff_excerpt:
                print("\nSnapshot mismatch diff (truncated):\n")
                print(diff_excerpt)

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
