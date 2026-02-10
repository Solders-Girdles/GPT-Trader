"""Strategy profile diff helpers."""

from __future__ import annotations

from typing import Any, Iterable, Literal, TypedDict

ProfileDiffStatus = Literal["changed", "unchanged", "missing"]


class ProfileDiffEntry(TypedDict):
    """Diff entry describing one field comparison."""

    path: str
    status: ProfileDiffStatus
    baseline_value: Any
    runtime_value: Any | None


DEFAULT_IGNORED_FIELDS: frozenset[str] = frozenset({"created_at"})


_MISSING = object()


def compute_profile_diff(
    baseline: dict[str, Any],
    runtime: dict[str, Any],
    *,
    ignore_fields: Iterable[str] | None = None,
) -> list[ProfileDiffEntry]:
    """Compare baseline profile data against runtime values.

    Args:
        baseline: Canonical profile dictionary.
        runtime: Active runtime profile dictionary.
        ignore_fields: Field names that should not appear in the diff (e.g., created_at).

    Returns:
        Ordered list of diff entries (sorted by path).
    """
    if ignore_fields is None:
        ignore_fields_set = set(DEFAULT_IGNORED_FIELDS)
    else:
        ignore_fields_set = set(ignore_fields)

    entries: list[ProfileDiffEntry] = []
    _compare_dict(baseline, runtime, "", entries, ignore_fields_set)
    return entries


def _compare_dict(
    baseline: dict[str, Any],
    runtime: dict[str, Any],
    path: str,
    entries: list[ProfileDiffEntry],
    ignore_fields: set[str],
) -> None:
    """Recursively compare nested dictionaries."""
    all_keys = sorted(set(baseline.keys()) | set(runtime.keys()))
    for key in all_keys:
        key_path = f"{path}.{key}" if path else key
        if _is_ignored(key_path, ignore_fields):
            continue
        baseline_value = baseline.get(key, _MISSING)
        runtime_value = runtime.get(key, _MISSING)
        _compare_value(baseline_value, runtime_value, key_path, entries, ignore_fields)


def _compare_value(
    baseline_value: Any,
    runtime_value: Any | object,
    path: str,
    entries: list[ProfileDiffEntry],
    ignore_fields: set[str],
) -> None:
    """Compare a single value or nested structure."""
    if _is_ignored(path, ignore_fields):
        return

    if isinstance(baseline_value, dict) and isinstance(runtime_value, dict):
        _compare_dict(baseline_value, runtime_value, path, entries, ignore_fields)
        return

    if baseline_value is _MISSING:
        entries.append(
            {
                "path": path,
                "status": "changed",
                "baseline_value": None,
                "runtime_value": runtime_value,
            }
        )
        return

    if runtime_value is _MISSING:
        entries.append(
            {
                "path": path,
                "status": "missing",
                "baseline_value": baseline_value,
                "runtime_value": None,
            }
        )
        return

    status: ProfileDiffStatus = (
        "unchanged" if _values_equal(baseline_value, runtime_value) else "changed"
    )
    entries.append(
        {
            "path": path,
            "status": status,
            "baseline_value": baseline_value,
            "runtime_value": runtime_value,
        }
    )


def _values_equal(left: Any, right: Any) -> bool:
    """Determine equality while normalizing nested dictionaries."""
    return _normalize(left) == _normalize(right)


def _normalize(value: Any) -> Any:
    """Recursively normalize dicts/lists for stable comparisons."""
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _is_ignored(path: str, ignore_fields: set[str]) -> bool:
    """Ignore any field whose final component is listed."""
    if not path:
        return False
    name = path.split(".")[-1]
    return name in ignore_fields
