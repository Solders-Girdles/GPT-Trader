"""Utilities for inspecting import performance and module footprint."""

from __future__ import annotations

import sys
import time
from typing import Any


def get_import_stats() -> dict[str, int]:
    """Return simple statistics about the currently loaded modules."""
    loaded_modules = list(sys.modules.keys())
    stdlib_modules: list[str] = []
    third_party_modules: list[str] = []
    local_modules: list[str] = []

    for module_name in loaded_modules:
        if module_name.startswith("_") or "." not in module_name:
            if module_name in sys.builtin_module_names:
                stdlib_modules.append(module_name)
        elif module_name.startswith(("bot_v2", "tests")):
            local_modules.append(module_name)
        else:
            third_party_modules.append(module_name)

    return {
        "total_modules": len(loaded_modules),
        "stdlib_modules": len(stdlib_modules),
        "third_party_modules": len(third_party_modules),
        "local_modules": len(local_modules),
        "memory_usage": sys.getsizeof(sys.modules),
    }


def optimize_imports() -> None:
    """Placeholder for future import optimization routines."""
    # Potential enhancements:
    # - Detect unused modules
    # - Unload seldom-used heavy modules
    # - Compile .pyc files eagerly
    return None


class ImportProfiler:
    """Profile import wall clock times for debugging slow startup."""

    def __init__(self) -> None:
        self.import_times: dict[str, float] = {}
        self._original_import = __builtins__["__import__"]  # type: ignore[index]

    def start_profiling(self) -> None:
        __builtins__["__import__"] = self._profiled_import  # type: ignore[index]

    def stop_profiling(self) -> None:
        __builtins__["__import__"] = self._original_import  # type: ignore[index]

    def _profiled_import(
        self,
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = None,
        level: int = 0,
    ) -> Any:
        start_time = time.time()
        try:
            return self._original_import(name, globals, locals, fromlist, level)
        finally:
            self.import_times[name] = time.time() - start_time

    def get_slow_imports(self, threshold: float = 0.1) -> list[tuple[str, float]]:
        return [
            (name, duration) for name, duration in self.import_times.items() if duration > threshold
        ]

    def print_report(self, threshold: float = 0.1) -> None:
        print("Import Performance Report")
        print("=" * 40)
        slow_imports = self.get_slow_imports(threshold)
        if slow_imports:
            print(f"Slow imports (> {threshold}s):")
            for name, duration in sorted(slow_imports, key=lambda item: item[1], reverse=True):
                print(f"  {name}: {duration:.3f}s")
        else:
            print(f"No imports slower than {threshold}s")

        print(f"\nTotal imports profiled: {len(self.import_times)}")
        if self.import_times:
            total_time = sum(self.import_times.values())
            avg_time = total_time / len(self.import_times)
            print(f"Total import time: {total_time:.3f}s")
            print(f"Average import time: {avg_time:.3f}s")


__all__ = ["ImportProfiler", "get_import_stats", "optimize_imports"]
