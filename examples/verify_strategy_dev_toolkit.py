"""Strategy dev toolkit smoke test.

This is an example script, not part of the production runtime. It exercises the
strategy-dev "lab" helpers to ensure the API surface stays importable and wired.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from gpt_trader.features.strategy_dev.lab.parameter_grid import ParameterGrid
from gpt_trader.features.strategy_dev.lab.tracker import ExperimentTracker
from gpt_trader.features.strategy_dev.monitor.performance_monitor import PerformanceMonitor


def main() -> None:
    print("Verifying Strategy Development Toolkit...")

    # Use a temporary directory so this example is safe to run from any checkout
    # (and won't create/modify repo-local files).
    with TemporaryDirectory(prefix="gpt_trader_experiments_") as tmpdir:
        tracker = ExperimentTracker(storage_path=Path(tmpdir))
        print("ExperimentTracker created.")

        # Define parameter grid
        grid = ParameterGrid()
        grid.add_parameter("fast_period", values=[10, 20, 30])
        grid.add_parameter("slow_period", values=[50, 100, 200])
        grid.add_constraint(lambda p: p["fast_period"] < p["slow_period"])
        print("ParameterGrid defined.")

        # Create experiments from grid
        experiments = tracker.create_grid_experiments(
            name_prefix="ma_optimization",
            strategy_name="moving_average",
            parameter_grid=grid,
        )
        print(f"Created {len(experiments)} experiments.")

        # Monitor performance
        monitor = PerformanceMonitor(initial_equity=Decimal("10000"))
        monitor.add_default_alerts()
        monitor.on("alert", lambda a: print(f"Alert: {a.message}"))
        print("PerformanceMonitor initialized.")

    print("Verification complete!")


if __name__ == "__main__":
    main()
