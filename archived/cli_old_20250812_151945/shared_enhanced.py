"""Enhanced shared utilities for the CLI with QoL improvements."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .cli_utils import (
    CLITheme,
    DataValidator,
    console,
    export_results,
)


class EnhancedDateParser:
    """Enhanced date parsing with smart defaults and validation."""

    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """Parse date string with enhanced validation."""
        try:
            return DataValidator.validate_date(date_str)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.") from e

    @staticmethod
    def parse_date_range(start: str, end: str) -> tuple[datetime, datetime]:
        """Parse and validate date range."""
        return DataValidator.validate_date_range(start, end)

    @staticmethod
    def get_default_date_range(days: int = 365) -> tuple[str, str]:
        """Get default date range for the last N days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    @staticmethod
    def format_date_for_display(date: datetime) -> str:
        """Format date for display."""
        return date.strftime("%Y-%m-%d")


class EnhancedRunDirectory:
    """Enhanced run directory management with better organization."""

    @staticmethod
    def ensure_run_dir(strategy: str, run_tag: str = "", run_dir: str = "") -> Path:
        """Create and ensure run directory exists with enhanced naming."""
        if run_dir:
            p = Path(run_dir)
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            if run_tag:
                p = Path("data/experiments") / strategy / f"{stamp}_{run_tag}"
            else:
                p = Path("data/experiments") / strategy / f"{stamp}_run"

        p.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for better organization
        (p / "logs").mkdir(exist_ok=True)
        (p / "results").mkdir(exist_ok=True)
        (p / "configs").mkdir(exist_ok=True)

        return p

    @staticmethod
    def cleanup_old_runs(directory: Path, max_age_days: int = 30) -> int:
        """Clean up old run directories."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0

        for item in directory.iterdir():
            if item.is_dir():
                try:
                    if item.stat().st_mtime < cutoff_time:
                        shutil.rmtree(item)
                        cleaned_count += 1
                        console.print(f"[dim]Cleaned up old run: {item.name}[/dim]")
                except OSError:
                    pass  # Skip if can't delete

        if cleaned_count > 0:
            console.print(CLITheme.info(f"Cleaned up {cleaned_count} old run directories"))

        return cleaned_count


class EnhancedUniverseReader:
    """Enhanced universe CSV reading with validation and error handling."""

    @staticmethod
    def read_universe_csv(path: str) -> list[str]:
        """Read universe CSV with enhanced validation."""
        try:
            file_path = DataValidator.validate_file_path(path)
            df = pd.read_csv(file_path)

            # Look for symbol column
            symbol_columns = ["symbol", "ticker", "Symbol", "Ticker", "SYMBOL", "TICKER"]
            symbol_col = None

            for col in symbol_columns:
                if col in df.columns:
                    symbol_col = col
                    break

            if symbol_col is None:
                # Try to guess from column names
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ["symbol", "ticker", "stock"]):
                        symbol_col = col
                        break

            if symbol_col is None:
                raise ValueError("Universe CSV must have a 'symbol' or 'ticker' column")

            # Extract and validate symbols
            symbols = [str(s).strip().upper() for s in df[symbol_col].dropna().unique()]
            validated_symbols = DataValidator.validate_symbols(symbols)

            console.print(
                CLITheme.success(f"Loaded {len(validated_symbols)} symbols from {file_path.name}")
            )
            return validated_symbols

        except Exception as e:
            console.print(CLITheme.error(f"Failed to read universe CSV: {e}"))
            raise

    @staticmethod
    def validate_universe_symbols(symbols: list[str]) -> list[str]:
        """Validate universe symbols and provide feedback."""
        if not symbols:
            raise ValueError("No symbols provided")

        validated = []
        invalid = []

        for symbol in symbols:
            try:
                validated.append(DataValidator.validate_symbol(symbol))
            except ValueError:
                invalid.append(symbol)

        if invalid:
            console.print(CLITheme.warning(f"Invalid symbols found: {', '.join(invalid)}"))

        if not validated:
            raise ValueError("No valid symbols found")

        return list(set(validated))  # Remove duplicates


class EnhancedBasenameComposer:
    """Enhanced basename composition with better formatting."""

    @staticmethod
    def compose_bt_basename(
        strategy: str,
        sym_label: str,
        start_s: str,
        end_s: str,
        don: int | None = None,
        atr: int | None = None,
        k: float | None = None,
        risk_pct: float | None = None,
        cadence: str = "daily",
        regime_on: bool = False,
        regime_win: int | None = None,
    ) -> str:
        """Compose backtest basename with enhanced formatting."""
        parts = [
            strategy,
            sym_label,
            f"{start_s}_{end_s}",
            f"don{don}" if don is not None else None,
            f"atr{atr}" if atr is not None else None,
            f"k{(k or 0):.2f}" if k is not None else None,
            f"r{(risk_pct or 0):.2f}" if risk_pct is not None else None,
            cadence,
            (
                f"reg{regime_win}"
                if regime_on and regime_win is not None
                else ("reg0" if regime_on else None)
            ),
        ]

        # Filter out None values and join
        filtered_parts = [p for p in parts if p is not None]
        return "__".join(filtered_parts)

    @staticmethod
    def parse_basename(basename: str) -> dict[str, Any]:
        """Parse basename back into components."""
        parts = basename.split("__")
        result = {}

        if len(parts) >= 3:
            result["strategy"] = parts[0]
            result["symbol_label"] = parts[1]
            result["date_range"] = parts[2]

        # Parse additional parameters
        for part in parts[3:]:
            if part.startswith("don"):
                result["donchian"] = int(part[3:])
            elif part.startswith("atr"):
                result["atr"] = int(part[3:])
            elif part.startswith("k"):
                result["atr_k"] = float(part[1:])
            elif part.startswith("r"):
                result["risk_pct"] = float(part[1:])
            elif part in ["daily", "weekly"]:
                result["cadence"] = part
            elif part.startswith("reg"):
                result["regime_window"] = int(part[3:])

        return result


class EnhancedFileWatcher:
    """Enhanced file watching with timeout and better error handling."""

    @staticmethod
    def wait_for_summary_since(
        before: set[str], timeout: float = 10.0, poll: float = 0.1
    ) -> Path | None:
        """Wait for summary file with enhanced timeout and feedback."""
        outdir = Path("data/backtests")
        deadline = time.time() + timeout

        with console.status(f"[dim]Waiting for results (timeout: {timeout}s)...[/dim]") as status:
            while time.time() < deadline:
                candidates = [p for p in outdir.glob("PORT_*_summary.csv") if str(p) not in before]
                if candidates:
                    newest = max(candidates, key=lambda p: p.stat().st_mtime)
                    status.update("[bold green]Results found!")
                    return newest

                time.sleep(poll)
                remaining = deadline - time.time()
                if remaining > 0:
                    status.update(f"[dim]Waiting for results... ({remaining:.1f}s remaining)[/dim]")

        console.print(CLITheme.warning(f"No summary file generated within {timeout}s"))
        return None


class EnhancedManifestManager:
    """Enhanced manifest management with better metadata."""

    @staticmethod
    def save_manifest(
        run_dir: Path, strategy: str, args: argparse.Namespace, symbols: list[str]
    ) -> None:
        """Save enhanced manifest with additional metadata."""
        manifest = {
            "strategy": strategy,
            "args": vars(args),
            "symbols": symbols,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "working_directory": str(Path.cwd()),
            },
        }

        manifest_file = run_dir / "configs" / "run.json"
        manifest_file.parent.mkdir(exist_ok=True)

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        console.print(CLITheme.info(f"Manifest saved to {manifest_file}"))

    @staticmethod
    def load_manifest(run_dir: Path) -> dict[str, Any]:
        """Load manifest from run directory."""
        manifest_file = run_dir / "configs" / "run.json"

        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_file) as f:
            return json.load(f)


class EnhancedFileMirror:
    """Enhanced file mirroring with progress tracking."""

    @staticmethod
    def mirror_backtest_triplet_from_summary(
        summary_path: Path, dest_dir: Path, base_name: str
    ) -> None:
        """Mirror backtest files with enhanced progress tracking."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = summary_path.name.replace("_summary.csv", "")
        src_dir = summary_path.parent

        suffixes = [".csv", ".png", "_summary.csv", "_trades.csv"]
        copied_files = []

        with console.status("[dim]Copying results...[/dim]") as status:
            for suffix in suffixes:
                src = src_dir / f"{stem}{suffix}"
                if src.exists():
                    dst = dest_dir / f"{base_name}{suffix}"
                    try:
                        dst.write_bytes(src.read_bytes())
                        copied_files.append(dst.name)
                        status.update(f"[dim]Copied {dst.name}...[/dim]")
                    except Exception as e:
                        console.print(CLITheme.warning(f"Failed to copy {src.name}: {e}"))

        if copied_files:
            console.print(CLITheme.success(f"Copied {len(copied_files)} files to {dest_dir}"))
            for file in copied_files:
                console.print(f"[dim]  - {file}[/dim]")


class EnhancedResultsProcessor:
    """Enhanced results processing with better formatting and export options."""

    @staticmethod
    def process_backtest_results(
        summary_path: Path, export_formats: list[str] = None
    ) -> dict[str, Any]:
        """Process backtest results with enhanced formatting and export."""
        if export_formats is None:
            export_formats = ["json"]

        try:
            df = pd.read_csv(summary_path)

            if df.empty:
                console.print(CLITheme.warning("No results found in summary file"))
                return {}

            # Extract key metrics
            results = {}
            metrics_mapping = {
                "total_return_pct": "Total Return (%)",
                "cagr_pct": "CAGR (%)",
                "sharpe_ratio": "Sharpe Ratio",
                "max_drawdown_pct": "Max Drawdown (%)",
                "win_rate_pct": "Win Rate (%)",
                "num_trades": "Total Trades",
                "avg_trade_return_pct": "Avg Trade Return (%)",
                "profit_factor": "Profit Factor",
                "calmar_ratio": "Calmar Ratio",
            }

            for col, display_name in metrics_mapping.items():
                if col in df.columns:
                    value = df[col].iloc[0]
                    results[display_name] = value

            # Export results in requested formats
            for fmt in export_formats:
                try:
                    export_results(
                        results, fmt, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                except Exception as e:
                    console.print(CLITheme.warning(f"Failed to export {fmt}: {e}"))

            return results

        except Exception as e:
            console.print(CLITheme.error(f"Failed to process results: {e}"))
            return {}

    @staticmethod
    def display_results_summary(results: dict[str, Any], title: str = "Backtest Results") -> None:
        """Display results in a formatted summary."""
        if not results:
            console.print(CLITheme.warning("No results to display"))
            return

        # Create results table
        from rich.table import Table

        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Format values based on type
        for metric, value in results.items():
            if isinstance(value, float):
                if "Return" in metric or "Rate" in metric or "Drawdown" in metric:
                    formatted_value = f"{value:.2f}%"
                elif "Ratio" in metric or "Factor" in metric:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            table.add_row(metric, formatted_value)

        console.print(table)

        # Add performance summary
        if "Total Return (%)" in results and "Max Drawdown (%)" in results:
            total_return = results["Total Return (%)"]
            max_dd = abs(results["Max Drawdown (%)"])

            if max_dd > 0:
                calmar = total_return / max_dd
                console.print(f"[dim]Calmar Ratio: {calmar:.2f}[/dim]")


# Convenience functions for backward compatibility
def _parse_date(s: str) -> datetime:
    """Backward compatibility wrapper."""
    return EnhancedDateParser.parse_date(s)


def _ensure_run_dir(strategy: str, run_tag: str = "", run_dir: str = "") -> Path:
    """Backward compatibility wrapper."""
    return EnhancedRunDirectory.ensure_run_dir(strategy, run_tag, run_dir)


def _read_universe_csv(path: str) -> list[str]:
    """Backward compatibility wrapper."""
    return EnhancedUniverseReader.read_universe_csv(path)


def _guesstimate_universe_label(symbol: str | None, symbol_list: str | None) -> str:
    """Backward compatibility wrapper."""
    if symbol:
        return symbol.upper()
    if symbol_list:
        return Path(symbol_list).stem
    return "UNKN"


def _compose_bt_basename(
    strategy: str,
    sym_label: str,
    start_s: str,
    end_s: str,
    don: int | None = None,
    atr: int | None = None,
    k: float | None = None,
    risk_pct: float | None = None,
    cadence: str = "daily",
    regime_on: bool = False,
    regime_win: int | None = None,
) -> str:
    """Backward compatibility wrapper."""
    return EnhancedBasenameComposer.compose_bt_basename(
        strategy, sym_label, start_s, end_s, don, atr, k, risk_pct, cadence, regime_on, regime_win
    )


def _wait_for_summary_since(
    before: set[str], timeout: float = 3.0, poll: float = 0.05
) -> Path | None:
    """Backward compatibility wrapper."""
    return EnhancedFileWatcher.wait_for_summary_since(before, timeout, poll)


def _save_manifest(
    run_dir: Path, strategy: str, args: argparse.Namespace, symbols: list[str]
) -> None:
    """Backward compatibility wrapper."""
    EnhancedManifestManager.save_manifest(run_dir, strategy, args, symbols)


def _mirror_backtest_triplet_from_summary(
    summary_path: Path, dest_dir: Path, base_name: str
) -> None:
    """Backward compatibility wrapper."""
    EnhancedFileMirror.mirror_backtest_triplet_from_summary(summary_path, dest_dir, base_name)
