"""
Dataset Preparation CLI for GPT-Trader

Integrates Historical Data Manager and Data Quality Framework to prepare
clean, validated training datasets for strategy development.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click
import pandas as pd
from bot.dataflow.data_quality_framework import create_data_quality_framework
from bot.dataflow.historical_data_manager import DataFrequency, create_historical_data_manager
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


# Common symbol universes
UNIVERSE_SP500_TOP100 = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "BRK-B",
    "UNH",
    "JNJ",
    "V",
    "PG",
    "JPM",
    "HD",
    "CVX",
    "MA",
    "PFE",
    "ABBV",
    "BAC",
    "KO",
    "AVGO",
    "PEP",
    "TMO",
    "COST",
    "DIS",
    "ABT",
    "WMT",
    "CRM",
    "LIN",
    "ACN",
    "VZ",
    "ADBE",
    "MCD",
    "CMCSA",
    "DHR",
    "NFLX",
    "NKE",
    "TXN",
    "NEE",
    "BMY",
    "UPS",
    "PM",
    "RTX",
    "T",
    "QCOM",
    "LOW",
    "SCHW",
    "HON",
    "ELV",
    "IBM",
    "COP",
    "SPGI",
    "LMT",
    "ORCL",
    "GS",
    "BLK",
    "MDT",
    "CAT",
    "AXP",
    "ISRG",
    "GILD",
    "BA",
    "TMUS",
    "SYK",
    "INTU",
    "BKNG",
    "C",
    "MU",
    "GE",
    "DE",
    "ADP",
    "TJX",
    "ZTS",
    "CVS",
    "MMM",
    "VRTX",
    "NOC",
    "AMT",
    "AMD",
    "INTC",
    "PLD",
    "MO",
    "SO",
    "WM",
    "CI",
    "REGN",
    "ICE",
    "DUK",
    "FDX",
    "SHW",
    "CB",
    "PNC",
    "CL",
    "BSX",
    "ITW",
    "PYPL",
    "AON",
    "GD",
    "EMR",
    "TGT",
]

UNIVERSE_TECH_FOCUS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "CRM",
    "ADBE",
    "NFLX",
    "INTC",
    "AMD",
    "QCOM",
    "TXN",
    "AVGO",
    "ORCL",
    "IBM",
    "CSCO",
    "UBER",
    "SNAP",
    "TWTR",
    "SPOT",
    "ZM",
    "DOCU",
    "CRWD",
    "SNOW",
    "PLTR",
    "RBLX",
    "COIN",
    "SQ",
]

UNIVERSE_DIVERSIFIED = [
    # Tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "CRM",
    "ADBE",
    "NFLX",
    # Finance
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "USB",
    "TFC",
    "PNC",
    "COF",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    "ABT",
    "TMO",
    "MRK",
    "DHR",
    "BMY",
    "AMGN",
    # Consumer
    "WMT",
    "HD",
    "PG",
    "KO",
    "PEP",
    "MCD",
    "DIS",
    "NKE",
    "SBUX",
    "TGT",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
    "MPC",
    "VLO",
    "PSX",
    "HES",
    "DVN",
    # Industrial
    "BA",
    "GE",
    "CAT",
    "MMM",
    "HON",
    "RTX",
    "LMT",
    "UPS",
    "FDX",
    "DE",
]

UNIVERSES = {
    "sp500_top100": UNIVERSE_SP500_TOP100,
    "tech_focus": UNIVERSE_TECH_FOCUS,
    "diversified": UNIVERSE_DIVERSIFIED,
}


class DatasetPreparationError(Exception):
    """Custom exception for dataset preparation errors"""

    pass


class DatasetPreparer:
    """Main class for preparing training datasets"""

    def __init__(self, output_dir: Path, min_quality_score: float = 75.0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_manager = create_historical_data_manager(
            min_quality_score=min_quality_score,
            cache_dir=str(self.output_dir / "cache"),
            max_concurrent_downloads=5,
        )

        self.quality_framework = create_data_quality_framework(
            min_quality_score=min_quality_score, outlier_method="iqr", missing_data_method="forward"
        )

        # Results tracking
        self.preparation_results = {}

    def prepare_datasets(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY,
        apply_cleaning: bool = True,
    ) -> dict[str, Any]:
        """Prepare datasets for multiple symbols"""

        console.print("\n🚀 [bold blue]Dataset Preparation Started[/bold blue]")
        console.print(f"   Symbols: {len(symbols)} symbols")
        console.print(f"   Date Range: {start_date.date()} to {end_date.date()}")
        console.print(f"   Frequency: {frequency.value}")
        console.print(f"   Output Directory: {self.output_dir}")

        start_time = datetime.now()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            # Step 1: Download historical data
            download_task = progress.add_task(
                "📡 Downloading historical data...", total=len(symbols)
            )

            datasets, metadata = self.data_manager.get_training_dataset(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                force_refresh=False,
            )

            progress.update(download_task, completed=len(symbols))

            # Step 2: Quality assessment and cleaning
            if apply_cleaning:
                quality_task = progress.add_task(
                    "🧹 Quality assessment & cleaning...", total=len(datasets)
                )

                cleaned_datasets = {}
                quality_reports = {}

                for symbol in datasets:
                    try:
                        raw_data = datasets[symbol]
                        cleaned_data, quality_report = self.quality_framework.clean_and_validate(
                            raw_data, symbol
                        )

                        cleaned_datasets[symbol] = cleaned_data
                        quality_reports[symbol] = quality_report

                        progress.update(quality_task, advance=1)

                    except Exception as e:
                        logger.error(f"Failed to clean data for {symbol}: {str(e)}")
                        continue

                datasets = cleaned_datasets

            else:
                # Just assess quality without cleaning
                quality_task = progress.add_task(
                    "📊 Assessing data quality...", total=len(datasets)
                )
                quality_reports = {}

                for symbol in datasets:
                    try:
                        quality_report = self.quality_framework.assess_quality(
                            datasets[symbol], symbol
                        )
                        quality_reports[symbol] = quality_report
                        progress.update(quality_task, advance=1)
                    except Exception as e:
                        logger.error(f"Failed to assess quality for {symbol}: {str(e)}")
                        continue

            # Step 3: Save datasets and metadata
            save_task = progress.add_task(
                "💾 Saving datasets and reports...", total=len(datasets) + 2
            )

            saved_datasets = self._save_datasets(datasets, progress, save_task)
            self._save_quality_reports(quality_reports, progress, save_task)
            self._save_preparation_metadata(
                metadata,
                quality_reports,
                start_date,
                end_date,
                frequency,
                apply_cleaning,
                progress,
                save_task,
            )

        # Generate summary
        end_time = datetime.now()
        duration = end_time - start_time

        summary = self._generate_preparation_summary(
            symbols, datasets, quality_reports, saved_datasets, duration
        )

        self._display_results(summary)

        return summary

    def _save_datasets(
        self, datasets: dict[str, pd.DataFrame], progress: Progress, task_id: Any
    ) -> dict[str, str]:
        """Save datasets to CSV files"""
        saved_datasets = {}

        datasets_dir = self.output_dir / "datasets"
        datasets_dir.mkdir(exist_ok=True)

        for symbol, data in datasets.items():
            try:
                filename = f"{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = datasets_dir / filename

                # Add metadata columns if not present
                if "Symbol" not in data.columns:
                    data["Symbol"] = symbol

                data.to_csv(filepath, index=True)
                saved_datasets[symbol] = str(filepath)

            except Exception as e:
                logger.error(f"Failed to save dataset for {symbol}: {str(e)}")
                continue

        progress.update(task_id, advance=1)
        return saved_datasets

    def _save_quality_reports(
        self, quality_reports: dict[str, Any], progress: Progress, task_id: Any
    ) -> None:
        """Save quality reports to JSON files"""
        reports_dir = self.output_dir / "quality_reports"
        reports_dir.mkdir(exist_ok=True)

        # Save individual reports
        for symbol, report in quality_reports.items():
            try:
                filename = f"{symbol}_quality_{datetime.now().strftime('%Y%m%d')}.json"
                filepath = reports_dir / filename

                # Convert report to dict for JSON serialization
                report_dict = {
                    "symbol": report.symbol,
                    "total_records": report.total_records,
                    "date_range": [
                        report.date_range[0].isoformat(),
                        report.date_range[1].isoformat(),
                    ],
                    "quality_score": report.quality_score,
                    "is_usable": report.is_usable,
                    "issues": [
                        {
                            "type": issue.issue_type.value,
                            "severity": issue.severity.value,
                            "description": issue.description,
                            "affected_rows": issue.affected_rows,
                            "percentage": issue.percentage,
                            "suggested_action": issue.suggested_action,
                        }
                        for issue in report.issues
                    ],
                    "cleaning_applied": report.cleaning_applied,
                    "recommendations": report.recommendations,
                    "metadata": report.metadata,
                }

                with open(filepath, "w") as f:
                    json.dump(report_dict, f, indent=2, default=str)

            except Exception as e:
                logger.error(f"Failed to save quality report for {symbol}: {str(e)}")
                continue

        # Save summary report
        try:
            summary_report = self.quality_framework.generate_quality_summary(quality_reports)
            summary_filepath = (
                reports_dir / f"quality_summary_{datetime.now().strftime('%Y%m%d')}.json"
            )

            with open(summary_filepath, "w") as f:
                json.dump(summary_report, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save quality summary: {str(e)}")

        progress.update(task_id, advance=1)

    def _save_preparation_metadata(
        self,
        original_metadata: Any,
        quality_reports: dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency,
        apply_cleaning: bool,
        progress: Progress,
        task_id: Any,
    ) -> None:
        """Save preparation metadata"""
        try:
            metadata = {
                "preparation_timestamp": datetime.now().isoformat(),
                "parameters": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "frequency": frequency.value,
                    "apply_cleaning": apply_cleaning,
                },
                "results": {
                    "total_symbols_requested": len(quality_reports) if quality_reports else 0,
                    "successful_symbols": (
                        len([r for r in quality_reports.values() if r.is_usable])
                        if quality_reports
                        else 0
                    ),
                    "average_quality_score": (
                        sum(r.quality_score for r in quality_reports.values())
                        / len(quality_reports)
                        if quality_reports
                        else 0
                    ),
                    "usable_datasets": (
                        len([r for r in quality_reports.values() if r.is_usable])
                        if quality_reports
                        else 0
                    ),
                },
                "data_sources": (
                    original_metadata.sources if hasattr(original_metadata, "sources") else []
                ),
                "cache_info": self.data_manager.get_cache_info(),
                "output_directory": str(self.output_dir),
            }

            metadata_filepath = (
                self.output_dir
                / f"preparation_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(metadata_filepath, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save preparation metadata: {str(e)}")

        progress.update(task_id, advance=1)

    def _generate_preparation_summary(
        self,
        requested_symbols: list[str],
        datasets: dict[str, pd.DataFrame],
        quality_reports: dict[str, Any],
        saved_datasets: dict[str, str],
        duration: timedelta,
    ) -> dict[str, Any]:
        """Generate preparation summary"""

        usable_datasets = [symbol for symbol, report in quality_reports.items() if report.is_usable]
        quality_scores = [report.quality_score for report in quality_reports.values()]

        return {
            "summary": {
                "requested_symbols": len(requested_symbols),
                "downloaded_datasets": len(datasets),
                "usable_datasets": len(usable_datasets),
                "success_rate": (len(usable_datasets) / len(requested_symbols)) * 100,
                "average_quality_score": (
                    sum(quality_scores) / len(quality_scores) if quality_scores else 0
                ),
                "preparation_time": str(duration),
                "output_directory": str(self.output_dir),
            },
            "datasets": {
                "usable_symbols": usable_datasets,
                "saved_files": saved_datasets,
                "total_records": sum(len(data) for data in datasets.values()),
                "average_records_per_symbol": (
                    sum(len(data) for data in datasets.values()) / len(datasets) if datasets else 0
                ),
            },
            "quality": {
                "quality_distribution": {
                    "excellent (90-100)": len([s for s in quality_scores if s >= 90]),
                    "good (80-89)": len([s for s in quality_scores if 80 <= s < 90]),
                    "moderate (70-79)": len([s for s in quality_scores if 70 <= s < 80]),
                    "poor (60-69)": len([s for s in quality_scores if 60 <= s < 70]),
                    "critical (<60)": len([s for s in quality_scores if s < 60]),
                },
                "reports_saved": len(quality_reports),
            },
            "recommendations": self._generate_summary_recommendations(
                quality_reports, usable_datasets, requested_symbols
            ),
        }

    def _generate_summary_recommendations(
        self,
        quality_reports: dict[str, Any],
        usable_datasets: list[str],
        requested_symbols: list[str],
    ) -> list[str]:
        """Generate recommendations based on preparation results"""
        recommendations = []

        success_rate = (len(usable_datasets) / len(requested_symbols)) * 100

        if success_rate < 50:
            recommendations.append(
                "Low success rate - consider reviewing data sources or relaxing quality thresholds"
            )
        elif success_rate < 80:
            recommendations.append(
                "Moderate success rate - some symbols may need alternative data sources"
            )
        else:
            recommendations.append("High success rate - datasets ready for strategy training")

        if quality_reports:
            avg_quality = sum(r.quality_score for r in quality_reports.values()) / len(
                quality_reports
            )
            if avg_quality < 70:
                recommendations.append(
                    "Average quality below threshold - consider additional data cleaning"
                )
            elif avg_quality > 90:
                recommendations.append(
                    "Excellent data quality - suitable for production strategy training"
                )

        recommendations.append(
            f"Ready for strategy training with {len(usable_datasets)} high-quality datasets"
        )
        recommendations.append("Monitor data quality over time and refresh datasets periodically")

        return recommendations

    def _display_results(self, summary: dict[str, Any]) -> None:
        """Display preparation results in a nice format"""

        console.print("\n" + "=" * 60)
        console.print("🎯 [bold green]DATASET PREPARATION COMPLETE[/bold green]")
        console.print("=" * 60)

        # Summary table
        summary_table = Table(
            title="📊 Preparation Summary", show_header=True, header_style="bold magenta"
        )
        summary_table.add_column("Metric", style="cyan", width=25)
        summary_table.add_column("Value", style="white", width=15)
        summary_table.add_column("Details", style="dim", width=15)

        summary_data = summary["summary"]
        summary_table.add_row("Symbols Requested", str(summary_data["requested_symbols"]), "")
        summary_table.add_row("Datasets Downloaded", str(summary_data["downloaded_datasets"]), "")
        summary_table.add_row(
            "Usable Datasets",
            str(summary_data["usable_datasets"]),
            f"{summary_data['success_rate']:.1f}% success",
        )
        summary_table.add_row(
            "Avg Quality Score", f"{summary_data['average_quality_score']:.1f}/100", ""
        )
        summary_table.add_row("Preparation Time", summary_data["preparation_time"], "")

        console.print(summary_table)

        # Quality distribution
        console.print("\n📈 [bold blue]Quality Distribution[/bold blue]")
        quality_dist = summary["quality"]["quality_distribution"]
        for quality_range, count in quality_dist.items():
            if count > 0:
                console.print(f"   • {quality_range}: {count} datasets")

        # Recommendations
        console.print("\n💡 [bold yellow]Recommendations[/bold yellow]")
        for rec in summary["recommendations"]:
            console.print(f"   • {rec}")

        console.print(f"\n📁 [bold]Output Directory:[/bold] {summary_data['output_directory']}")
        console.print("\n✅ [bold green]Datasets ready for strategy training![/bold green]\n")


# CLI Commands
@click.group()
def prepare() -> None:
    """Dataset preparation commands"""
    pass


@prepare.command()
@click.option(
    "--universe",
    type=click.Choice(["sp500_top100", "tech_focus", "diversified", "custom"]),
    default="sp500_top100",
    help="Symbol universe to prepare",
)
@click.option(
    "--symbols", type=str, help="Custom comma-separated list of symbols (when universe=custom)"
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=(datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d"),
    help="Start date for data (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=datetime.now().strftime("%Y-%m-%d"),
    help="End date for data (YYYY-MM-DD)",
)
@click.option(
    "--frequency", type=click.Choice(["1d", "1h", "1m"]), default="1d", help="Data frequency"
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/training_datasets",
    help="Output directory for prepared datasets",
)
@click.option(
    "--min-quality-score",
    type=float,
    default=75.0,
    help="Minimum quality score for usable datasets",
)
@click.option("--apply-cleaning/--no-cleaning", default=True, help="Apply automated data cleaning")
@click.option("--force-refresh", is_flag=True, default=False, help="Force refresh cached data")
def datasets(
    universe,
    symbols,
    start_date,
    end_date,
    frequency,
    output_dir,
    min_quality_score,
    apply_cleaning,
    force_refresh,
) -> None:
    """Prepare training datasets for strategy development"""

    # Parse symbols
    if universe == "custom":
        if not symbols:
            raise click.ClickException(
                "Custom symbols must be provided with --symbols when using custom universe"
            )
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = UNIVERSES[universe]

    # Parse frequency
    freq_map = {"1d": DataFrequency.DAILY, "1h": DataFrequency.HOURLY, "1m": DataFrequency.MINUTE}
    data_frequency = freq_map[frequency]

    console.print(
        Panel(
            f"[bold blue]Dataset Preparation Configuration[/bold blue]\n"
            f"Universe: {universe} ({len(symbol_list)} symbols)\n"
            f"Date Range: {start_date.date()} to {end_date.date()}\n"
            f"Frequency: {frequency}\n"
            f"Min Quality Score: {min_quality_score}\n"
            f"Apply Cleaning: {apply_cleaning}\n"
            f"Output Directory: {output_dir}",
            title="🚀 Starting Dataset Preparation",
        )
    )

    try:
        # Initialize preparer
        preparer = DatasetPreparer(output_dir=Path(output_dir), min_quality_score=min_quality_score)

        # Force refresh cache if requested
        if force_refresh:
            console.print("🔄 [yellow]Clearing cache to force fresh data download...[/yellow]")
            preparer.data_manager.clear_cache()

        # Prepare datasets
        summary = preparer.prepare_datasets(
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            frequency=data_frequency,
            apply_cleaning=apply_cleaning,
        )

        # Success message
        usable_count = summary["summary"]["usable_datasets"]
        console.print(
            f"\n🎉 [bold green]Successfully prepared {usable_count} usable datasets![/bold green]"
        )

    except Exception as e:
        console.print(f"\n❌ [bold red]Dataset preparation failed:[/bold red] {str(e)}")
        raise click.ClickException(str(e))


@prepare.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/training_datasets",
    help="Dataset directory to analyze",
)
def status(output_dir) -> None:
    """Show status of prepared datasets"""

    output_path = Path(output_dir)
    if not output_path.exists():
        console.print(f"❌ [red]Output directory does not exist: {output_dir}[/red]")
        return

    # Find dataset files
    datasets_dir = output_path / "datasets"
    quality_dir = output_path / "quality_reports"

    if datasets_dir.exists():
        dataset_files = list(datasets_dir.glob("*.csv"))
        console.print(f"📁 Found {len(dataset_files)} dataset files")

        if dataset_files:
            table = Table(title="📊 Available Datasets")
            table.add_column("Symbol", style="cyan")
            table.add_column("File", style="white")
            table.add_column("Size", style="dim")
            table.add_column("Modified", style="dim")

            for file in sorted(dataset_files)[:20]:  # Show first 20
                try:
                    symbol = file.stem.split("_")[0]
                    size_mb = file.stat().st_size / (1024 * 1024)
                    modified = datetime.fromtimestamp(file.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    )

                    table.add_row(symbol, file.name, f"{size_mb:.1f} MB", modified)
                except Exception:
                    continue

            console.print(table)

    if quality_dir.exists():
        quality_files = list(quality_dir.glob("quality_summary_*.json"))
        if quality_files:
            latest_summary = max(quality_files, key=lambda f: f.stat().st_mtime)
            console.print(f"\n📊 [bold]Latest Quality Summary:[/bold] {latest_summary.name}")

            try:
                with open(latest_summary) as f:
                    summary_data = json.load(f)

                console.print(f"   Total Datasets: {summary_data.get('total_datasets', 0)}")
                console.print(f"   Usable Datasets: {summary_data.get('usable_datasets', 0)}")
                console.print(
                    f"   Usable Percentage: {summary_data.get('usable_percentage', 0):.1f}%"
                )
                console.print(
                    f"   Average Quality Score: {summary_data.get('average_quality_score', 0):.1f}"
                )

            except Exception as e:
                console.print(f"   [red]Error reading summary: {str(e)}[/red]")


if __name__ == "__main__":
    prepare()
