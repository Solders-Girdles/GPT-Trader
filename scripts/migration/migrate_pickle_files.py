#!/usr/bin/env python3
"""
Migration script for converting existing pickle files to secure formats.

This script scans for pickle files and converts them to appropriate secure formats:
- ML models -> joblib
- DataFrames -> parquet
- Arrays -> numpy format
- Other data -> JSON

SECURITY NOTE: This script temporarily uses pickle ONLY for reading existing
files during migration. After migration, all pickle usage should be eliminated.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.utils.serialization import migrate_pickle_file

logger = logging.getLogger(__name__)


def find_pickle_files(search_paths: list[str]) -> list[Path]:
    """Find all pickle files in the given paths."""
    pickle_files = []

    for search_path in search_paths:
        path = Path(search_path)
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            continue

        # Find all pickle files
        patterns = ["*.pkl", "*.pickle", "*.p"]
        for pattern in patterns:
            pickle_files.extend(path.rglob(pattern))

    return list(set(pickle_files))  # Remove duplicates


def determine_target_format(filepath: Path) -> str:
    """Determine the target format based on file location and name."""
    path_str = str(filepath).lower()

    if any(
        keyword in path_str for keyword in ["model", "ml", "sklearn", "classifier", "regressor"]
    ):
        return "model"
    elif any(keyword in path_str for keyword in ["dataframe", "data", "df"]):
        return "dataframe"
    elif any(keyword in path_str for keyword in ["array", "numpy", "matrix"]):
        return "array"
    else:
        return "json"  # Default fallback


def get_target_path(old_path: Path, data_type: str) -> Path:
    """Generate target path for the migrated file."""
    parent = old_path.parent
    stem = old_path.stem

    if data_type == "model":
        return parent / f"{stem}.joblib"
    elif data_type == "dataframe":
        return parent / f"{stem}.parquet"
    elif data_type == "array":
        return parent / f"{stem}.npy"
    else:  # json
        return parent / f"{stem}.json"


def migrate_files(pickle_files: list[Path], dry_run: bool = False) -> dict[str, Any]:
    """Migrate pickle files to secure formats."""
    results = {
        "total_files": len(pickle_files),
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
    }

    for pickle_file in pickle_files:
        try:
            logger.info(f"Processing {pickle_file}")

            # Determine data type and target path
            data_type = determine_target_format(pickle_file)
            target_path = get_target_path(pickle_file, data_type)

            if target_path.exists():
                logger.info(f"Target file already exists, skipping: {target_path}")
                results["skipped"] += 1
                continue

            if dry_run:
                logger.info(
                    f"DRY RUN: Would migrate {pickle_file} -> {target_path} (type: {data_type})"
                )
                results["successful"] += 1
                continue

            # Perform migration
            success = migrate_pickle_file(pickle_file, target_path, data_type)

            if success:
                logger.info(f"Successfully migrated {pickle_file} -> {target_path}")
                results["successful"] += 1

                # Create backup of original file
                backup_path = pickle_file.with_suffix(pickle_file.suffix + ".bak")
                pickle_file.rename(backup_path)
                logger.info(f"Original file backed up to {backup_path}")
            else:
                logger.error(f"Failed to migrate {pickle_file}")
                results["failed"] += 1
                results["errors"].append(str(pickle_file))

        except Exception as e:
            logger.error(f"Error processing {pickle_file}: {e}")
            results["failed"] += 1
            results["errors"].append(f"{pickle_file}: {e}")

    return results


def generate_report(results: dict[str, Any], output_file: Path):
    """Generate migration report."""
    report = {
        "migration_date": datetime.now().isoformat(),
        "summary": {
            "total_files": results["total_files"],
            "successful_migrations": results["successful"],
            "failed_migrations": results["failed"],
            "skipped_files": results["skipped"],
        },
        "errors": results["errors"] if results["errors"] else None,
    }

    with open(output_file, "w") as f:
        import json

        json.dump(report, f, indent=2)

    logger.info(f"Migration report saved to {output_file}")


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate pickle files to secure formats")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["models", "data", "cache"],
        help="Paths to search for pickle files",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be migrated without doing it"
    )
    parser.add_argument(
        "--report", default="pickle_migration_report.json", help="Path for migration report"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"pickle_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ],
    )

    logger.info("Starting pickle file migration")
    logger.info(f"Search paths: {args.paths}")
    logger.info(f"Dry run: {args.dry_run}")

    # Find pickle files
    pickle_files = find_pickle_files(args.paths)
    logger.info(f"Found {len(pickle_files)} pickle files")

    if not pickle_files:
        logger.info("No pickle files found, migration complete")
        return

    # Display files to be migrated
    logger.info("Files to migrate:")
    for pf in pickle_files:
        data_type = determine_target_format(pf)
        target_path = get_target_path(pf, data_type)
        logger.info(f"  {pf} -> {target_path} (type: {data_type})")

    if not args.dry_run:
        response = input("Proceed with migration? (y/N): ")
        if response.lower() != "y":
            logger.info("Migration cancelled by user")
            return

    # Perform migration
    results = migrate_files(pickle_files, dry_run=args.dry_run)

    # Generate report
    generate_report(results, Path(args.report))

    # Summary
    logger.info("Migration completed:")
    logger.info(f"  Total files: {results['total_files']}")
    logger.info(f"  Successful: {results['successful']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Skipped: {results['skipped']}")

    if results["errors"]:
        logger.error("Errors encountered:")
        for error in results["errors"]:
            logger.error(f"  {error}")

    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
