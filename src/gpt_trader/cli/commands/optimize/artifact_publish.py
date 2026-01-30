"""Publish command for optimization CLI strategy artifacts."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.research.artifacts import StrategyArtifactError, StrategyArtifactStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")

COMMAND_NAME = "optimize artifact-publish"


def register(subparsers: Any) -> None:
    """Register the artifact-publish subcommand."""
    parser = subparsers.add_parser(
        "artifact-publish",
        help="Approve a strategy artifact for live use",
        description="Mark a strategy artifact as approved.",
    )

    parser.add_argument(
        "artifact",
        type=str,
        help="Artifact id or path",
    )

    parser.add_argument(
        "--approved-by",
        type=str,
        help="Approver identifier (optional)",
    )

    parser.add_argument(
        "--notes",
        type=str,
        help="Approval notes (optional)",
    )

    parser.add_argument(
        "--format",
        dest="output_format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    parser.set_defaults(handler=execute, subcommand="artifact-publish")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the artifact-publish command."""
    output_format = getattr(args, "output_format", "text")
    store = StrategyArtifactStore()

    try:
        artifact = store.load(args.artifact)
    except StrategyArtifactError as exc:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.FILE_NOT_FOUND,
                message=str(exc),
                details={"artifact": args.artifact},
            )
        logger.error(str(exc))
        return 1

    try:
        published = store.publish(
            artifact.artifact_id,
            approved_by=args.approved_by,
            notes=args.notes,
        )
    except StrategyArtifactError as exc:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.OPERATION_FAILED,
                message=str(exc),
                details={"artifact_id": artifact.artifact_id},
            )
        logger.error(str(exc))
        return 1

    artifact_path = store.artifact_path(published.artifact_id)

    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data={
                "artifact_id": published.artifact_id,
                "approved": published.approved,
                "approved_at": published.approved_at,
                "approved_by": published.approved_by,
                "notes": published.notes,
                "path": str(artifact_path),
            },
        )

    print(f"Published artifact {published.artifact_id}")
    print(f"Path: {artifact_path}")
    if published.approved_by:
        print(f"Approved by: {published.approved_by}")
    if published.notes:
        print(f"Notes: {published.notes}")
    return 0


__all__ = ["register", "execute"]
