"""Activate command for optimization CLI strategy artifacts."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.research.artifacts import StrategyArtifactError, StrategyArtifactStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")

COMMAND_NAME = "optimize artifact-activate"


def register(subparsers: Any) -> None:
    """Register the artifact-activate subcommand."""
    parser = subparsers.add_parser(
        "artifact-activate",
        help="Set an artifact as active for a profile",
        description="Activate a strategy artifact for a given profile.",
    )

    parser.add_argument(
        "artifact",
        type=str,
        help="Artifact id or path",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="optimized",
        help="Profile name for activation (default: optimized)",
    )

    parser.add_argument(
        "--allow-unapproved",
        action="store_true",
        help="Allow activating an unapproved artifact",
    )

    parser.add_argument(
        "--format",
        dest="output_format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    parser.set_defaults(handler=execute, subcommand="artifact-activate")


def execute(args: Namespace) -> CliResponse | int:
    """Execute the artifact-activate command."""
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

    warnings: list[str] = []
    if not artifact.approved and not args.allow_unapproved:
        message = (
            f"Artifact {artifact.artifact_id} is not approved. "
            "Publish it first or pass --allow-unapproved."
        )
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.VALIDATION_ERROR,
                message=message,
                details={"artifact_id": artifact.artifact_id},
            )
        logger.error(message)
        return 1
    if not artifact.approved and args.allow_unapproved:
        warnings.append("Activating unapproved artifact")

    registry = store.set_active(args.profile, artifact.artifact_id)

    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data={
                "artifact_id": artifact.artifact_id,
                "profile": args.profile,
                "registry": registry.to_dict(),
            },
            warnings=warnings,
        )

    print(f"Activated artifact {artifact.artifact_id} for profile '{args.profile}'")
    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}")
    return 0


__all__ = ["register", "execute"]
