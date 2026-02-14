"""Feature seed helpers for telemetry-derived backlog candidates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_DEFAULT_SUFFIX_LENGTH = 8


@dataclass(frozen=True)
class FeatureSeed:
    """Stable seed identifier and title for backlog candidates."""

    key: str
    title: str


def _slugify(value: str) -> str:
    normalized = _SLUG_PATTERN.sub("-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "seed"


def _hash_signature(signature: Mapping[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_feature_seed(
    title: str,
    *,
    signature: Mapping[str, Any] | None = None,
    suffix_length: int = _DEFAULT_SUFFIX_LENGTH,
) -> FeatureSeed:
    """Build a deterministic seed key and collision-resistant title.

    The suffix is derived from a hash of the title and signature, which keeps
    titles unique even when upstream systems merge on normalized titles.
    """

    base_title = title.strip() or "feature"
    normalized = _slugify(base_title)
    payload: dict[str, Any] = {"title": base_title}
    if signature:
        payload["signature"] = signature
    digest = _hash_signature(payload)
    suffix_length = max(4, min(12, int(suffix_length)))
    suffix = digest[:suffix_length]
    return FeatureSeed(
        key=f"{normalized}-{suffix}",
        title=f"{base_title} [{suffix}]",
    )
