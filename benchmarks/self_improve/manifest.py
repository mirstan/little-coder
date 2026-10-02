"""Pre-registered run manifest (DESIGN_FAST_ITERATION.md §3 P1, §7.5, §8).

Written into a run's out-dir before any spend, so the splits, metrics,
tolerances and budget a run is judged by are fixed before the optimizer
sees a single score. The optimizer never reads it back to edit it.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MANIFEST_SCHEMA_VERSION = 1
SPLIT_NAMES = ("search", "acceptance", "test")


def _default_metrics() -> dict[str, Any]:
    return {
        "primary": "pass",
        "secondary": ["attempt_weighted_score"],
        "efficiency": "decode+prompt tokens",
    }


@dataclass
class Manifest:
    created_at: str
    seed: int
    benchmark: str
    language: str
    splits: dict[str, list[str]]
    searchable_components: list[str]
    sampling: dict[str, Any]
    budget: dict[str, Any]
    env_fingerprint: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=_default_metrics)
    alpha_t4: float = 0.01
    root_tolerance_pp: float = 2.0
    mde: float = 0.2
    t5_max_candidates: int = 3
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} is not supported (expected {MANIFEST_SCHEMA_VERSION})"
            )
        if set(self.splits) != set(SPLIT_NAMES):
            raise ValueError(f"splits must have exactly the keys {', '.join(SPLIT_NAMES)}, got {sorted(self.splits)}")
        for name in SPLIT_NAMES:
            if not self.splits[name]:
                raise ValueError(f"split {name!r} is empty")
        for i, a in enumerate(SPLIT_NAMES):
            for b in SPLIT_NAMES[i + 1:]:
                shared = sorted(set(self.splits[a]) & set(self.splits[b]))
                if shared:
                    raise ValueError(f"splits {a!r} and {b!r} are not disjoint: {shared}")
        # Temperature 0 removes the very sampling variance the acceptance
        # protocol is estimating (§7.5), so it is required and must be > 0.
        temperature = self.sampling.get("temperature")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError(f"sampling.temperature must be given and > 0, got {temperature!r}")
        if self.t5_max_candidates < 1:
            raise ValueError(f"t5_max_candidates must be >= 1, got {self.t5_max_candidates}")
        for name in ("alpha_t4", "mde"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1), got {value}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown manifest field(s): {unknown}")
        return cls(**data)

    def save(self, path: Path) -> None:
        """Write-once: exclusive create, so saving over an existing manifest
        raises FileExistsError instead of re-registering a run in place."""
        with Path(path).open("x") as f:
            f.write(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})
