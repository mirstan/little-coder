"""ExerciseSpec, deterministic seeded exercise selection, and exercise
description text for reflection prompts."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class ExerciseSpec:
    exercise: str
    language: str = "python"

    @property
    def task_id(self) -> str:
        return f"{self.language}/{self.exercise}"

    @property
    def results_key(self) -> str:
        """Matches aider_polyglot.py:861's `f"{agent}/{language}/{name}"` key
        format for the pi agent (the only agent live-eval drives)."""
        return f"pi/{self.language}/{self.exercise}"


def practice_dir(benchmark_root: Path, language: str) -> Path:
    return Path(benchmark_root) / language / "exercises" / "practice"


def discover_exercises(practice_dir_path: Path) -> list[str]:
    practice_dir_path = Path(practice_dir_path)
    if not practice_dir_path.is_dir():
        return []
    return sorted(p.name for p in practice_dir_path.iterdir() if p.is_dir())


def select_exercises(
    available: list[str], *, count: int, seed: int, language: str = "python",
    explicit: list[str] | None = None, exclude: tuple[str, ...] = (),
) -> list[ExerciseSpec]:
    """Deterministic given seed. `explicit` bypasses sampling entirely and is
    validated against `available`, raising with the full available list on a
    typo -- surfacing a misconfiguration before a scratch worktree is even
    created, rather than as a "status": "error" deep inside a live run."""
    if explicit:
        unknown = sorted(set(explicit) - set(available))
        if unknown:
            raise ValueError(f"unknown exercise(s) {unknown} -- available: {sorted(available)}")
        chosen = list(explicit)
    else:
        pool = sorted(set(available) - set(exclude))
        if count > len(pool):
            raise ValueError(f"requested {count} exercises but only {len(pool)} available")
        rng = random.Random(seed)
        rng.shuffle(pool)
        chosen = pool[:count]
    return [ExerciseSpec(exercise=name, language=language) for name in chosen]


def split_train_val(
    specs: list[ExerciseSpec], *, val_count: int, seed: int,
) -> tuple[list[ExerciseSpec], list[ExerciseSpec]]:
    """A genuine held-out split -- there is no success label to stratify on
    before any exercise has actually run, unlike the old design's
    stratify-by-(benchmark,success) over historical trajectories."""
    if val_count <= 0 or val_count >= len(specs):
        raise ValueError(f"val_count must be between 1 and {len(specs) - 1}, got {val_count}")
    shuffled = sorted(specs)
    random.Random(seed).shuffle(shuffled)
    return shuffled[val_count:], shuffled[:val_count]


def describe_exercise(spec: ExerciseSpec, practice_dir_path: Path, *, max_chars: int = 2000) -> str:
    """Real exercise description for a reflective feedback record -- prefers
    the exercise's own instructions/README, falling back to listing stub and
    test filenames plus the test file's own first lines when neither exists
    (matches the local fixture layout, which has no .docs/instructions.md)."""
    ex_dir = Path(practice_dir_path) / spec.exercise
    docs = ex_dir / ".docs" / "instructions.md"
    if docs.exists():
        return docs.read_text()[:max_chars]
    readme = ex_dir / "README.md"
    if readme.exists():
        return readme.read_text()[:max_chars]
    if not ex_dir.is_dir():
        return f"(exercise directory not found: {ex_dir})"
    stub_files = sorted(p.name for p in ex_dir.glob("*.py") if not p.name.endswith("_test.py"))
    test_files = sorted(p.name for p in ex_dir.glob("*_test.py"))
    parts = [f"Exercise: {spec.exercise}", f"Stub files: {stub_files}", f"Test files: {test_files}"]
    for tf in test_files:
        content = (ex_dir / tf).read_text()
        parts.append(f"\n--- {tf} (first 40 lines) ---\n" + "\n".join(content.splitlines()[:40]))
    return "\n".join(parts)[:max_chars]
