"""Terminal-Bench 2.1 train/validate/test split.

Re-runnable and reproducible (fixed random seed), not a hand-typed list --
regenerate with `python3 benchmarks/tb21_split.py`, which overwrites
benchmarks/tb21_splits.json.

Rationale for a train/validate/test split at all: this session's harness
development (permission-gate whitelist behavior, deadline-derivation,
TB-aware finalize messaging, etc.) iterated directly against a hand-picked
subset of tasks. Any pass-rate computed on those same tasks going forward is
optimistic -- the harness has effectively been tuned against them. Train
keeps using that already-contaminated set (no loss, since it was never a
clean measurement anyway); validate is a small held-out set for periodic
regression checks during ongoing development; test is the true held-out
remainder, meant to be touched rarely, for an honest final read.
"""
from __future__ import annotations

import glob
import json
import random
import tomllib
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARBOR_TASK_CACHE = Path.home() / ".cache" / "harbor" / "tasks"
OUTPUT_PATH = Path(__file__).resolve().parent / "tb21_splits.json"

EXPECTED_TASK_COUNT = 89

# Every distinct task name that appeared under benchmarks/harbor_runs/ this
# session (not just the final curated 9 -- includes earlier subset runs from
# before the harness settled), i.e. every task this session's harness fixes
# were actually diagnosed and iterated against:
#   find benchmarks/harbor_runs -mindepth 2 -maxdepth 2 -type d \
#     | sed 's|.*/||; s|__.*||' | sort -u
TRAIN_TASKS = sorted([
    "break-filter-js-from-html",
    "cobol-modernization",
    "configure-git-webserver",
    "constraints-scheduling",
    "fix-git",
    "gpt2-codegolf",
    "llm-inference-batching-scheduler",
    "mteb-leaderboard",
    "nginx-request-logging",
    "overfull-hbox",
    "prove-plus-comm",
    "raman-fitting",
    "reshard-c4-data",
    "train-fasttext",
    "vulnerable-secret",
    "write-compressor",
])

VALIDATE_SEED = 42
VALIDATE_SIZE = 12


def load_all_tasks() -> list[dict]:
    """Parse every cached TB2.1 task.toml. Raises if the cache is missing or
    doesn't have exactly EXPECTED_TASK_COUNT tasks, rather than silently
    producing a split derived from a stale/partial cache."""
    pattern = str(HARBOR_TASK_CACHE / "packages" / "terminal-bench" / "*" / "*" / "task.toml")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(
            f"No task.toml files found under {pattern}. "
            'Run `harbor download "terminal-bench/terminal-bench-2-1" --cache` first.'
        )
    tasks = []
    for p in paths:
        name = Path(p).parent.parent.name
        with open(p, "rb") as f:
            data = tomllib.load(f)
        meta = data.get("metadata", {})
        agent = data.get("agent", {})
        tasks.append({
            "name": name,
            "difficulty": meta.get("difficulty"),
            "category": meta.get("category"),
            "tags": meta.get("tags", []),
            "timeout_sec": agent.get("timeout_sec"),
        })
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} tasks, found {len(tasks)}. "
            "The dataset may have changed -- re-review the split before trusting it."
        )
    return tasks


def compute_validate_set(
    all_tasks: list[dict], train: set[str], seed: int = VALIDATE_SEED, n: int = VALIDATE_SIZE,
) -> list[dict]:
    """Stratified selection: guarantee one task from every category with zero
    train coverage first, then allocate remaining slots proportional to
    remaining-pool size among categories train already touches (largest
    category gets 2, next few get 1 each)."""
    rng = random.Random(seed)
    remaining = [t for t in all_tasks if t["name"] not in train]
    by_cat: dict[str, list[dict]] = {}
    for t in remaining:
        by_cat.setdefault(t["category"], []).append(t)

    train_cats = {t["category"] for t in all_tasks if t["name"] in train}
    uncovered_cats = sorted(c for c in by_cat if c not in train_cats)
    covered_cats = [c for c in by_cat if c in train_cats]

    validate: list[dict] = []
    for cat in uncovered_cats:
        pool = sorted(by_cat[cat], key=lambda t: t["name"])
        pick = rng.choice(pool)
        validate.append(pick)
        by_cat[cat].remove(pick)

    remaining_slots = n - len(validate)
    covered_sorted = sorted(covered_cats, key=lambda c: -len(by_cat[c]))
    allocation = [("software-engineering", 2)]
    others = [c for c in covered_sorted if c != "software-engineering" and by_cat.get(c)]
    allocation += [(c, 1) for c in others[: max(remaining_slots - 2, 0)]]

    for cat, count in allocation:
        pool = sorted(by_cat[cat], key=lambda t: t["name"])
        picks = rng.sample(pool, min(count, len(pool)))
        validate.extend(picks)

    return validate


def main() -> None:
    all_tasks = load_all_tasks()
    train = set(TRAIN_TASKS)
    missing_from_cache = train - {t["name"] for t in all_tasks}
    if missing_from_cache:
        raise RuntimeError(f"TRAIN_TASKS not found in the dataset: {sorted(missing_from_cache)}")

    validate_tasks = compute_validate_set(all_tasks, train)
    validate = {t["name"] for t in validate_tasks}
    test = {t["name"] for t in all_tasks} - train - validate

    # The one correctness property that must never silently break.
    assert train & validate == set(), "train/validate overlap"
    assert train & test == set(), "train/test overlap"
    assert validate & test == set(), "validate/test overlap"
    assert train | validate | test == {t["name"] for t in all_tasks}, \
        "split does not cover every task exactly once"

    by_name = {t["name"]: t for t in all_tasks}

    def summarize(label: str, names: set[str]) -> None:
        tasks = [by_name[n] for n in names]
        print(f"{label}: {len(tasks)} tasks")
        print(f"  difficulty: {dict(Counter(t['difficulty'] for t in tasks))}")
        print(f"  category:   {dict(Counter(t['category'] for t in tasks))}")

    summarize("TRAIN", train)
    summarize("VALIDATE", validate)
    summarize("TEST", test)

    result = {
        "generated_by": "benchmarks/tb21_split.py",
        "dataset": "terminal-bench/terminal-bench-2-1",
        "seed": VALIDATE_SEED,
        "train": sorted(train),
        "validate": sorted(validate),
        "test": sorted(test),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
