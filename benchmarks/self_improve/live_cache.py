"""On-disk memo: (candidate, run_config, exercise_id) -> live run result.

Investigated whether GEPA's own EvaluationCache (gepa/core/state.py, enabled
via gepa.optimize(cache_evaluation=True)) makes this unnecessary. Verified
directly against the installed source that it does not: reflective_mutation.py's
parent (:306) and child (:550) minibatch evaluations call _batch_evaluate
UNCONDITIONALLY -- the cache is only written afterward, never read first.
GEPA's cache is only read-consulted on the valset path
(engine.py:193-210's _evaluate_programs_on_valset). So GEPA's own cache only
ever saves a valset re-evaluation (paid on every accepted proposal); it does
nothing for the parent/child minibatch re-evaluation that happens on EVERY
iteration -- the dominant cost term. This memo, consulted inside
PolyglotGEPAAdapter.evaluate() regardless of which GEPA-internal path called
it, transparently covers whatever GEPA's cache didn't already filter out.

Never caches an environmental failure (timeout/error/empty-response/
harness_error) -- those are properties of the environment, not the
candidate; caching one would permanently pin a candidate at a false low
score for the rest of a run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

#: Result statuses that describe the ENVIRONMENT failing, not the candidate
#: -- never cached. Mirrors aider_polyglot.py's own status vocabulary
#: (_classify_status) plus "harness_error" for a live_eval-level failure
#: (missing results file, unparseable output, etc).
ENVIRONMENTAL_STATUSES = frozenset({"error", "fail_timeout", "empty_response", "harness_error"})


def candidate_hash(candidate: Mapping[str, str]) -> str:
    """Stable regardless of dict insertion order."""
    canonical = json.dumps(dict(sorted(candidate.items())), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_config_hash(config: Mapping[str, Any]) -> str:
    """config should cover everything besides the candidate text that
    changes what a score means: model, max_attempts, thinking, the scratch
    base commit sha, and the sha256 of the harness files themselves (so a
    harness bugfix mid-project can't let a stale cache entry poison a later
    comparison)."""
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LiveResultCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path_for(self, cand_hash: str, cfg_hash: str, exercise_id: str) -> Path:
        safe_exercise = exercise_id.replace("/", "__")
        return self.root / cfg_hash[:12] / cand_hash[:16] / f"{safe_exercise}.json"

    def get(
        self, candidate: Mapping[str, str], run_config: Mapping[str, Any], exercise_id: str,
    ) -> dict | None:
        path = self._path_for(candidate_hash(candidate), run_config_hash(run_config), exercise_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, UnicodeError):
            # A corrupt or unreadable entry is a MISS, never an exception --
            # a cache read must not be able to kill an expensive in-flight run.
            return None
        # Valid JSON that isn't an object (e.g. a bare list or number) would
        # otherwise be returned as-is and later blow up far from here, deep
        # inside LiveRunResult.from_dict()'s **dict(d) -- treat it as a miss
        # at the point where it's actually detected instead.
        return payload if isinstance(payload, dict) else None

    def put(
        self, candidate: Mapping[str, str], run_config: Mapping[str, Any],
        exercise_id: str, result: Mapping[str, Any],
    ) -> None:
        if result.get("status") in ENVIRONMENTAL_STATUSES:
            return
        path = self._path_for(candidate_hash(candidate), run_config_hash(run_config), exercise_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(dict(result), indent=2))
        tmp.replace(path)
