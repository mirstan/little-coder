"""On-disk memo: (candidate, run_config, exercise_id, sample_index) -> live
run result.

Callers must pass the SANITIZED candidate (live_eval._sanitize_candidate) --
the text that actually gets materialized and run -- not GEPA's raw proposal:
two proposals differing only by a re-emitted frontmatter block execute
byte-identical files and must share one entry. (Sanitizing can't happen in
here: live_eval imports this module.)

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

Never caches a possibly-environmental failure (timeout/error/empty-response/
harness_error): the environment can cause any of them, and caching one would
pin a candidate at a false low score across runs. This keeps them out of
this memo only -- GEPA's own EvaluationCache still records whatever score
the adapter returned. Two tiers: UNSCOREABLE_STATUSES are also
retried in place by live_eval's run_batch() (see there for what happens
when they persist); fail_timeout is scored as-is but still never cached.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

#: Result statuses with no scoreable agent attempt behind them: live_eval's
#: run_batch() retries these in place rather than scoring the first one.
#: "error"/"empty_response" come from aider_polyglot.py's own status
#: vocabulary (_classify_status); "harness_error" is a live_eval-level
#: failure (missing results file, unparseable output, etc).
UNSCOREABLE_STATUSES = frozenset({"harness_error", "error", "empty_response"})
#: Result statuses the ENVIRONMENT can cause -- never written to this memo.
#: Kept out of live_cache only: GEPA's own EvaluationCache (valset and
#: minibatch put_batch) still records the 0.0 the adapter returns. "error"/
#: "empty_response" may also be candidate-caused (context overflow, a
#: crashing component). A superset of UNSCOREABLE_STATUSES by construction:
#: anything too unreliable to score first time is too unreliable to cache.
#: fail_timeout is scored but not cached, because the per-attempt deadline is
#: wall-clock, so a slow model server can cause it as easily as the candidate.
ENVIRONMENTAL_STATUSES = UNSCOREABLE_STATUSES | {"fail_timeout"}


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

    def _path_for(self, cand_hash: str, cfg_hash: str, exercise_id: str, sample_index: int = 0) -> Path:
        safe_exercise = exercise_id.replace("/", "__")
        # Sample 0 keeps the pre-sample_index filename so an existing memo
        # (for any candidate with no frontmatter to sanitize away) stays warm.
        suffix = "" if sample_index == 0 else f".sample{int(sample_index)}"
        return self.root / cfg_hash[:12] / cand_hash[:16] / f"{safe_exercise}{suffix}.json"

    def get(
        self, candidate: Mapping[str, str], run_config: Mapping[str, Any], exercise_id: str,
        sample_index: int = 0,
    ) -> dict | None:
        path = self._path_for(candidate_hash(candidate), run_config_hash(run_config), exercise_id, sample_index)
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
        exercise_id: str, result: Mapping[str, Any], sample_index: int = 0,
    ) -> None:
        if result.get("status") in ENVIRONMENTAL_STATUSES:
            return
        path = self._path_for(candidate_hash(candidate), run_config_hash(run_config), exercise_id, sample_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(dict(result), indent=2))
        tmp.replace(path)
