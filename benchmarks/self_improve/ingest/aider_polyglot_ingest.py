"""Ingest an aider_polyglot.py benchmark run into NormalizedTrajectory objects.

One trajectory per (lang, exercise) key in results_full_polyglot.json -- NOT
per attempt file. The aggregate record's status already captures which
attempt (if any) passed; per-attempt trajectory_<n>.json files under
log_root/<lang>/<exercise>/ are used only to populate summarized_transcript
and raw_paths from the latest attempt.

Confirmed gap (TDD_SPEC.md §0/§4.2): aider_polyglot's _dump_trajectory does
not persist rpc.notifications(), so components_used is always empty here
until that upstream capture is added.
"""
import json
import logging
from pathlib import Path

from benchmarks.self_improve.ingest.common import summarize_for_reflection
from benchmarks.self_improve.schema import NormalizedTrajectory

logger = logging.getLogger(__name__)

_STATUS_TO_SCORE = {
    "pass_1": (True, 1.0),
    "pass_2": (True, 0.7),
}


def load(log_root: Path, results_json_path: Path) -> list[NormalizedTrajectory]:
    results_json_path = Path(results_json_path)
    if not results_json_path.exists():
        raise FileNotFoundError(f"results_full_polyglot.json not found: {results_json_path}")

    results = json.loads(results_json_path.read_text())
    exercises = results.get("exercises", {})

    trajectories = []
    for key, record in exercises.items():
        lang, _, exercise = key.partition("/")
        trajectories.append(_build_trajectory(Path(log_root), lang, exercise, key, record))
    return trajectories


def _build_trajectory(log_root: Path, lang: str, exercise: str, key: str, record: dict) -> NormalizedTrajectory:
    status = record.get("status", "fail")
    success, partial_score = _STATUS_TO_SCORE.get(status, (False, 0.0))

    stop_reason = record.get("stop_reason_2") or record.get("stop_reason_1") or "unknown"

    ex_dir = log_root / lang / exercise
    assistant_text = ""
    tool_calls: list[dict] = []
    raw_paths: dict[str, str] = {}

    if ex_dir.is_dir():
        attempt_files = sorted(
            ex_dir.glob("trajectory_*.json"),
            key=lambda p: p.stem,
        )
        if attempt_files:
            latest = attempt_files[-1]
            try:
                data = json.loads(latest.read_text())
                assistant_text = data.get("assistant_text", "")
                tool_calls = data.get("tool_calls", [])
                raw_paths["trajectory"] = str(latest)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("aider_polyglot_ingest: failed to read %s: %s", latest, e)

    return NormalizedTrajectory(
        benchmark="aider_polyglot",
        task_id=key,
        success=success,
        stop_reason=stop_reason,
        turn_count=record.get("turn_count", 0) or 0,
        partial_score=partial_score,
        components_used=[],
        failure_signals=[] if success else [status],
        summarized_transcript=summarize_for_reflection(assistant_text, tool_calls),
        raw_paths=raw_paths,
    )
