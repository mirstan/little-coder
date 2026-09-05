"""Ingest a gaia.py benchmark run directory into NormalizedTrajectory objects.

Layout (confirmed against benchmarks/gaia.py, TDD_SPEC.md §0):
  <log_root>/manifest.json, results.json, submission.jsonl   (run-level, ignored here)
  <log_root>/<task_id>/{prompt.txt, transcript.txt, tool_calls.jsonl,
                        notifications.txt, stderr.log, result.json}

Confirmed gap: gaia's per-task result.json does not carry stop_reason (only
the run-level manifest.json's aggregate does) -- default to "agent_end" for a
correct/complete run, "unknown" otherwise, per TDD_SPEC.md §3.2.
"""
import json
import logging
from pathlib import Path

from benchmarks.self_improve.ingest.common import merge_component_usage, summarize_for_reflection
from benchmarks.self_improve.schema import NormalizedTrajectory

logger = logging.getLogger(__name__)


def load(log_root: Path) -> list[NormalizedTrajectory]:
    log_root = Path(log_root)
    if not log_root.is_dir():
        logger.warning("gaia_ingest: log_root does not exist or is not a directory: %s", log_root)
        return []

    trajectories: list[NormalizedTrajectory] = []
    for entry in sorted(log_root.iterdir()):
        # manifest.json/results.json/submission.jsonl are run-level FILES,
        # not directories, so this already excludes them -- no separate
        # name-based check needed (a prior one here was dead code: it ran
        # after this same is_dir() filter, so it could never match).
        if not entry.is_dir():
            continue
        traj = _load_task(entry)
        if traj is not None:
            trajectories.append(traj)
    return trajectories


def _load_task(task_dir: Path) -> NormalizedTrajectory | None:
    result_path = task_dir / "result.json"
    if not result_path.exists():
        logger.warning("gaia_ingest: skipping %s, no result.json", task_dir)
        return None

    try:
        result = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("gaia_ingest: malformed result.json at %s: %s", result_path, e)
        return None
    success = bool(result.get("correct", False))

    stderr_path = task_dir / "stderr.log"
    failure_signals = []
    if stderr_path.exists() and stderr_path.read_text().strip():
        failure_signals.append("harness_error")

    notif_path = task_dir / "notifications.txt"
    notif_lines = notif_path.read_text().splitlines() if notif_path.exists() else []
    components_used = merge_component_usage(notif_lines)

    tool_calls_path = task_dir / "tool_calls.jsonl"
    tool_calls = []
    if tool_calls_path.exists():
        for line in tool_calls_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                tool_calls.append(json.loads(line))
            except json.JSONDecodeError as e:
                # A partially-flushed final line is the most likely
                # corruption for an append-only file written during a
                # crashed run -- skip it rather than failing the whole task.
                logger.warning("gaia_ingest: malformed line in %s: %s", tool_calls_path, e)

    transcript_path = task_dir / "transcript.txt"
    assistant_text = transcript_path.read_text() if transcript_path.exists() else ""

    stop_reason = "agent_end" if success else ("harness_error" if failure_signals else "unknown")

    raw_paths = {
        name: str(task_dir / fname)
        for name, fname in {
            "transcript": "transcript.txt",
            "prompt": "prompt.txt",
            "tool_calls": "tool_calls.jsonl",
            "notifications": "notifications.txt",
            "result": "result.json",
        }.items()
        if (task_dir / fname).exists()
    }

    return NormalizedTrajectory(
        benchmark="gaia",
        task_id=task_dir.name,
        success=success,
        stop_reason=stop_reason,
        turn_count=len(tool_calls),
        partial_score=1.0 if success else 0.0,
        components_used=components_used,
        failure_signals=failure_signals,
        summarized_transcript=summarize_for_reflection(assistant_text, tool_calls),
        raw_paths=raw_paths,
    )
