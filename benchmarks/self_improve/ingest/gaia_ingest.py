"""Ingest a gaia.py benchmark run directory into NormalizedTrajectory objects.

Layout (confirmed against benchmarks/gaia.py, TDD_SPEC.md §0):
  <log_root>/manifest.json, results.json, submission.jsonl   (run-level, ignored here)
  <log_root>/<task_id>/{prompt.txt, transcript.txt, tool_calls.jsonl,
                        notifications.txt, stderr.log, result.json}

result.json DOES carry a real per-task stop_reason and turn_count (both
straight from PiRpc's own result, see benchmarks/gaia.py:199,206,235,240) --
an earlier version of this comment claimed otherwise (stale/incomplete
research); a computed fallback is used only for older data written before
gaia.py recorded these fields. "correct" (and therefore success/gold) is
only present when gaia.py was run with a gold answer to score against
(score_against_gold=True) -- an unlabeled task has no "correct" key at all
and is skipped entirely, not treated as a failure (real bug, confirmed by
review: `result.get("correct", False)` silently turned every unlabeled task
into a false negative).
"""
import json
import logging
from pathlib import Path

from benchmarks.self_improve.ingest.common import (
    build_knowledge_topic_index,
    merge_component_usage,
    summarize_for_reflection,
)
from benchmarks.self_improve.schema import NormalizedTrajectory

logger = logging.getLogger(__name__)


def load(log_root: Path, repo_root: Path | None = None) -> list[NormalizedTrajectory]:
    """repo_root, if given, resolves knowledge-inject component usage against
    the real skills/knowledge and skills/protocols files (see
    build_knowledge_topic_index()). Without it, knowledge-inject usage is
    dropped (skill-inject usage is unaffected either way)."""
    log_root = Path(log_root)
    if not log_root.is_dir():
        logger.warning("gaia_ingest: log_root does not exist or is not a directory: %s", log_root)
        return []

    knowledge_topic_index = build_knowledge_topic_index(repo_root) if repo_root else {}

    trajectories: list[NormalizedTrajectory] = []
    for entry in sorted(log_root.iterdir()):
        # manifest.json/results.json/submission.jsonl are run-level FILES,
        # not directories, so this already excludes them -- no separate
        # name-based check needed (a prior one here was dead code: it ran
        # after this same is_dir() filter, so it could never match).
        if not entry.is_dir():
            continue
        try:
            traj = _load_task(entry, knowledge_topic_index)
        except (OSError, UnicodeDecodeError) as e:
            # Real gap, confirmed by review: an unguarded read_text() on
            # stderr.log/notifications.txt/transcript.txt (e.g. non-UTF-8
            # content from a crashed run) previously propagated all the way
            # out of load(), where run_gepa.py's _ingest_all's broad
            # except-Exception would discard EVERY task from this log_root,
            # not just the one bad one.
            logger.warning("gaia_ingest: skipping %s, failed to read task files: %s", entry, e)
            continue
        if traj is not None:
            trajectories.append(traj)
    return trajectories


def _load_task(task_dir: Path, knowledge_topic_index: dict[str, str] | None = None) -> NormalizedTrajectory | None:
    result_path = task_dir / "result.json"
    if not result_path.exists():
        logger.warning("gaia_ingest: skipping %s, no result.json", task_dir)
        return None

    try:
        result = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("gaia_ingest: malformed result.json at %s: %s", result_path, e)
        return None

    # Real bug, confirmed by review against benchmarks/gaia.py:244-249:
    # "correct" is only added to result.json `if score_against_gold:` -- a
    # gaia run against tasks with no gold answer (the unlabeled/test split)
    # produces a result.json with NO "correct" key at all. The previous
    # `result.get("correct", False)` treated every such unlabeled task as a
    # hard FAILURE, feeding false negatives into GEPA's scoring. Skip
    # unscoreable tasks entirely instead -- NormalizedTrajectory.success is
    # a required bool, there's no honest value to put there.
    if "correct" not in result:
        logger.warning("gaia_ingest: skipping %s, no gold answer to score against (no 'correct' key)", task_dir)
        return None
    success = bool(result["correct"])

    stderr_path = task_dir / "stderr.log"
    failure_signals = []
    if stderr_path.exists() and stderr_path.read_text().strip():
        failure_signals.append("harness_error")

    notif_path = task_dir / "notifications.txt"
    notif_lines = notif_path.read_text().splitlines() if notif_path.exists() else []
    components_used = merge_component_usage(notif_lines, knowledge_topic_index=knowledge_topic_index)

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

    # Real bug, confirmed by review against benchmarks/gaia.py:206,240: gaia.py
    # DOES persist a real per-task stop_reason in result.json (from
    # PiRpc's actual result, e.g. "agent_end"/"deadline"/"process_exit") --
    # this module's own prior "gaia doesn't carry stop_reason" comment was
    # wrong (based on stale/incomplete research). Prefer the persisted value;
    # only fall back to the computed heuristic for older data written before
    # gaia.py recorded this field (empty string there too, per gaia.py:183).
    stop_reason = result.get("stop_reason") or (
        "agent_end" if success else ("harness_error" if failure_signals else "unknown")
    )

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

    # Real bug, confirmed by review against benchmarks/gaia.py:199,235: gaia.py
    # persists a real per-task turn_count (from PiRpc's own turn accounting) --
    # len(tool_calls) is a distinct count of individual tool invocations, not
    # turns (multiple tool calls can happen in one turn). Prefer the
    # persisted value; fall back for older data that predates this field.
    turn_count = result.get("turn_count")
    if not isinstance(turn_count, int):
        turn_count = len(tool_calls)

    return NormalizedTrajectory(
        benchmark="gaia",
        task_id=task_dir.name,
        success=success,
        stop_reason=stop_reason,
        turn_count=turn_count,
        partial_score=1.0 if success else 0.0,
        components_used=components_used,
        failure_signals=failure_signals,
        summarized_transcript=summarize_for_reflection(assistant_text, tool_calls),
        raw_paths=raw_paths,
    )
