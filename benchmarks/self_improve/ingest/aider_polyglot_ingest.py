"""Ingest an aider_polyglot.py benchmark run into NormalizedTrajectory objects.

One trajectory per (lang, exercise) key in results_full_polyglot.json -- NOT
per attempt file. The aggregate record's status already captures which
attempt (if any) passed; per-attempt trajectory_<n>.json files under
log_root/<lang>/<exercise>/ populate summarized_transcript and raw_paths from
the LATEST successfully-parsed attempt, but components_used is a UNION across
EVERY successfully-parsed attempt (a skill injected on an earlier, failed
attempt but not re-triggered on the one that ultimately passed still
genuinely influenced the outcome; docstring corrected per review -- it
previously said "latest attempt" here too, which stopped matching the
implementation once that was fixed).

Formerly a confirmed gap (TDD_SPEC.md §0/§4.2): aider_polyglot's
_dump_trajectory did not persist rpc.notifications(), so components_used was
always empty here. Closed upstream (benchmarks/aider_polyglot.py's
_dump_trajectory now takes a notifications= kwarg) -- components_used
populates from each attempt's "notifications" field when present, and
degrades to [] gracefully for older-format data that predates this field.

Result-record schema evolved twice, both handled here with fallback:
- stop_reason_1/stop_reason_2 (fixed two-attempt fields) -> stop_reasons
  (a list, one entry per attempt, since --max-attempts generalized beyond 2).
- status "pass_1"/"pass_2" only -> "pass_N" for any N (same reason). A
  hardcoded 2-entry score lookup would silently score any pass_3+ as a
  failure -- confirmed real bug, fixed via _pass_n_score() below.
"""
import json
import logging
import re
from pathlib import Path

from benchmarks.self_improve.ingest.common import (
    build_knowledge_topic_index,
    merge_component_usage,
    summarize_for_reflection,
)
from benchmarks.self_improve.path_safety import resolve_contained_path
from benchmarks.self_improve.schema import NormalizedTrajectory

logger = logging.getLogger(__name__)

_PASS_N_RE = re.compile(r"^pass_(\d+)$")
#: Score decays 0.3 per attempt past the first, floored so a genuine pass
#: (however many attempts it took) is never scored as low as an outright
#: failure. pass_1=1.0, pass_2=0.7 (exact values from the original design,
#: preserved for backward compat), pass_3+=0.4.
_PASS_N_FLOOR = 0.4
_PASS_N_DECAY = 0.3


def _pass_n_score(status: str) -> tuple[bool, float] | None:
    """Return (success, partial_score) for any "pass_N" status, or None if
    status doesn't match that pattern at all (a real failure/error status)."""
    m = _PASS_N_RE.match(status)
    if not m:
        return None
    n = int(m.group(1))
    score = max(_PASS_N_FLOOR, 1.0 - _PASS_N_DECAY * (n - 1))
    return True, score


#: Public alias for live_eval.py -- reused rather than reimplemented, since
#: it's the exact same graded scoring scheme (pass_1=1.0/pass_2=0.7/pass_3+=0.4)
#: applied to a real aider_polyglot.py "status" string either way.
pass_n_score = _pass_n_score


def load(
    log_root: Path, results_json_path: Path, repo_root: Path | None = None
) -> list[NormalizedTrajectory]:
    """repo_root, if given, resolves knowledge-inject component usage against
    the real skills/knowledge and skills/protocols files (see
    ingest.common.build_knowledge_topic_index()). Without it, knowledge-inject
    usage is dropped (skill-inject usage is unaffected either way)."""
    results_json_path = Path(results_json_path)
    if not results_json_path.exists():
        raise FileNotFoundError(f"results_full_polyglot.json not found: {results_json_path}")

    log_root = Path(log_root)
    results = json.loads(results_json_path.read_text())
    exercises = results.get("exercises", {})
    knowledge_topic_index = build_knowledge_topic_index(repo_root) if repo_root else {}

    trajectories = []
    for key, record in exercises.items():
        lang, _, exercise = key.partition("/")
        # results_full_polyglot.json is repo-controlled data today, but a
        # corrupted or malicious "lang/exercise" key (e.g. containing "..")
        # would otherwise let ex_dir escape log_root entirely -- cheap to
        # validate, so validate it (shared with components.py's identical
        # check -- see path_safety.py).
        try:
            resolve_contained_path(log_root, f"{lang}/{exercise}")
        except ValueError:
            logger.warning("aider_polyglot_ingest: skipping %r, escapes log_root", key)
            continue
        trajectories.append(_build_trajectory(log_root, lang, exercise, key, record, knowledge_topic_index))
    return trajectories


def _build_trajectory(
    log_root: Path, lang: str, exercise: str, key: str, record: dict,
    knowledge_topic_index: dict[str, str] | None = None,
) -> NormalizedTrajectory:
    status = record.get("status", "fail")
    success, partial_score = _pass_n_score(status) or (False, 0.0)

    stop_reasons = record.get("stop_reasons")
    if stop_reasons:
        stop_reason = stop_reasons[-1] or "unknown"
    else:
        # Backward compat: real, already-captured data predates the
        # stop_reasons list and only has these two fixed fields.
        stop_reason = record.get("stop_reason_2") or record.get("stop_reason_1") or "unknown"

    ex_dir = log_root / lang / exercise
    assistant_text = ""
    tool_calls: list[dict] = []
    raw_paths: dict[str, str] = {}
    all_notif_lines: list[str] = []

    if ex_dir.is_dir():
        # Sort by the parsed attempt NUMBER, not the filename string --
        # lexicographic order puts "trajectory_10" before "trajectory_2"
        # (real bug, confirmed by review: with >=10 attempts the wrong file
        # supplied summarized_transcript/components_used/raw_paths).
        def _attempt_num(p: Path) -> int:
            m = re.search(r"_(\d+)$", p.stem)
            return int(m.group(1)) if m else -1

        attempt_files = sorted(ex_dir.glob("trajectory_*.json"), key=_attempt_num)
        for attempt_file in attempt_files:
            try:
                data = json.loads(attempt_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("aider_polyglot_ingest: failed to read %s: %s", attempt_file, e)
                continue
            # components_used is a UNION across every successfully-parsed
            # attempt: a skill injected on an earlier, failed attempt but not
            # re-triggered on the attempt that ultimately passed still
            # genuinely influenced the outcome -- real gap, confirmed by
            # review, in previously reading only the latest attempt's
            # notifications. Present only for trajectories dumped after
            # aider_polyglot.py's notifications= addition to
            # _dump_trajectory -- absent for older data, degrades to [].
            all_notif_lines.extend(
                f"[{n.get('notifyType', 'info')}] {n.get('message', '')}"
                for n in data.get("notifications", [])
            )
            # summarized_transcript/raw_paths reflect the LATEST
            # successfully-parsed attempt only -- that conversation is the
            # one actually relevant to reflection_lm.
            assistant_text = data.get("assistant_text", "")
            tool_calls = data.get("tool_calls", [])
            raw_paths["trajectory"] = str(attempt_file)

    components_used = merge_component_usage(all_notif_lines, knowledge_topic_index=knowledge_topic_index)

    return NormalizedTrajectory(
        benchmark="aider_polyglot",
        task_id=key,
        success=success,
        stop_reason=stop_reason,
        turn_count=record.get("turn_count", 0) or 0,
        partial_score=partial_score,
        components_used=components_used,
        failure_signals=[] if success else [status],
        summarized_transcript=summarize_for_reflection(assistant_text, tool_calls),
        raw_paths=raw_paths,
    )
