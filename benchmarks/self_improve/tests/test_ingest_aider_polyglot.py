import json
from pathlib import Path

import pytest

from benchmarks.self_improve.ingest import aider_polyglot_ingest


def _write_trajectory(path, **fields):
    base = {
        "attempt": "1", "agent_ended": True, "turn_count": 1,
        "compaction_events": 0, "assistant_text": "", "tool_calls": [],
    }
    base.update(fields)
    path.write_text(json.dumps(base))


@pytest.fixture
def aider_run(tmp_path):
    log_root = tmp_path / "logs"
    two_fer = log_root / "python" / "two-fer"
    two_fer.mkdir(parents=True)
    _write_trajectory(
        two_fer / "trajectory_1.json",
        turn_count=4,
        tool_calls=[{"name": "write", "args": {}, "result_text": "ok", "is_error": False}],
    )
    (two_fer / "final_output_1.txt").write_text("1 passed")

    leap = log_root / "python" / "leap"
    leap.mkdir(parents=True)
    _write_trajectory(leap / "trajectory_1.json", agent_ended=False, turn_count=1)
    _write_trajectory(leap / "trajectory_2.json", turn_count=3)
    (leap / "final_output_2.txt").write_text("FAILED test_leap")

    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/two-fer": {
            "status": "pass_1", "stop_reason_1": "agent_end", "stop_reason_2": None,
            "turn_count": 4,
        },
        "python/leap": {
            "status": "fail", "stop_reason_1": "deadline", "stop_reason_2": "agent_end",
            "turn_count": 4,
        },
    }}))

    return {"log_root": log_root, "results_json": results_json}


def test_load_one_trajectory_per_exercise_using_latest_attempt(aider_run):
    """One NormalizedTrajectory per exercise (not per attempt file) -- the
    exercise's FINAL recorded status is what matters for scoring; attempt_1
    of a two-attempt exercise must not each become a separate example."""
    trajs = aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])
    assert {t.task_id for t in trajs} == {"python/two-fer", "python/leap"}


def test_load_maps_status_to_success_and_partial_score(aider_run):
    trajs = {t.task_id: t for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])}
    assert trajs["python/two-fer"].success is True
    assert trajs["python/two-fer"].partial_score == 1.0
    assert trajs["python/leap"].success is False
    assert trajs["python/leap"].partial_score == 0.0


def test_load_pass_2_gets_partial_credit_score(tmp_path):
    """A status of pass_2 (fixed on retry) scores 0.7, per the metric design
    in the plan -- distinguishes a clean first-try pass from a
    needed-a-nudge pass without conflating either with a hard failure."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "ex"
    ex.mkdir(parents=True)
    _write_trajectory(ex / "trajectory_1.json")
    _write_trajectory(ex / "trajectory_2.json")
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/ex": {"status": "pass_2", "stop_reason_1": "agent_end",
                       "stop_reason_2": "agent_end", "turn_count": 2},
    }}))
    trajs = aider_polyglot_ingest.load(log_root, results_json)
    assert trajs[0].success is True
    assert trajs[0].partial_score == 0.7


def test_load_uses_stop_reason_2_when_present_else_stop_reason_1(aider_run):
    trajs = {t.task_id: t for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])}
    assert trajs["python/leap"].stop_reason == "agent_end"      # stop_reason_2
    assert trajs["python/two-fer"].stop_reason == "agent_end"   # only stop_reason_1 exists


def test_load_components_used_degrades_gracefully_for_older_trajectories_without_notifications(aider_run):
    """The gap this test used to lock in permanently is now closed upstream
    (aider_polyglot.py's _dump_trajectory takes notifications=), but older
    trajectory_*.json files written before that change have no
    "notifications" key at all -- must degrade to [], not KeyError."""
    for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"]):
        assert t.components_used == []


def test_load_components_used_populates_when_notifications_present(tmp_path):
    """The gap is closed: when a trajectory_*.json DOES carry a
    "notifications" field (post-fix data), components_used must actually
    populate from it via the same merge_component_usage() helper
    gaia_ingest.py already uses -- not silently ignored."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "bash-heavy"
    ex.mkdir(parents=True)
    _write_trajectory(
        ex / "trajectory_1.json",
        notifications=[
            {"message": "skill-inject: +1 [bash]", "notifyType": "info"},
            {"message": "skill-inject: +1 [bash]", "notifyType": "info"},
        ],
    )
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/bash-heavy": {"status": "pass_1", "stop_reason_1": "agent_end",
                               "stop_reason_2": None, "turn_count": 1},
    }}))
    traj = aider_polyglot_ingest.load(log_root, results_json)[0]
    assert len(traj.components_used) == 1
    assert traj.components_used[0].pred_name == "skills_tools_bash"
    assert traj.components_used[0].invocation_count == 2


def test_load_skips_exercise_missing_from_results_json(tmp_path):
    """An exercise dir exists on disk (e.g. from a stale/aborted run) but has
    no entry in results_full_polyglot.json -- must be skipped, not crash."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "orphan"
    ex.mkdir(parents=True)
    _write_trajectory(ex / "trajectory_1.json")
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {}}))
    trajs = aider_polyglot_ingest.load(log_root, results_json)
    assert trajs == []


def test_load_missing_results_json_raises_clear_error(tmp_path):
    """Distinguish 'no results yet' (caller error, should be loud) from
    'exercise not yet scored' (expected, silent skip)."""
    with pytest.raises(FileNotFoundError):
        aider_polyglot_ingest.load(tmp_path, tmp_path / "does_not_exist.json")


# ── New result-record schema (post dev merge): stop_reasons is a list,
# generalized from the old fixed stop_reason_1/stop_reason_2 fields, and
# status can now be pass_N for any N (--max-attempts is no longer capped at
# 2). Old-format fixtures above (stop_reason_1/2, pass_1/pass_2 only) are
# kept as explicit backward-compat regression tests, not replaced -- real,
# already-captured data still uses that shape.

def test_load_uses_last_stop_reasons_list_entry_when_present(tmp_path):
    """New schema: aider_polyglot.py now records stop_reasons as a list
    (one entry per attempt that ran), not fixed stop_reason_1/2 fields."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "ex"
    ex.mkdir(parents=True)
    _write_trajectory(ex / "trajectory_1.json")
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/ex": {"status": "fail", "stop_reasons": ["deadline", "agent_end"],
                       "turn_count": 2},
    }}))
    traj = aider_polyglot_ingest.load(log_root, results_json)[0]
    assert traj.stop_reason == "agent_end"  # last entry, matching old stop_reason_2 semantics


def test_load_falls_back_to_old_stop_reason_fields_when_stop_reasons_absent(tmp_path):
    """Backward compat: real, already-captured data predates the
    stop_reasons list and only has stop_reason_1/stop_reason_2."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "ex"
    ex.mkdir(parents=True)
    _write_trajectory(ex / "trajectory_1.json")
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/ex": {"status": "fail", "stop_reason_1": "deadline",
                       "stop_reason_2": "process_exit", "turn_count": 2},
    }}))
    traj = aider_polyglot_ingest.load(log_root, results_json)[0]
    assert traj.stop_reason == "process_exit"


@pytest.mark.parametrize("status,expected_score", [
    ("pass_1", 1.0),
    ("pass_2", 0.7),
    ("pass_3", 0.4),
    ("pass_4", 0.4),   # floors rather than going negative/zero for a genuine pass
    ("pass_10", 0.4),
])
def test_load_generalizes_pass_n_scoring_beyond_pass_2(tmp_path, status, expected_score):
    """Real bug, confirmed against dev's generalized --max-attempts: attempt
    can now be pass_3, pass_4, ... for higher --max-attempts, not just
    pass_1/pass_2. A hardcoded 2-entry lookup silently treated any pass_3+
    as a FAILURE (score 0.0, success=False) -- scoring a genuine pass as a
    loss. Must generalize to any pass_N while still exactly preserving the
    documented pass_1=1.0/pass_2=0.7 values."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "ex"
    ex.mkdir(parents=True)
    _write_trajectory(ex / "trajectory_1.json")
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/ex": {"status": status, "stop_reasons": ["agent_end"], "turn_count": 1},
    }}))
    traj = aider_polyglot_ingest.load(log_root, results_json)[0]
    assert traj.success is True, f"{status} must be a genuine pass, not scored as a failure"
    assert traj.partial_score == expected_score


def test_load_picks_latest_attempt_by_numeric_not_lexicographic_order(tmp_path):
    """Real bug, confirmed by review: sorting trajectory_*.json by filename
    STRING put "trajectory_10" before "trajectory_2" (lexicographic), so
    with >=10 attempts the wrong (earlier) file supplied
    summarized_transcript/components_used/raw_paths for what should be the
    latest attempt."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "ex"
    ex.mkdir(parents=True)
    for n in range(1, 11):
        _write_trajectory(ex / f"trajectory_{n}.json", assistant_text=f"attempt {n}")
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/ex": {"status": "fail", "stop_reasons": ["agent_end"], "turn_count": 10},
    }}))
    traj = aider_polyglot_ingest.load(log_root, results_json)[0]
    assert "attempt 10" in traj.summarized_transcript
    assert traj.raw_paths["trajectory"].endswith("trajectory_10.json")


def test_load_components_used_is_union_across_all_attempts_not_just_latest(tmp_path):
    """Real gap, confirmed by review: a skill injected on an earlier, failed
    attempt but not re-triggered on the attempt that ultimately passed still
    genuinely influenced the outcome -- reading only the latest attempt's
    notifications silently lost that signal."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "two-attempts"
    ex.mkdir(parents=True)
    _write_trajectory(
        ex / "trajectory_1.json",
        notifications=[{"message": "skill-inject: +1 [bash]", "notifyType": "info"}],
    )
    _write_trajectory(
        ex / "trajectory_2.json",
        notifications=[{"message": "skill-inject: +1 [read]", "notifyType": "info"}],
    )
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/two-attempts": {"status": "pass_2", "stop_reasons": ["deadline", "agent_end"],
                                 "turn_count": 2},
    }}))
    traj = aider_polyglot_ingest.load(log_root, results_json)[0]
    names = {u.pred_name for u in traj.components_used}
    assert names == {"skills_tools_bash", "skills_tools_read"}
    # raw_paths/summarized_transcript still reflect only the latest attempt
    assert traj.raw_paths["trajectory"].endswith("trajectory_2.json")


def test_load_resolves_knowledge_inject_usage_when_repo_root_given(tmp_path):
    """Real bug, confirmed by review: knowledge-inject notification names are
    the topic FRONTMATTER FIELD (e.g. "Binary Search"), not a slug -- must
    resolve against the REAL skills/knowledge files via repo_root."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "ex"
    ex.mkdir(parents=True)
    _write_trajectory(
        ex / "trajectory_1.json",
        notifications=[{"message": "knowledge-inject: +1 [Binary Search]", "notifyType": "info"}],
    )
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/ex": {"status": "pass_1", "stop_reasons": ["agent_end"], "turn_count": 1},
    }}))
    real_repo_root = Path(__file__).parent.parent.parent.parent  # little-coder-self-improve/
    traj = aider_polyglot_ingest.load(log_root, results_json, repo_root=real_repo_root)[0]
    names = {u.pred_name for u in traj.components_used}
    assert names == {"skills_knowledge_binary_search"}


def test_load_skips_exercise_key_that_would_escape_log_root(tmp_path):
    """Real bug (hardening), confirmed by review: a corrupted/malicious
    "lang/exercise" key containing ".." would otherwise let ex_dir escape
    log_root entirely. results_full_polyglot.json is repo-controlled data
    today, but this is cheap to validate regardless."""
    log_root = tmp_path / "logs"
    log_root.mkdir(parents=True)
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "../../etc/passwd": {"status": "pass_1", "stop_reasons": ["agent_end"], "turn_count": 1},
    }}))
    trajs = aider_polyglot_ingest.load(log_root, results_json)
    assert trajs == []
