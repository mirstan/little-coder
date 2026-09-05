from pathlib import Path

from benchmarks.self_improve.ingest import gaia_ingest


def test_load_returns_one_trajectory_per_task_dir(gaia_run):
    trajs = gaia_ingest.load(gaia_run)
    assert {t.task_id for t in trajs} == {"task-001", "task-002"}


def test_load_maps_correct_field_to_success(gaia_run):
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    assert trajs["task-001"].success is True
    assert trajs["task-002"].success is False


def test_load_sets_benchmark_literal(gaia_run):
    for t in gaia_ingest.load(gaia_run):
        assert t.benchmark == "gaia"


def test_load_extracts_components_used_from_notifications(gaia_run):
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    names = {u.pred_name for u in trajs["task-001"].components_used}
    assert "skills_tools_bash" in names
    assert trajs["task-002"].components_used == []


def test_load_sets_failure_signal_on_stderr_present(gaia_run):
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    assert "process_exit" in trajs["task-002"].failure_signals or \
           "harness_error" in trajs["task-002"].failure_signals


def test_load_skips_task_dir_missing_result_json(tmp_path):
    """A task dir with no result.json (interrupted run) must be skipped,
    not crash the whole ingest -- log a warning instead."""
    (tmp_path / "task-incomplete").mkdir()
    (tmp_path / "task-incomplete" / "transcript.txt").write_text("partial")
    trajs = gaia_ingest.load(tmp_path)
    assert trajs == []


def test_load_ignores_manifest_and_results_json_at_root(gaia_run):
    """manifest.json / results.json / submission.jsonl live at the run root,
    not inside a task dir -- must not be misread as task directories."""
    (gaia_run / "manifest.json").write_text("{}")
    (gaia_run / "results.json").write_text("{}")
    (gaia_run / "submission.jsonl").write_text("")
    trajs = gaia_ingest.load(gaia_run)
    assert len(trajs) == 2  # unchanged from task-001/task-002


def test_load_raw_paths_point_back_to_originals(gaia_run):
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    assert Path(trajs["task-001"].raw_paths["transcript"]).exists()


def test_load_defaults_stop_reason_when_absent(gaia_run):
    """gaia's per-task result.json does not carry stop_reason (confirmed gap,
    TDD_SPEC.md §3.2) -- must default rather than KeyError."""
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    assert trajs["task-001"].stop_reason == "agent_end"
    assert trajs["task-002"].stop_reason != ""


def test_load_skips_task_with_malformed_result_json(tmp_path):
    """Real bug, confirmed by review: an unguarded json.loads() on a
    truncated/malformed result.json (e.g. harness killed mid-write) crashed
    the whole ingestion run instead of skipping just that one task."""
    good = tmp_path / "task-good"
    good.mkdir()
    (good / "result.json").write_text('{"correct": true}')

    bad = tmp_path / "task-bad"
    bad.mkdir()
    (bad / "result.json").write_text("not valid json{{{")

    trajs = gaia_ingest.load(tmp_path)
    assert {t.task_id for t in trajs} == {"task-good"}


def test_load_skips_malformed_line_in_tool_calls_jsonl(tmp_path):
    """Real bug, confirmed by review: a partially-flushed final line in an
    append-only tool_calls.jsonl (the most likely corruption from a crashed
    run) crashed the whole task's ingestion instead of just skipping that
    one line."""
    task = tmp_path / "task-001"
    task.mkdir()
    (task / "result.json").write_text('{"correct": true}')
    (task / "tool_calls.jsonl").write_text(
        '{"name": "bash", "args": {}}\n'
        '{"name": "read", "args": {}\n'  # truncated mid-write, invalid JSON
    )
    trajs = gaia_ingest.load(tmp_path)
    assert len(trajs) == 1
    assert trajs[0].turn_count == 1  # only the valid line counted


def test_load_handles_missing_log_root_gracefully(tmp_path):
    """load() must never raise on a missing/non-directory log_root --
    matches the module's own graceful-degradation contract."""
    assert gaia_ingest.load(tmp_path / "does-not-exist") == []
