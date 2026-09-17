import json

from benchmarks.self_improve.spend_log import SpendLog, summarize


def test_log_writes_and_flushes_per_line(tmp_path):
    path = tmp_path / "spend_log.jsonl"
    with SpendLog(path) as log:
        log.run_start(argv=["run_gepa.py"])
        log.exercise(exercise_id="python/wordy", status="pass_1", score=1.0, memo_hit=False, duration_s=120.0)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "run_start"
    assert json.loads(lines[1])["event"] == "exercise"


def test_log_line_written_even_if_written_between_flushes(tmp_path):
    """Flushed after every write -- a SIGKILL mid-run must not lose
    already-completed records."""
    path = tmp_path / "spend_log.jsonl"
    log = SpendLog(path)
    log.exercise(exercise_id="python/a", status="pass_1", score=1.0)
    # Read the file WITHOUT closing the log -- if this weren't flushed per
    # line, the write could still be sitting in an internal buffer.
    assert "python/a" in path.read_text()
    log.close()


def test_summarize_totals_match_the_individual_records(tmp_path):
    path = tmp_path / "spend_log.jsonl"
    with SpendLog(path) as log:
        log.exercise(exercise_id="python/a", status="pass_1", score=1.0, memo_hit=False, duration_s=100.0)
        log.exercise(exercise_id="python/b", status="fail", score=0.0, memo_hit=True, duration_s=50.0)
        log.exercise(exercise_id="python/c", status="pass_1", score=1.0, memo_hit=False, duration_s=100.0)
    summary = summarize(path)
    assert summary["total_exercises"] == 3
    assert summary["memo_hits"] == 1
    assert summary["memo_hit_rate"] == 1 / 3
    assert summary["by_status"] == {"pass_1": 2, "fail": 1}
    assert summary["total_wall_s"] == 250.0


def test_summarize_ignores_non_exercise_events(tmp_path):
    path = tmp_path / "spend_log.jsonl"
    with SpendLog(path) as log:
        log.run_start(argv=[])
        log.iteration_end(accepted=True)
        log.exercise(exercise_id="python/a", status="pass_1", score=1.0)
        log.run_end(reason="completed")
    summary = summarize(path)
    assert summary["total_exercises"] == 1


def test_summarize_skips_a_truncated_final_line_instead_of_crashing(tmp_path):
    """Mirrors a SIGKILL mid-write -- the last line may be a partial JSON
    object. Must be skipped, not raise."""
    path = tmp_path / "spend_log.jsonl"
    with SpendLog(path) as log:
        log.exercise(exercise_id="python/a", status="pass_1", score=1.0)
    with open(path, "a") as fh:
        fh.write('{"event": "exercise", "status": "fail"')  # no closing brace, no newline
    summary = summarize(path)
    assert summary["total_exercises"] == 1


def test_summarize_handles_empty_log(tmp_path):
    path = tmp_path / "spend_log.jsonl"
    path.write_text("")
    summary = summarize(path)
    assert summary["total_exercises"] == 0
    assert summary["memo_hit_rate"] == 0.0
