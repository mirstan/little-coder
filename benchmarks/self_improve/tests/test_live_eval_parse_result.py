"""Unit tests for PolyglotLiveRunner._parse_result() against a synthetic
results.json -- no git, no subprocess, no fake_pi needed, since
_parse_result is a pure function of (results_file, log_root) once no
trajectory workdir is present to trigger the diff-computation branch."""
import json

from benchmarks.self_improve.exercises import ExerciseSpec
from benchmarks.self_improve.live_eval import _MAX_TRANSCRIPT_CHARS, PolyglotLiveRunner


def _bare_runner() -> PolyglotLiveRunner:
    """_parse_result only touches self.* via _compute_diff, which is never
    reached unless a trajectory workdir exists on disk -- these tests never
    create one, so a runner with no attributes set is safe here."""
    return object.__new__(PolyglotLiveRunner)


def _write_results(results_file, results_key, record):
    results_file.write_text(json.dumps({"exercises": {results_key: record}}))


def test_parse_result_caps_cumulative_self_reported_lessons_length(tmp_path):
    """Each individual LESSON: line is capped
    at aider_polyglot.py's own LESSON_MAX_CHARS (500) when extracted, but
    with --max-attempts set high the cumulative joined text this adapter
    passes into GEPA reflection feedback had no overall cap -- a long chain
    of near-500-char lessons could still overflow reflection context."""
    runner = _bare_runner()
    results_file = tmp_path / "results.json"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    lessons = ["x" * 500 for _ in range(20)]  # 10,000 raw chars, well over the cap
    _write_results(results_file, "pi/python/wordy", {"status": "pass_1", "lessons": lessons})

    spec = ExerciseSpec("wordy")
    result = runner._parse_result(spec, results_file, log_root, stderr="", exit_code=0)

    total_len = sum(len(lesson) for lesson in result.self_reported_lessons)
    assert total_len <= _MAX_TRANSCRIPT_CHARS
    # Earlier lessons survive intact (in order) up to the budget, rather than
    # e.g. truncating every entry uniformly or dropping the earliest ones.
    assert result.self_reported_lessons[0] == "x" * 500


def test_parse_result_leaves_short_lessons_list_untouched(tmp_path):
    runner = _bare_runner()
    results_file = tmp_path / "results.json"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    _write_results(results_file, "pi/python/wordy", {"status": "pass_2", "lessons": ["needed clearer guidance"]})

    spec = ExerciseSpec("wordy")
    result = runner._parse_result(spec, results_file, log_root, stderr="", exit_code=0)

    assert result.self_reported_lessons == ["needed clearer guidance"]


def test_parse_result_defaults_to_empty_lessons_when_field_absent(tmp_path):
    runner = _bare_runner()
    results_file = tmp_path / "results.json"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    _write_results(results_file, "pi/python/wordy", {"status": "pass_1"})

    spec = ExerciseSpec("wordy")
    result = runner._parse_result(spec, results_file, log_root, stderr="", exit_code=0)

    assert result.self_reported_lessons == []


def test_parse_result_carries_the_records_reason_into_error(tmp_path):
    """aider_polyglot.py records why an exercise ended in "error" (e.g.
    "exercise not found at ...") under "reason"; run_batch() needs it to tell
    a config error from a runtime one, and reflection feedback needs a cause."""
    runner = _bare_runner()
    results_file = tmp_path / "results.json"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    _write_results(results_file, "pi/python/wordy", {"status": "error", "reason": "X"})

    result = runner._parse_result(ExerciseSpec("wordy"), results_file, log_root, stderr="", exit_code=0)

    assert result.status == "error"
    assert result.error == "X"


def test_parse_result_leaves_error_unset_when_the_record_has_no_reason(tmp_path):
    runner = _bare_runner()
    results_file = tmp_path / "results.json"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    _write_results(results_file, "pi/python/wordy", {"status": "fail"})

    result = runner._parse_result(ExerciseSpec("wordy"), results_file, log_root, stderr="", exit_code=0)

    assert result.error is None


def test_parse_result_carries_the_tamper_flag_and_capped_reasons(tmp_path):
    runner = _bare_runner()
    results_file = tmp_path / "results.json"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    reasons = [f"protected file 'x{i}_test.py' was modified " + "y" * 300 for i in range(15)]
    _write_results(results_file, "pi/python/wordy",
                   {"status": "fail", "tests_tampered": True, "tamper_reasons": reasons})

    result = runner._parse_result(ExerciseSpec("wordy"), results_file, log_root, stderr="", exit_code=0)

    assert result.tests_tampered is True
    assert len(result.tamper_reasons) == 10
    assert all(len(r) <= 200 for r in result.tamper_reasons)
    assert result.tamper_reasons[0].startswith("protected file 'x0_test.py'")


def test_parse_result_ignores_malformed_tamper_fields(tmp_path):
    runner = _bare_runner()
    results_file = tmp_path / "results.json"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    _write_results(results_file, "pi/python/wordy",
                   {"status": "fail", "tests_tampered": "yes", "tamper_reasons": ["ok", 3, None]})
    result = runner._parse_result(ExerciseSpec("wordy"), results_file, log_root, stderr="", exit_code=0)
    assert result.tests_tampered is False
    assert result.tamper_reasons == ["ok"]

    _write_results(results_file, "pi/python/wordy", {"status": "fail", "tamper_reasons": "x"})
    result = runner._parse_result(ExerciseSpec("wordy"), results_file, log_root, stderr="", exit_code=0)
    assert result.tests_tampered is False and result.tamper_reasons == []


def test_compute_diff_does_not_follow_a_symlink_in_the_snapshot(tmp_path):
    """aider_polyglot's trajectory snapshot keeps symlinks as symlinks; one
    pointing at /dev/zero must not be read here."""
    runner = _bare_runner()
    root = tmp_path / "bench"
    pristine = root / "python" / "exercises" / "practice" / "wordy"
    pristine.mkdir(parents=True)
    (pristine / "wordy.py").write_text("stub\n")
    runner.benchmark_root = root
    workdir = tmp_path / "workdir_1"
    workdir.mkdir()
    (workdir / "wordy.py").write_text("solved\n")
    (workdir / "zero.py").symlink_to("/dev/zero")
    diff = runner._compute_diff(ExerciseSpec("wordy"), workdir)
    assert "+solved" in diff
    assert "zero.py" not in diff


import pytest


@pytest.mark.parametrize("payload", [
    [],
    "not an object",
    {"exercises": []},
    {"exercises": "x"},
    {"exercises": {"pi/python/wordy": []}},
    {"exercises": {"pi/python/wordy": "pass_1"}},
    {"exercises": {"pi/python/wordy": {"status": 5}}},
    {"exercises": {"pi/python/wordy": {"status": ["pass_1"]}}},
    # Raw bytes, written as-is: not UTF-8 (UnicodeDecodeError is a
    # ValueError, not a JSONDecodeError), and nested deep enough for
    # RecursionError. Neither starts with a UTF-16 byte-order mark.
    pytest.param(b'{"exercises": {"pi/python/wordy": {"status": "pass_1\xff"}}}', id="not-utf8"),
    pytest.param(b"[" * 100_000, id="deeply-nested"),
])
def test_parse_result_treats_a_schema_invalid_results_file_as_a_harness_error(tmp_path, payload):
    """Valid JSON of the wrong shape used to raise AttributeError/TypeError
    out of _parse_result, past run_batch()'s harness_error retry path."""
    runner = _bare_runner()
    results_file = tmp_path / "results.json"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    if isinstance(payload, bytes):
        results_file.write_bytes(payload)
    else:
        results_file.write_text(json.dumps(payload))

    result = runner._parse_result(ExerciseSpec("wordy"), results_file, log_root, stderr="", exit_code=0)

    assert result.status == "harness_error"
    assert result.score == 0.0 and result.success is False
    assert result.error
