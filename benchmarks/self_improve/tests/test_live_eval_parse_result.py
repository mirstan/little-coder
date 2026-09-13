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
