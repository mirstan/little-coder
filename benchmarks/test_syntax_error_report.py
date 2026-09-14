"""Tests for syntax_error_report -- entirely synthetic fixtures, never the
real (live-changing) benchmarks/harbor_runs/ tree."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import syntax_error_report as S  # noqa: E402


def _trial_dir(tmp_path, name, live_log_text):
    trial = tmp_path / name
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "little_coder.live.log").write_text(live_log_text)
    return trial


def test_is_trial_dir_true_only_with_a_live_log(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", ">> ShellSession({})\n")
    assert S.is_trial_dir(trial)
    assert not S.is_trial_dir(tmp_path)


def test_find_trial_dirs_on_a_run_directory_finds_every_trial(tmp_path):
    _trial_dir(tmp_path, "task-a__1", ">> ShellSession({})\n")
    _trial_dir(tmp_path, "task-b__2", ">> ShellSession({})\n")
    (tmp_path / "not_a_trial").mkdir()
    found = S.find_trial_dirs(tmp_path)
    assert sorted(p.name for p in found) == ["task-a__1", "task-b__2"]


def test_find_trial_dirs_on_a_trial_directory_returns_itself(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", ">> ShellSession({})\n")
    assert S.find_trial_dirs(trial) == [trial]


def test_python_syntax_error_is_counted():
    text = "  File \"f.py\", line 3\nSyntaxError: invalid syntax\n"
    assert S.count_errors(text)["python_errors"] == 1


def test_perl_syntax_error_is_counted():
    text = 'syntax error at /tmp/v3.pl line 25, near "$ctx]"\n'
    assert S.count_errors(text)["perl_errors"] == 1


def test_perl_global_symbol_error_is_counted():
    text = 'Global symbol "$foo" requires explicit package name\n'
    assert S.count_errors(text)["perl_errors"] == 1


def test_gcc_error_lines_are_counted_one_per_diagnostic():
    text = (
        "analyze.c:13:14: error: expected identifier or '(' before 'long'\n"
        "analyze.c:17:12: error: 'run' undeclared (first use in this function)\n"
    )
    assert S.count_errors(text)["gcc_errors"] == 2


def test_env_exec_timeout_is_not_mistaken_for_a_gcc_error():
    """The exact false positive found analyzing real trajectories: an unanchored
    `error: ` search matches env.exec's own timeout message."""
    text = "env.exec error: Command timed out after 30 seconds\n"
    assert S.count_errors(text)["gcc_errors"] == 0


def test_bash_syntax_error_is_counted():
    text = "bash: -c: line 1: syntax error near unexpected token `)'\n"
    assert S.count_errors(text)["bash_errors"] == 1


def test_shell_calls_are_counted_from_the_tool_call_marker(tmp_path):
    text = ">> ShellSession({'command': 'ls'})\n<< \n>> ShellSession({'command': 'pwd'})\n<< \n"
    trial = _trial_dir(tmp_path, "task__abc", text)
    row = S.trial_row(trial, "run")
    assert row["shell_calls"] == 2


def test_trial_row_computes_the_error_rate_as_a_percentage(tmp_path):
    text = (
        ">> ShellSession({})\n<< \n"
        ">> ShellSession({})\n<< syntax error at f.pl line 1, near \"x\"\n"
    )
    trial = _trial_dir(tmp_path, "task__abc", text)
    row = S.trial_row(trial, "my-run")
    assert row["shell_calls"] == 2
    assert row["perl_errors"] == 1
    assert row["total_errors"] == 1
    assert row["error_rate_pct"] == 50.0
    assert row["run_label"] == "my-run"


def test_trial_row_with_zero_shell_calls_does_not_divide_by_zero(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", "")
    row = S.trial_row(trial, "run")
    assert row["shell_calls"] == 0
    assert row["error_rate_pct"] == 0.0


def test_collect_rows_tags_each_trial_with_its_source_run_label(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    _trial_dir(before, "task__1", ">> ShellSession({})\n")
    _trial_dir(after, "task__1", ">> ShellSession({})\n")
    rows = S.collect_rows([before, after])
    assert {r["run_label"] for r in rows} == {"before", "after"}


def test_csv_round_trips_the_agreed_columns(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", ">> ShellSession({})\n<< SyntaxError: bad\n")
    rows = [S.trial_row(trial, "run")]
    out = tmp_path / "out.csv"
    S.write_csv(out, rows)
    import csv as csv_module

    with out.open() as handle:
        reader = csv_module.DictReader(handle)
        assert reader.fieldnames == S.CSV_COLUMNS
        read_rows = list(reader)
    assert read_rows[0]["python_errors"] == "1"
    assert read_rows[0]["trial"] == "task__abc"


def test_main_writes_csv_and_reports_nonzero_trials(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _trial_dir(run_dir, "task-a__1", ">> ShellSession({})\n<< SyntaxError: bad\n")
    _trial_dir(run_dir, "task-b__2", ">> ShellSession({})\n<< ok\n")
    out = tmp_path / "report.csv"
    code = S.main(["--out", str(out), str(run_dir)])
    assert code == 0
    assert out.exists()
    captured = capsys.readouterr().out
    assert "trials with >=1 syntax error: 1/2" in captured


def test_main_accepts_multiple_run_directories_for_a_before_after_comparison(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    _trial_dir(before, "task__1", ">> ShellSession({})\n<< SyntaxError: bad\n")
    _trial_dir(after, "task__1", ">> ShellSession({})\n<< ok\n")
    out = tmp_path / "report.csv"
    code = S.main(["--out", str(out), str(before), str(after)])
    assert code == 0
    text = out.read_text()
    assert "before" in text
    assert "after" in text


def test_main_rejects_a_path_that_is_not_a_directory(tmp_path):
    missing = tmp_path / "nope"
    try:
        S.main([str(missing)])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code != 0


def test_a_trial_directory_passed_directly_is_also_accepted(tmp_path, capsys):
    trial = _trial_dir(tmp_path, "task__abc", ">> ShellSession({})\n<< ok\n")
    out = tmp_path / "report.csv"
    code = S.main(["--out", str(out), str(trial)])
    assert code == 0
    import csv as csv_module

    with out.open() as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["trial"] == "task__abc"
