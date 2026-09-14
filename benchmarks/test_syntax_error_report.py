"""Tests for syntax_error_report -- entirely synthetic fixtures, never the
real (live-changing) benchmarks/harbor_runs/ tree.

Fixtures deliberately match real little_coder.live.log shape exactly (the
"<< " prefix on the FIRST output line, the "[exit=... ]" footer, blank lines
between blocks) -- a fixture shaped any looser than the real thing is
precisely how the gcc line-anchor bug and the command-text false-positive
this file guards against both went unnoticed originally."""
from __future__ import annotations

import csv as csv_module
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import syntax_error_report as S  # noqa: E402


def _call(command: str, output: str, exit_code: int = 0) -> str:
    """One realistic `>> Tool(...)` / `<< ...` / `[exit=...]` block."""
    return (
        f">> ShellSession({{'command': {command!r}}})\n"
        f"<< {output}\n"
        f"[exit={exit_code} cwd=/app timed_out=false backend=harbor-env]\n"
        "\n\n"
    )


def _trial_dir(tmp_path, name, live_log_text):
    trial = tmp_path / name
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "little_coder.live.log").write_text(live_log_text)
    return trial


# --- path handling -----------------------------------------------------


def test_is_trial_dir_true_only_with_a_live_log(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", _call("ls", "f.txt"))
    assert S.is_trial_dir(trial)
    assert not S.is_trial_dir(tmp_path)


def test_find_trial_dirs_on_a_run_directory_finds_every_trial(tmp_path):
    _trial_dir(tmp_path, "task-a__1", _call("ls", "ok"))
    _trial_dir(tmp_path, "task-b__2", _call("ls", "ok"))
    (tmp_path / "not_a_trial").mkdir()
    trials, skipped = S.find_trial_dirs(tmp_path)
    assert sorted(p.name for p in trials) == ["task-a__1", "task-b__2"]
    assert skipped == []


def test_find_trial_dirs_on_a_trial_directory_returns_itself(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", _call("ls", "ok"))
    trials, skipped = S.find_trial_dirs(trial)
    assert trials == [trial]
    assert skipped == []


def test_a_trial_dir_shaped_directory_with_no_log_yet_is_reported_as_skipped(tmp_path):
    """A trial that crashed before its first write, or is still mid-setup:
    has agent/ but no live_log -- must not silently vanish from the count."""
    _trial_dir(tmp_path, "finished__1", _call("ls", "ok"))
    crashed = tmp_path / "crashed__2"
    (crashed / "agent").mkdir(parents=True)
    trials, skipped = S.find_trial_dirs(tmp_path)
    assert [p.name for p in trials] == ["finished__1"]
    assert [p.name for p in skipped] == ["crashed__2"]


def test_a_not_yet_ready_trial_dir_passed_directly_is_still_reported_as_skipped(tmp_path):
    """Without checking has_agent_dir_but_no_log first, a trial dir passed
    directly (not via its run-dir parent) falls through to treating its OWN
    children (agent/, result.json, ...) as if they were sibling trials --
    none of which satisfy is_trial_dir either, so it would silently vanish
    from both lists instead of landing in skipped."""
    crashed = tmp_path / "crashed__2"
    (crashed / "agent").mkdir(parents=True)
    trials, skipped = S.find_trial_dirs(crashed)
    assert trials == []
    assert skipped == [crashed]


# --- extraction: only real tool OUTPUT is scanned -----------------------


def test_perl_syntax_error_in_real_tool_output_is_counted():
    text = _call("perl -c v3.pl", 'syntax error at v3.pl line 25, near "$ctx]"', exit_code=255)
    errors, calls_with_errors = S.count_errors(text)
    assert errors["perl_errors"] == 1
    assert calls_with_errors == 1


def test_perl_global_symbol_error_is_counted():
    text = _call("perl -c v3.pl", 'Global symbol "$foo" requires explicit package name', exit_code=255)
    errors, _ = S.count_errors(text)
    assert errors["perl_errors"] == 1


def test_python_syntax_error_on_the_first_output_line_is_counted():
    """The exact shape a real traceback's first line has -- right after "<< "."""
    text = _call("python3 -c 'bad('", "SyntaxError: invalid syntax", exit_code=1)
    errors, _ = S.count_errors(text)
    assert errors["python_errors"] == 1


def test_gcc_error_on_the_first_output_line_is_counted():
    """Regression test for the real bug: a `^`-anchored pattern can never
    match here, because the first output line always starts with "<< ", not
    at column 0 -- the diagnostic being the very first thing gcc prints
    (a file-scope error, e.g. a stray token) is a realistic shape."""
    text = _call("gcc -fsyntax-only a.c", "a.c:1:1: error: expected identifier", exit_code=1)
    errors, _ = S.count_errors(text)
    assert errors["gcc_errors"] == 1


def test_gcc_error_lines_are_counted_one_per_diagnostic():
    text = _call(
        "gcc -fsyntax-only analyze.c",
        "analyze.c:13:14: error: expected identifier or '(' before 'long'\n"
        "analyze.c:17:12: error: 'run' undeclared (first use in this function)",
        exit_code=1,
    )
    errors, _ = S.count_errors(text)
    assert errors["gcc_errors"] == 2


def test_gcc_error_matches_a_hyphenated_filename():
    text = _call("gcc -fsyntax-only run-sim.c", "run-sim.c:4:1: error: bad", exit_code=1)
    errors, _ = S.count_errors(text)
    assert errors["gcc_errors"] == 1


def test_cobol_syntax_error_is_counted():
    """Real example (cobol-modernization__ZadiHUd): missing PROGRAM-ID."""
    text = _call(
        "cobc -x -o out src/program.cbl",
        "src/program.cbl:3: error: syntax error, unexpected ENVIRONMENT, expecting FUNCTION-ID or PROGRAM-ID",
        exit_code=1,
    )
    errors, _ = S.count_errors(text)
    assert errors["cobol_errors"] == 1


def test_bash_syntax_error_is_counted():
    text = _call("bash -n f.sh", "f.sh: line 1: syntax error near unexpected token `)'", exit_code=2)
    errors, _ = S.count_errors(text)
    assert errors["bash_errors"] == 1


def test_env_exec_timeout_is_not_mistaken_for_a_gcc_error():
    """The exact false positive found analyzing real trajectories."""
    text = _call("sleep 999", "env.exec error: Command timed out after 30 seconds", exit_code=-1)
    errors, _ = S.count_errors(text)
    assert errors["gcc_errors"] == 0


def test_command_text_is_never_scanned_even_when_it_names_a_real_error_pattern():
    """The bug this design fixes: the agent's own command can contain a
    heredoc'd script whose *source* legitimately mentions these exact
    phrases -- e.g. an except-SyntaxError guard, or a self-check wrapper the
    agent writes specifically because it's trying to avoid this class of
    mistake. None of that is a real error and must not be counted."""
    command = (
        "cat > check.py <<'EOF'\n"
        "try:\n"
        "    compile(src, 'f', 'exec')\n"
        "except SyntaxError:\n"
        "    print('bad')\n"
        "EOF\n"
        "perl -c f.pl && echo 'no syntax error at all'"
    )
    text = _call(command, "no syntax error at all", exit_code=0)
    errors, calls_with_errors = S.count_errors(text)
    assert errors["python_errors"] == 0
    assert errors["perl_errors"] == 0
    assert calls_with_errors == 0


def test_narration_between_calls_is_never_scanned():
    text = (
        "The last attempt failed with a SyntaxError, let me look closer.\n\n"
        + _call("cat f.py", "print('ok')")
    )
    errors, _ = S.count_errors(text)
    assert errors["python_errors"] == 0


def test_an_unterminated_final_block_with_no_exit_footer_is_not_matched():
    """A trial killed mid-command: no footer ever arrives. Must not match
    into it and must not crash."""
    text = _call("ls", "ok") + ">> ShellSession({'command': 'perl -c f.pl'})\n<< syntax error at f.pl line 1"
    errors, calls_with_errors = S.count_errors(text)
    assert errors["perl_errors"] == 0
    assert calls_with_errors == 0


def test_a_footerless_result_does_not_swallow_a_later_calls_command_text():
    """Real bug: non-shell tools ("write", "webfetch", "websearch") produce
    no [exit=...] footer at all -- in every real trial checked, `>> ` calls
    outnumber `[exit=` footers. A naive non-greedy match would keep looking
    past a footerless result for the NEXT footer anywhere later in the file,
    swallowing an intervening ShellSession call's own command text (which
    can legitimately contain the word "SyntaxError", e.g. writing a script
    with an except-SyntaxError guard) as if it were the first result's
    output -- the exact confound this file exists to avoid."""
    text = (
        ">> write({'path': '/tmp/f.py'})\n"
        "<< wrote 40 lines\n"
        "\n\n"
        + _call("python3 -c \"raise SyntaxError('boom')\"", "SyntaxError: boom", exit_code=1)
    )
    errors, calls_with_errors = S.count_errors(text)
    assert errors["python_errors"] == 1  # only the real interpreter error
    assert calls_with_errors == 1


def test_a_footerless_result_produces_no_match_of_its_own():
    text = ">> write({'path': '/tmp/f.py'})\n<< wrote 40 lines\n\n\n"
    assert S.result_text(text) == ""


# --- shell-call counting -------------------------------------------------


def test_shell_calls_are_counted_from_the_tool_call_marker():
    text = _call("ls", "ok") + _call("pwd", "/app")
    errors_text = S.result_text(text)
    assert errors_text.count("ok") == 1 and errors_text.count("/app") == 1
    assert len(S._SHELL_CALL_RE.findall(text)) == 2


def test_shell_calls_also_count_gaias_bash_tool_name():
    """GAIA trials (benchmarks/gaia.py's ALLOWED_TOOLS) call the shell tool
    "bash"/"Bash", not "ShellSession"."""
    text = ">> bash({'command': 'ls'})\n<< ok\n[exit=0]\n\n>> Bash({'command': 'pwd'})\n<< /app\n[exit=0]\n\n"
    assert len(S._SHELL_CALL_RE.findall(text)) == 2


# --- trial_row / rate semantics ------------------------------------------


def test_trial_row_computes_the_calls_with_errors_rate(tmp_path):
    text = _call("ls", "ok") + _call("perl -c f.pl", 'syntax error at f.pl line 1, near "x"', exit_code=255)
    trial = _trial_dir(tmp_path, "task__abc", text)
    row = S.trial_row(trial, "my-run")
    assert row["shell_calls"] == 2
    assert row["perl_errors"] == 1
    assert row["total_errors"] == 1
    assert row["calls_with_errors"] == 1
    assert row["error_rate_pct"] == 50.0
    assert row["run_label"] == "my-run"


def test_error_rate_pct_cannot_exceed_100_percent_even_with_many_diagnostics_in_one_call(tmp_path):
    """A single call's output can contain several diagnostics (e.g. one gcc
    invocation printing 4 errors) -- the rate must stay a real percentage,
    counting affected CALLS, not raw diagnostic lines."""
    text = _call(
        "gcc -fsyntax-only a.c",
        "\n".join(f"a.c:{n}:1: error: bad" for n in range(1, 6)),
        exit_code=1,
    )
    trial = _trial_dir(tmp_path, "task__abc", text)
    row = S.trial_row(trial, "run")
    assert row["shell_calls"] == 1
    assert row["gcc_errors"] == 5
    assert row["calls_with_errors"] == 1
    assert row["error_rate_pct"] == 100.0


def test_trial_row_with_zero_shell_calls_does_not_divide_by_zero(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", "")
    row = S.trial_row(trial, "run")
    assert row["shell_calls"] == 0
    assert row["error_rate_pct"] == 0.0


# --- multi-run / before-after ---------------------------------------------


def test_collect_rows_tags_each_trial_with_its_source_run_label(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    _trial_dir(before, "task__1", _call("ls", "ok"))
    _trial_dir(after, "task__1", _call("ls", "ok"))
    rows, skipped = S.collect_rows([before, after])
    labels = {r["run_label"] for r in rows}
    assert len(labels) == 2  # distinct even though both trial dirs share a name
    assert skipped == []


def test_run_labels_do_not_collide_when_two_paths_share_a_basename(tmp_path):
    """The exact shape a before/after comparison invites: .../before/job-7
    and .../after/job-7 -- same basename, different parent."""
    a = tmp_path / "before" / "job-7"
    b = tmp_path / "after" / "job-7"
    _trial_dir(a, "task__1", _call("ls", "ok"))
    _trial_dir(b, "task__1", _call("ls", "ok"))
    rows, _ = S.collect_rows([a, b])
    assert rows[0]["run_label"] != rows[1]["run_label"]


# --- CSV / summary / CLI ---------------------------------------------------


def test_csv_round_trips_the_agreed_columns(tmp_path):
    text = _call("python3 bad.py", "SyntaxError: bad", exit_code=1)
    trial = _trial_dir(tmp_path, "task__abc", text)
    rows = [S.trial_row(trial, "run")]
    out = tmp_path / "out.csv"
    S.write_csv(out, rows)
    with out.open() as handle:
        reader = csv_module.DictReader(handle)
        assert reader.fieldnames == S.CSV_COLUMNS
        read_rows = list(reader)
    assert read_rows[0]["python_errors"] == "1"
    assert read_rows[0]["trial"] == "task__abc"


def test_main_writes_csv_and_reports_nonzero_trials(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _trial_dir(run_dir, "task-a__1", _call("python3 bad.py", "SyntaxError: bad", exit_code=1))
    _trial_dir(run_dir, "task-b__2", _call("ls", "ok"))
    out = tmp_path / "report.csv"
    code = S.main(["--out", str(out), str(run_dir)])
    assert code == 0
    assert out.exists()
    captured = capsys.readouterr().out
    assert "trials with >=1 syntax error: 1/2" in captured


def test_main_accepts_multiple_run_directories_for_a_before_after_comparison(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    _trial_dir(before, "task__1", _call("python3 bad.py", "SyntaxError: bad", exit_code=1))
    _trial_dir(after, "task__1", _call("ls", "ok"))
    out = tmp_path / "report.csv"
    code = S.main(["--out", str(out), str(before), str(after)])
    assert code == 0
    text = out.read_text()
    assert "before" in text
    assert "after" in text


def test_main_rejects_a_path_that_is_not_a_directory(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(SystemExit) as excinfo:
        S.main([str(missing)])
    assert excinfo.value.code != 0


def test_a_trial_directory_passed_directly_is_also_accepted(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", _call("ls", "ok"))
    out = tmp_path / "report.csv"
    code = S.main(["--out", str(out), str(trial)])
    assert code == 0
    with out.open() as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["trial"] == "task__abc"


def test_main_returns_nonzero_and_does_not_overwrite_an_existing_csv_when_nothing_is_found(tmp_path):
    empty_dir = tmp_path / "nothing_here"
    empty_dir.mkdir()
    out = tmp_path / "report.csv"
    out.write_text("a previous good report\n")
    code = S.main(["--out", str(out), str(empty_dir)])
    assert code == 1
    assert out.read_text() == "a previous good report\n"


def test_main_creates_the_output_directory_if_it_does_not_exist(tmp_path):
    trial = _trial_dir(tmp_path, "task__abc", _call("ls", "ok"))
    out = tmp_path / "nested" / "dir" / "report.csv"
    code = S.main(["--out", str(out), str(trial)])
    assert code == 0
    assert out.exists()


def test_print_summary_reports_skipped_trial_shaped_dirs(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _trial_dir(run_dir, "finished__1", _call("ls", "ok"))
    crashed = run_dir / "crashed__2"
    (crashed / "agent").mkdir(parents=True)
    out = tmp_path / "report.csv"
    S.main(["--out", str(out), str(run_dir)])
    captured = capsys.readouterr().out
    assert "skipped 1 trial-shaped dir" in captured
    assert "crashed__2" in captured
