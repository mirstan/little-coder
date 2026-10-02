"""Two scoring guards that only self-improve's live_eval turns on.

POLYGLOT_CRASH_IS_ERROR=1: a pi crash on any attempt of an exercise that did
not pass is an "error" (retried, never cached), not a "fail" GEPA learns from.
POLYGLOT_RESTORE_TESTS=1: each attempt is scored in a tree the harness builds
itself -- the agent's solution bytes plus pristine copies of everything else
it was given, with files the agent added left out -- by a runner whose pass
needs a verified report, so an agent that edits its tests or drops a runner
hook cannot manufacture a pass.
Both change the benchmark's accounting, so a plain run leaves them off.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aider_polyglot as AP  # noqa: E402


# ── crash on a later attempt ─────────────────────────────────────────────
def test_a_later_crash_stays_a_fail_by_default():
    assert AP._classify_status(False, None, ["completed", "process_exit"]) == "fail"


def test_a_later_crash_is_an_error_when_opted_in():
    def status(passed, attempt, outcomes):
        return AP._classify_status(passed, attempt, outcomes, crash_is_error=True)
    assert status(False, None, ["completed", "process_exit"]) == "error"
    assert status(False, None, ["empty_response", "process_exit"]) == "error"
    # a pass still wins, and a crash-free failure is still a fail
    assert status(True, "pass_2", ["completed", "process_exit"]) == "pass_2"
    assert status(False, None, ["completed", "completed"]) == "fail"
    assert status(False, None, ["completed", "deadline"]) == "fail_timeout"


def test_run_exercise_reads_the_crash_gate(monkeypatch, tmp_path):
    """A pi crash on attempt 2 of a failing exercise, end to end."""
    monkeypatch.setenv("POLYGLOT_CRASH_IS_ERROR", "1")
    monkeypatch.delenv("POLYGLOT_RESTORE_TESTS", raising=False)
    _exercise(monkeypatch, tmp_path)
    reasons = iter(["agent_end", "process_exit"])

    class FakeRpc:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def notifications(self):
            return []

        def prompt_and_collect(self, message, timeout=900):
            return AP.PromptResult(turn_count=2, agent_ended=True, stop_reason=next(reasons),
                                   assistant_text="tried", tool_calls=[{"name": "edit"}])

    monkeypatch.setattr(AP, "PiRpc", FakeRpc)
    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False,
                              retry=True, max_attempts=2)
    assert record["stop_reasons"] == ["agent_end", "process_exit"]
    assert record["status"] == "error"


# ── edited tests (fake descriptor) ───────────────────────────────────────
def _exercise(monkeypatch, tmp_path, *, score_in_copy=True, config_tests=None):
    """A python-shaped exercise whose tests pass only if ex_test.py says so,
    and an agent that 'solves' it by rewriting ex_test.py."""
    src = tmp_path / "practice" / "ex"
    (src / ".meta").mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("real tests")
    (src / "test_utils.py").write_text("real helper")
    if config_tests is not None:
        (src / ".meta" / "config.json").write_text(json.dumps({"files": {"test": config_tests}}))

    def prepare(s, w):
        AP._copy_exercise(s, w)
        # like _prepare_python: test_utils.py is not *_test.py, so it is a "stub"
        return [w / "ex.py", w / "test_utils.py"], [w / "ex_test.py"]

    def run_tests(work, timeout):
        ok = (work / "ex_test.py").read_text() == "trivial" or \
             (work / "test_utils.py").read_text() == "trivial"
        return ok, "ok" if ok else "FAILED"

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "score_in_copy": score_in_copy,
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": run_tests,
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")


def _run(monkeypatch, tmp_path, tamper="ex_test.py"):
    seen = {}

    class FakeRpc:
        def __init__(self, *a, cwd=None, **kw):
            self.cwd = cwd

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def notifications(self):
            return []

        def prompt_and_collect(self, message, timeout=900):
            (Path(self.cwd) / tamper).write_text("trivial")
            seen["work"] = Path(self.cwd)
            seen["after"] = (Path(self.cwd) / tamper).read_text()
            return AP.PromptResult(turn_count=2, agent_ended=True, stop_reason="agent_end",
                                   assistant_text="edited the test")

    orig = AP._score_gated

    def spy(desc, work, timeout, m):
        out = orig(desc, work, timeout, m)
        seen["after_scoring"] = (work / tamper).read_text()
        return out

    monkeypatch.setattr(AP, "_score_gated", spy)
    monkeypatch.setattr(AP, "PiRpc", FakeRpc)
    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False,
                              retry=True, max_attempts=2)
    return record, seen


def test_edited_tests_still_pass_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("POLYGLOT_RESTORE_TESTS", raising=False)
    _exercise(monkeypatch, tmp_path)
    record, _ = _run(monkeypatch, tmp_path)
    assert record["status"] == "pass_1"
    assert "tests_tampered" not in record
    assert "tamper_reasons" not in record and "added_files" not in record


@pytest.mark.parametrize("score_in_copy", [True, False])
def test_edited_tests_are_scored_from_the_pristine_copy_when_opted_in(
        monkeypatch, tmp_path, score_in_copy):
    """score_in_copy no longer matters when gated: both score a harness-built
    tree and both leave the agent's own tree exactly as the agent wrote it."""
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    _exercise(monkeypatch, tmp_path, score_in_copy=score_in_copy)
    record, seen = _run(monkeypatch, tmp_path)
    assert record["status"] == "fail"
    assert record["tests_tampered"] is True
    assert any("ex_test.py" in r for r in record["tamper_reasons"])
    assert seen["after"] == "trivial"
    assert seen["after_scoring"] == "trivial"


def test_config_listed_test_fixtures_are_restored_too(monkeypatch, tmp_path):
    """paasio ships test_utils.py as a test file in .meta/config.json, though
    _prepare_python's *_test.py glob hands it to the agent as a stub."""
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    _exercise(monkeypatch, tmp_path, config_tests=["ex_test.py", "test_utils.py"])
    record, _ = _run(monkeypatch, tmp_path, tamper="test_utils.py")
    assert record["status"] == "fail"
    assert record["tests_tampered"] is True


def test_untouched_tests_are_not_reported_as_tampered(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    _exercise(monkeypatch, tmp_path)
    record, _ = _run(monkeypatch, tmp_path, tamper="ex.py")
    assert record["status"] == "fail"
    assert record["tests_tampered"] is False
    assert record["tamper_reasons"] == [] and record["added_files"] == []


def test_scoring_params_record_the_gates_only_when_on(monkeypatch):
    desc = {"timeout_s": 5}
    monkeypatch.delenv("POLYGLOT_CRASH_IS_ERROR", raising=False)
    monkeypatch.delenv("POLYGLOT_RESTORE_TESTS", raising=False)
    off = AP._scoring_params("m", "python", True, desc)
    assert "crash_is_error" not in off and "restore_tests" not in off
    assert "scoring_interpreter" not in off
    monkeypatch.setenv("POLYGLOT_CRASH_IS_ERROR", "1")
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    on = AP._scoring_params("m", "python", True, desc)
    assert on["crash_is_error"] is True and on["restore_tests"] == "manifest-v2"


def test_scoring_params_record_the_scoring_interpreter_when_gated(monkeypatch):
    """A7: the gated python runner scores with sys.executable, so which
    interpreter (and pytest) did the scoring is a scoring input."""
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    on = AP._scoring_params("m", "python", True, AP.LANG_DESCRIPTORS["python"])
    interp = on["scoring_interpreter"]
    assert interp["python_executable"] == sys.executable
    assert interp["python_version"] == sys.version.split()[0]
    assert interp["pytest_version"] == pytest.__version__
    monkeypatch.delenv("POLYGLOT_RESTORE_TESTS")
    off = AP._scoring_params("m", "python", True, AP.LANG_DESCRIPTORS["python"])
    assert "scoring_interpreter" not in off


# ── the real python runner on a real exercise ───────────────────────────
_STUB = "def two_fer(name='you'):\n    pass\n"
_EXAMPLE = "def two_fer(name='you'):\n    return f'One for {name}, one for me.'\n"
_TEST = '''import unittest

from two_fer import two_fer


class TwoFerTest(unittest.TestCase):
    def test_no_name_given(self):
        self.assertEqual(two_fer(), "One for you, one for me.")

    def test_a_name_given(self):
        self.assertEqual(two_fer("Alice"), "One for Alice, one for me.")
'''


def _py_exercise(tmp_path):
    src = tmp_path / "practice" / "two-fer"
    (src / ".meta").mkdir(parents=True)
    (src / ".docs").mkdir()
    (src / ".docs" / "instructions.md").write_text("say two-fer\n")
    (src / "two_fer.py").write_text(_STUB)
    (src / "two_fer_test.py").write_text(_TEST)
    (src / ".meta" / "example.py").write_text(_EXAMPLE)
    (src / ".meta" / "config.json").write_text(json.dumps({"files": {
        "solution": ["two_fer.py"], "test": ["two_fer_test.py"], "example": [".meta/example.py"]}}))
    return src


def _gated(tmp_path, mutate=None, desc=None):
    desc = desc or AP.LANG_DESCRIPTORS["python"]
    src = _py_exercise(tmp_path)
    work = tmp_path / "work" / "two-fer"
    stubs, tests = AP._prepare_python(src, work)
    m = AP._snapshot(src, work, stubs, tests, desc)
    if mutate is not None:
        mutate(work)
    return AP._score_gated(desc, work, 60, m)


def _honest(work):
    (work / "two_fer.py").write_text(_EXAMPLE)


def _add(name, text, honest=False):
    def mutate(work):
        if honest:
            _honest(work)
        (work / name).write_text(text)
    return mutate


def test_honest_solution_passes(tmp_path):
    passed, out, reasons, notes, added = _gated(tmp_path, _honest)
    assert passed, out
    assert reasons == [] and notes == [] and added == []


def test_untouched_stub_fails(tmp_path):
    passed, _, reasons, _, _ = _gated(tmp_path)
    assert not passed and reasons == []


def test_snapshot_keeps_tests_pristine_and_names_the_solution(tmp_path):
    src = _py_exercise(tmp_path)
    work = tmp_path / "work" / "two-fer"
    stubs, tests = AP._prepare_python(src, work)
    m = AP._snapshot(src, work, stubs, tests, AP.LANG_DESCRIPTORS["python"])
    assert m.solution == frozenset({"two_fer.py"})
    assert m.tests == frozenset({"two_fer_test.py"})
    assert "two_fer.py" not in m.files and "two_fer_test.py" in m.files
    assert m.expected_tests == frozenset({
        ("two_fer_test.TwoFerTest", "test_no_name_given"),
        ("two_fer_test.TwoFerTest", "test_a_name_given"),
    })


@pytest.mark.parametrize("name,text", [
    ("conftest.py", "import os\nos._exit(0)\n"),
    ("pytest.py", "raise SystemExit(0)\n"),
    ("conftest.py", (
        "import pytest\n"
        "@pytest.hookimpl(hookwrapper=True)\n"
        "def pytest_runtest_makereport(item, call):\n"
        "    outcome = yield\n"
        "    outcome.get_result().outcome = 'passed'\n")),
    ("pytest.ini", "[pytest]\naddopts = -p no:python --co\n"),
    ("sitecustomize.py", "import os\nos._exit(0)\n"),
])
def test_added_runner_hooks_cannot_pass_a_stub(tmp_path, name, text):
    """They are left out of the scoring tree, and reported for information
    only (A4): an added hook file cannot change the score either way."""
    passed, _, reasons, notes, added = _gated(tmp_path, _add(name, text))
    assert not passed
    assert reasons == []
    assert added == [name]
    assert any(repr(name) in n for n in notes)


def test_an_added_hook_file_does_not_fail_an_honest_solution(tmp_path):
    """A4: e.g. a pyproject.toml added for a linter is not a forced fail."""
    passed, out, reasons, notes, added = _gated(
        tmp_path, _add("pyproject.toml", "[tool.ruff]\n", honest=True))
    assert passed, out
    assert reasons == [] and added == ["pyproject.toml"]
    assert notes and all(n.startswith("info:") for n in notes)


def test_an_added_scratch_file_is_listed_but_harmless(tmp_path):
    passed, out, reasons, notes, added = _gated(tmp_path, _add("debug.py", "print(1)\n", honest=True))
    assert passed, out
    assert added == ["debug.py"] and reasons == [] and notes == []


def test_pycache_and_pytest_cache_are_noise(tmp_path):
    def mutate(work):
        _honest(work)
        (work / "__pycache__").mkdir()
        (work / "__pycache__" / "two_fer.cpython-311.pyc").write_bytes(b"x")
        (work / ".pytest_cache").mkdir()
        (work / ".DS_Store").write_bytes(b"x")
    passed, out, reasons, notes, added = _gated(tmp_path, mutate)
    assert passed, out
    assert added == [] and reasons == [] and notes == []


def test_an_exit_before_pytest_finishes_needs_a_report(tmp_path):
    """No tripwire word in sight, so only the missing report catches it."""
    stub = "import os as o\ngetattr(o, '_e' + 'xit')(0)\n"
    passed, out, reasons, _, _ = _gated(tmp_path, _add("two_fer.py", stub))
    assert not passed
    assert reasons == []
    assert "no JUnit report" in out


def test_an_exit_inside_the_solution_needs_a_report(tmp_path):
    stub = ("import os as o\n"
            "def two_fer(name='you'):\n"
            "    getattr(o, '_e' + 'xit')(0)\n")
    passed, out, reasons, _, _ = _gated(tmp_path, _add("two_fer.py", stub))
    assert not passed and "no JUnit report" in out


@pytest.mark.parametrize("stub", [
    # patches the unittest runner in-process
    ("import _pytest.unittest as u\n"
     "u.TestCaseFunction.runtest = lambda self: None\n" + _STUB),
    # the critic's atexit forger: writes a passing report, then exits 0
    ("import atexit, os, sys\n"
     "_p = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--junitxml=')), None)\n"
     "def _w():\n"
     "    if _p:\n"
     "        open(_p, 'w').write('<testsuites><testsuite tests=\"2\" failures=\"0\" "
     "errors=\"0\" skipped=\"0\"></testsuite></testsuites>')\n"
     "    os._exit(0)\n"
     "atexit.register(_w)\n" + _STUB),
    "from os import _exit\n" + _STUB,
    "import sys\nsys.modules.pop('x', None)\n" + _STUB,
    "__import__('os')\n" + _STUB,
])
def test_the_solution_tripwire_forces_a_fail(tmp_path, stub):
    passed, out, reasons, _, _ = _gated(tmp_path, _add("two_fer.py", stub))
    assert not passed
    assert any("two_fer.py" in r and "uses" in r for r in reasons), reasons


def test_the_tripwire_ignores_words_in_comments_and_strings(tmp_path):
    """A6: an AST check, so prose that mentions pytest is no hit."""
    text = '"""Run with pytest; see unittest docs."""\n# conftest, sys.argv, os._exit\n' + _EXAMPLE
    passed, out, reasons, _, _ = _gated(tmp_path, _add("two_fer.py", text))
    assert passed, out
    assert reasons == []


def test_an_edited_test_fails_and_the_agents_tree_is_left_alone(tmp_path):
    edited = "def test_ok():\n    assert True\n"

    def mutate(work):
        _honest(work)
        (work / "two_fer_test.py").write_text(edited)
    seen = {}
    src = _py_exercise(tmp_path)
    work = tmp_path / "work" / "two-fer"
    stubs, tests = AP._prepare_python(src, work)
    m = AP._snapshot(src, work, stubs, tests, AP.LANG_DESCRIPTORS["python"])
    mutate(work)
    passed, out, reasons, _, _ = AP._score_gated(AP.LANG_DESCRIPTORS["python"], work, 60, m)
    seen["after"] = (work / "two_fer_test.py").read_text()
    assert not passed
    assert any("two_fer_test.py" in r and "modified" in r for r in reasons)
    assert seen["after"] == edited
    # A2: the note is the LAST thing in the output, so out[-4000:] keeps it
    assert out.rstrip().splitlines()[-1].startswith("[scorer] scored as a failure")


def test_an_edited_test_cannot_pass_a_stub(tmp_path):
    passed, _, reasons, _, _ = _gated(
        tmp_path, _add("two_fer_test.py", "def test_ok():\n    assert True\n"))
    assert not passed and reasons


def test_a_removed_test_is_a_forced_fail(tmp_path):
    def mutate(work):
        _honest(work)
        (work / "two_fer_test.py").unlink()
    passed, _, reasons, _, _ = _gated(tmp_path, mutate)
    assert not passed and any("two_fer_test.py" in r for r in reasons)


def test_an_edited_doc_is_information_only(tmp_path):
    def mutate(work):
        _honest(work)
        (work / ".docs" / "instructions.md").write_text("notes\n")
    passed, out, reasons, notes, _ = _gated(tmp_path, mutate)
    assert passed, out
    assert reasons == [] and any("instructions.md" in n for n in notes)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX only")
def test_a_fifo_in_place_of_the_stub_does_not_hang(tmp_path):
    box = {}

    def mutate(work):
        (work / "two_fer.py").unlink()
        os.mkfifo(work / "two_fer.py")

    t = threading.Thread(target=lambda: box.setdefault("r", _gated(tmp_path, mutate)), daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "scoring hung on a FIFO"
    passed, _, reasons, _, _ = box["r"]
    assert not passed and any("two_fer.py" in r for r in reasons)


def test_a_symlink_in_place_of_the_stub_is_not_followed(tmp_path):
    def mutate(work):
        (work / "two_fer.py").unlink()
        (work / "two_fer.py").symlink_to("/dev/zero")
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", _gated(tmp_path, mutate)), daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "scoring followed a symlink to /dev/zero"
    passed, _, reasons, _, _ = box["r"]
    assert not passed and any("two_fer.py" in r for r in reasons)


# ── report verification (unit) ──────────────────────────────────────────
def _report(tmp_path, cases, *, failures=0, errors=0, skipped=0, tests=None):
    tcs = "".join(f'<testcase classname="{c}" name="{n}" />' for c, n in cases)
    tests = len(cases) if tests is None else tests
    p = tmp_path / "report.xml"
    p.write_text(f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="{errors}" '
                 f'skipped="{skipped}">{tcs}</testsuite></testsuites>')
    return p


_EXPECTED = frozenset({("m_test.T", "test_a"), ("m_test.T", "test_b")})


def test_junit_ok_accepts_every_expected_name(tmp_path):
    ok, _ = AP._junit_ok(_report(tmp_path, [("m_test.T", "test_a"), ("m_test.T", "test_b[1]")]), _EXPECTED)
    assert ok


@pytest.mark.parametrize("kw,why", [
    ({"cases": [("m_test.T", "test_a")]}, "missing"),
    ({"cases": [("m_test.T", "test_a"), ("m_test.T", "test_b")], "skipped": 1}, "skipped"),
    ({"cases": [("m_test.T", "test_a"), ("m_test.T", "test_b")], "failures": 1}, "failures"),
    ({"cases": [], "tests": 0}, "no tests"),
])
def test_junit_ok_rejects(tmp_path, kw, why):
    ok, reason = AP._junit_ok(_report(tmp_path, **kw), _EXPECTED)
    assert not ok and why in reason


def test_junit_ok_needs_a_regular_file(tmp_path):
    ok, reason = AP._junit_ok(tmp_path / "absent.xml", _EXPECTED)
    assert not ok and "no JUnit report" in reason
    real = _report(tmp_path, [("m_test.T", "test_a"), ("m_test.T", "test_b")])
    link = tmp_path / "link.xml"
    link.symlink_to(real)
    assert not AP._junit_ok(link, _EXPECTED)[0]


def test_python_test_names_follow_pytest_collection():
    src = (b"def test_top(): pass\n"
           b"def helper(): pass\n"
           b"class TestPlain:\n    def test_p(self): pass\n"
           b"class Base:\n    def test_inherited_only(self): pass\n"
           b"class Real(unittest.TestCase):\n"
           b"    def test_r(self):\n        def test_nested(): pass\n")
    names = AP._python_test_names({"sub/x_test.py": src}, ["sub/x_test.py"])
    assert names == frozenset({("sub.x_test", "test_top"), ("sub.x_test.TestPlain", "test_p"),
                               ("sub.x_test.Real", "test_r")})
    assert AP._python_test_names({"x_test.py": b"def ("}, ["x_test.py"]) is None


# ── generic manifest behaviour (fake descriptor) ─────────────────────────
def _fake(tmp_path, monkeypatch, *, files, stubs, tests, links=(), protected=(), run=None):
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    for rel, text in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    seen = {}

    def prepare(s, w):
        AP._copy_exercise(s, w)
        for rel, target in links:
            (w / rel).symlink_to(target)
        return [w / r for r in stubs], [w / r for r in tests]

    def run_tests(work, timeout):
        seen["target"] = work
        seen["files"] = {p.relative_to(work).as_posix(): (p.read_text() if not p.is_symlink() else
                                                          "->" + os.readlink(p))
                         for p in work.rglob("*") if p.is_file() or p.is_symlink()}
        return (run(work) if run else True), "ran"

    desc = {"score_in_copy": True, "practice_dir": tmp_path / "practice", "prepare": prepare,
            "run_tests": run_tests, "syntax_hint": "", "timeout_s": 5,
            "protected_names": frozenset(protected)}
    work = tmp_path / "work" / "ex"
    s, t = prepare(src, work)
    m = AP._snapshot(src, work, s, t, desc)
    return desc, work, m, seen


def test_a_symlinked_test_directory_is_never_written_through(tmp_path, monkeypatch):
    """cubic P1: the agent swaps `sub` for a symlink to a directory outside
    its tree. The harness never writes into an agent-controlled path."""
    desc, work, m, seen = _fake(tmp_path, monkeypatch, files={"ex.py": "stub", "sub/ex_test.py": "tests"},
                                stubs=["ex.py"], tests=["sub/ex_test.py"])
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "ex_test.py").write_text("victim")
    shutil.rmtree(work / "sub")
    (work / "sub").symlink_to(victim)
    passed, out, reasons, _, _ = AP._score_gated(desc, work, 5, m)
    assert (victim / "ex_test.py").read_text() == "victim"
    assert not passed
    assert any("symlink" in r and "sub" in r for r in reasons)
    assert seen["files"]["sub/ex_test.py"] == "tests"


def test_a_js_shaped_tree_scores_pristine_config_and_relinks_node_modules(tmp_path, monkeypatch):
    shared = tmp_path / "shared_nm"
    shared.mkdir()
    desc, work, m, seen = _fake(
        tmp_path, monkeypatch,
        files={"ex.js": "stub", "ex.spec.js": "spec", "package.json": '{"scripts":{"test":"jest ./*"}}'},
        stubs=["ex.js"], tests=["ex.spec.js"], links=[("node_modules", str(shared))],
        protected={"package.json"})
    (work / "package.json").write_text('{"scripts":{"test":"true"}}')
    passed, _, reasons, _, _ = AP._score_gated(desc, work, 5, m)
    assert not passed
    assert any("package.json" in r for r in reasons)
    assert seen["files"]["package.json"] == '{"scripts":{"test":"jest ./*"}}'
    assert seen["files"]["node_modules"] == "->" + str(shared)


def test_npm_install_droppings_are_information_only(tmp_path, monkeypatch):
    """A4: `npm install` writes package-lock.json and may replace the
    node_modules symlink; neither can change the score, so neither forces a
    fail."""
    shared = tmp_path / "shared_nm"
    shared.mkdir()
    desc, work, m, seen = _fake(
        tmp_path, monkeypatch,
        files={"ex.js": "stub", "ex.spec.js": "spec", "package.json": "{}"},
        stubs=["ex.js"], tests=["ex.spec.js"], links=[("node_modules", str(shared))],
        protected={"package.json"})
    (work / "node_modules").unlink()
    (work / "node_modules").mkdir()
    (work / "package-lock.json").write_text("{}")
    passed, _, reasons, notes, added = AP._score_gated(desc, work, 5, m)
    assert passed and reasons == []
    assert "package-lock.json" in added
    assert any("node_modules" in n for n in notes)
    assert seen["files"]["node_modules"] == "->" + str(shared)


# ── per attempt, end to end ──────────────────────────────────────────────
def _real_python(monkeypatch, tmp_path):
    _py_exercise(tmp_path)
    monkeypatch.setitem(AP.LANG_DESCRIPTORS["python"], "practice_dir", tmp_path / "practice")
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")


def test_a_reverted_edit_scores_on_the_next_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    _real_python(monkeypatch, tmp_path)
    prompts = []

    class FakeRpc:
        def __init__(self, *a, cwd=None, session_id="", **kw):
            self.cwd = Path(cwd)
            self.n = int(session_id.rsplit("attempt", 1)[-1])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def notifications(self):
            return []

        def prompt_and_collect(self, message, timeout=900):
            prompts.append(message)
            if self.n == 1:
                (self.cwd / "two_fer_test.py").write_text("def test_ok():\n    assert True\n")
            else:
                (self.cwd / "two_fer_test.py").write_text(_TEST)
                (self.cwd / "two_fer.py").write_text(_EXAMPLE)
            return AP.PromptResult(turn_count=2, agent_ended=True, stop_reason="agent_end",
                                   assistant_text="x", tool_calls=[{"name": "edit"}])

    monkeypatch.setattr(AP, "PiRpc", FakeRpc)
    record = AP._run_exercise("python", "two-fer", "fake/model", agent="pi", verbose=False,
                              retry=True, max_attempts=2)
    assert record["status"] == "pass_2"
    assert record["tests_tampered"] is True
    assert "[scorer] scored as a failure" in prompts[1]


def test_the_plain_protocol_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("POLYGLOT_RESTORE_TESTS", raising=False)
    _real_python(monkeypatch, tmp_path)
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, "1 failed", "")

    monkeypatch.setattr(AP.subprocess, "run", fake_run)

    class FakeRpc:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def notifications(self):
            return []

        def prompt_and_collect(self, message, timeout=900):
            return AP.PromptResult(turn_count=2, agent_ended=True, stop_reason="agent_end",
                                   assistant_text="x", tool_calls=[{"name": "edit"}])

    monkeypatch.setattr(AP, "PiRpc", FakeRpc)
    record = AP._run_exercise("python", "two-fer", "fake/model", agent="pi", verbose=False,
                              retry=False)
    assert cmds == [["python3", "-m", "pytest", "-x", "-q"]]
    assert record["status"] == "fail"
    for key in ("tests_tampered", "tamper_reasons", "added_files"):
        assert key not in record


def test_preflight_runs_what_scoring_runs_and_raises_before_any_agent(monkeypatch, tmp_path):
    """A3: probe `sys.executable -I -m pytest --version`, not find_spec in
    the harness process, and fail loudly before an agent is started."""
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    _real_python(monkeypatch, tmp_path)
    monkeypatch.setattr(AP, "_PREFLIGHT_OK", set())
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "No module named pytest")

    monkeypatch.setattr(AP.subprocess, "run", fake_run)

    def boom(*a, **kw):
        raise AssertionError("an agent was started")

    monkeypatch.setattr(AP, "PiRpc", boom)
    with pytest.raises(RuntimeError, match="pytest"):
        AP._run_exercise("python", "two-fer", "fake/model", agent="pi", verbose=False, retry=False)
    assert cmds == [[sys.executable, "-I", "-m", "pytest", "--version"]]


# ── the JS runner (real jest, when the shared install exists) ───────────
_JEST = AP._JS_SHARED_NODE_MODULES / "jest" / "bin" / "jest.js"
_needs_jest = pytest.mark.skipif(not _JEST.is_file() or shutil.which("node") is None,
                                 reason="shared JS install or node not present")


def _js_tree(tmp_path, solution):
    work = tmp_path / "work" / "ex"
    work.mkdir(parents=True)
    (work / "ex.js").write_text(solution)
    (work / "ex.spec.js").write_text(
        "const { f } = require('./ex');\n"
        "test('one', () => { expect(f()).toBe(1); });\n"
        "test('two', () => { expect(f() + 1).toBe(2); });\n")
    (work / "package.json").write_text('{"scripts": {"test": "jest ./*"}}')
    (work / "node_modules").symlink_to(AP._JS_SHARED_NODE_MODULES)
    return work


@_needs_jest
@pytest.mark.parametrize("solution,ok", [
    ("module.exports.f = () => 1;\n", True),
    ("module.exports.f = () => 0;\n", False),
    ("process.exit(0);\nmodule.exports.f = () => 0;\n", False),
])
def test_the_js_runner_needs_a_clean_report(tmp_path, solution, ok):
    work = _js_tree(tmp_path, solution)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    passed, out = AP._run_javascript_gated(work, 120, scratch, None)
    assert passed is ok, out


@pytest.mark.parametrize("report,why", [
    ({"success": True, "numTotalTests": 2, "numFailedTests": 0, "numPendingTests": 1,
      "numTodoTests": 0}, "pending"),
    ({"success": True, "numTotalTests": 0, "numFailedTests": 0, "numPendingTests": 0,
      "numTodoTests": 0}, "no tests"),
    ({"success": False, "numTotalTests": 2, "numFailedTests": 1, "numPendingTests": 0,
      "numTodoTests": 0}, "fail"),
])
def test_jest_report_ok_rejects(tmp_path, report, why):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(report))
    ok, reason = AP._jest_report_ok(p)
    assert not ok and why in reason


def test_jest_report_ok_accepts_a_clean_report(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"success": True, "numTotalTests": 2, "numFailedTests": 0,
                             "numPendingTests": 0, "numTodoTests": 0}))
    assert AP._jest_report_ok(p)[0]
    assert not AP._jest_report_ok(tmp_path / "absent.json")[0]


# ── trajectory snapshots (A5) ────────────────────────────────────────────
class _R:
    agent_ended = True
    turn_count = 1
    compaction_events = 0
    assistant_text = "a"
    tool_calls = []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX only")
def test_dump_trajectory_skips_special_files_and_does_not_follow_symlinks(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    (work / "solution.py").write_text("code")
    os.mkfifo(work / "pipe")
    (work / "zero.py").symlink_to("/dev/zero")
    (work / "big.bin").write_bytes(b"\0" * (AP._SAFE_READ_LIMIT + 1))
    t = threading.Thread(target=AP._dump_trajectory, args=(log_dir, "1", _R(), work), daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "snapshot hung"
    snap = log_dir / "workdir_1"
    assert (snap / "solution.py").read_text() == "code"
    assert not (snap / "pipe").exists() and not (snap / "pipe").is_symlink()
    assert (snap / "zero.py").is_symlink()
    assert not (snap / "big.bin").exists()
