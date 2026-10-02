"""End-to-end live-eval pipeline test: a REAL private scratch repo, REAL
candidate file writes, a REAL subprocess invocation of aider_polyglot.py, a
REAL `pytest -x -q` scoring run, REAL result-JSON parsing -- with `pi`
routed through fake_pi.py so zero model calls happen. This is the centerpiece
test of the whole live-execution rewrite: it exercises the exact mechanism
(rpc_client.REPO_ROOT resolving to the SCRATCH tree because
aider_polyglot.py is invoked as a subprocess whose own __file__ lives there)
that makes a candidate's text actually reach a live agent.
"""
import base64
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import benchmarks.self_improve.live_eval as live_eval
from benchmarks.self_improve.exercises import ExerciseSpec
from benchmarks.self_improve.live_cache import LiveResultCache, run_config_hash
from benchmarks.self_improve.live_eval import PolyglotLiveRunner
from benchmarks.self_improve.scratch_worktree import scratch_worktree

# This module unconditionally drives real `git init`/`git fetch` and
# real subprocesses -- unlike the sibling fake_pi tests, it never skips on a
# minimal runner without git, which would otherwise hard-error every test
# here (masking the rest of the self_improve suite's own results) instead of
# skipping gracefully, the way the polyglot tests skip when
# node_modules/.bin/pi is absent.
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required for this e2e suite")

REAL_REPO_ROOT = Path(__file__).resolve().parents[3]  # little-coder-self-improve/
FAKE_PI = REAL_REPO_ROOT / "benchmarks" / "fake_pi.py"

_WORDY_TEST = '''
import wordy

def test_wordy():
    assert wordy.solve() == "42"
'''
_WORDY_STUB = "def solve():\n    pass\n"
_WORDY_SOLUTION = 'def solve():\n    return "42"\n'


@pytest.fixture
def source_repo(tmp_path):
    """A throwaway git repo with committed COPIES (not stubs) of the real
    aider_polyglot.py/rpc_client.py, real-shaped AGENTS.md/skill file, and a
    components.yaml mapping -- copies, not symlinks, so the worktree
    contains real files at real paths and rpc_client.REPO_ROOT resolves to
    it, exactly like a real scratch checkout."""
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    (repo / "AGENTS.md").write_text("# little-coder\n\nOriginal instructions.\n")
    skills_dir = repo / "skills" / "tools"
    skills_dir.mkdir(parents=True)
    (skills_dir / "bash.md").write_text("---\nname: bash\ntype: tool-guidance\n---\nOriginal bash guidance.\n")

    (repo / "config").mkdir()
    (repo / "config" / "components.yaml").write_text(yaml.dump({
        "agents_md": "AGENTS.md",
        "skills_tools_bash": "skills/tools/bash.md",
    }))

    (repo / "benchmarks").mkdir()
    shutil.copy(REAL_REPO_ROOT / "benchmarks" / "aider_polyglot.py", repo / "benchmarks" / "aider_polyglot.py")
    shutil.copy(REAL_REPO_ROOT / "benchmarks" / "rpc_client.py", repo / "benchmarks" / "rpc_client.py")
    # Also hashed into run_config's harness_hash (the graded score has
    # depended on these two since the compaction penalty / token_cost
    # estimator landed) -- present here so a test can confirm that.
    (repo / "benchmarks" / "self_improve" / "ingest").mkdir(parents=True)
    (repo / "benchmarks" / "self_improve" / "ingest" / "aider_polyglot_ingest.py").write_text("SCORING_V1 = 1\n")
    (repo / "benchmarks" / "self_improve" / "components.py").write_text("SCORING_V1 = 1\n")

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


@pytest.fixture
def fake_practice(tmp_path):
    """One real exercise: a stub that fails a genuine pytest test until
    solved, matching aider_polyglot.py's own _prepare_python/_run_python
    layout exactly (glob *.py minus *_test.py for stubs, *_test.py for tests)."""
    practice_root = tmp_path / "polyglot-benchmark"
    ex_dir = practice_root / "python" / "exercises" / "practice" / "wordy"
    ex_dir.mkdir(parents=True)
    (ex_dir / "wordy.py").write_text(_WORDY_STUB)
    (ex_dir / "wordy_test.py").write_text(_WORDY_TEST)
    return practice_root


@pytest.fixture
def runner_factory(source_repo, fake_practice, tmp_path, monkeypatch):
    """Real env for the child subprocess: fast attempt timeout (this suite
    must not be able to hang for 900s), pi routed through fake_pi.py."""
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", str(FAKE_PI))

    def _make(cache=None, budget=None, pi_bin=FAKE_PI, per_exercise_timeout_s=60, on_result=None):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=pi_bin) as wt:
            yield PolyglotLiveRunner(
                worktree=wt,
                components_yaml=source_repo / "config" / "components.yaml",
                model="fake/model",
                max_attempts=2,
                benchmark_root=fake_practice,
                cache=cache,
                per_exercise_timeout_s=per_exercise_timeout_s,
                budget=budget,
                on_result=on_result,
            )

    return _make


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_pipeline_scores_a_pass_as_one(runner_factory, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert results[0].status == "pass_1"
    assert results[0].score == 1.0
    assert results[0].success is True


def test_reasoning_excerpt_is_reconstructed_from_thinking_delta_chunks(runner_factory, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "emit_multi_thinking_delta")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    result = results[0]
    assert result.status == "pass_1"
    # The three thinking_delta chunks, concatenated in order -- not the
    # text_delta content, which is a separate channel entirely.
    assert result.reasoning_excerpt == (
        "Let me read the stub and the test file first. Now I understand the task."
    )
    assert "Implemented and tests pass" not in result.reasoning_excerpt
    assert "Implemented and tests pass" in result.transcript_excerpt
    assert "Let me read the stub" not in result.transcript_excerpt


def test_summarized_transcript_surfaces_a_recoverable_tool_error(runner_factory, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "emit_tool_error_then_solve")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    result = results[0]
    # The attempt still ultimately passes -- the point is that reflection
    # sees the recoverable error anyway, not just the happy ending.
    assert result.status == "pass_1"
    assert "[ERROR]" in result.summarized_transcript
    assert "bash" in result.summarized_transcript
    assert "Permission denied" in result.summarized_transcript


def test_compaction_total_is_captured_and_lowers_the_score(runner_factory, monkeypatch):
    """Compaction_events was discarded
    entirely between aider_polyglot.py and the live GEPA loop -- a
    candidate whose injected text forces context compaction (real prompt
    bloat) never affected its score or reflection feedback at all."""
    monkeypatch.setenv("FAKE_PI_MODE", "emit_compactions_then_solve")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    result = results[0]
    assert result.status == "pass_1"
    assert result.compaction_total == 2
    assert result.score == 1.0 - 0.05 * 2  # pass_1's 1.00 minus the 2-compaction penalty


def test_pipeline_scores_a_genuine_failure_as_zero(runner_factory, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "clean")  # writes nothing relevant
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert results[0].status == "fail"
    assert results[0].score == 0.0
    assert results[0].success is False
    assert "assert" in results[0].test_output_tail.lower() or "fail" in results[0].test_output_tail.lower()


def test_pipeline_scores_a_second_attempt_pass_as_partial_credit(runner_factory, tmp_path, monkeypatch):
    """noop_then_solve writes nothing on attempt 1, the real solution on
    attempt 2 -- attempts are separate PROCESSES (a fresh PiRpc/fake_pi per
    attempt), so this also proves the graded formula flows end to end."""
    monkeypatch.setenv("FAKE_PI_MODE", "noop_then_solve")
    monkeypatch.setenv("FAKE_PI_STATE_FILE", str(tmp_path / "attempt_state.txt"))
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert results[0].status == "pass_2"
    assert results[0].score == 0.7



# fake_pi.py's TURN_USAGE (input 100 + cacheRead 10 + cacheWrite 0, output
# 20), one turn per attempt.
_ONE_ATTEMPT_USAGE = {"input_tokens": 110, "cache_read_tokens": 10, "output_tokens": 20}


def test_token_usage_reaches_the_live_run_result(runner_factory, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert results[0].usage == _ONE_ATTEMPT_USAGE


def test_token_usage_is_summed_across_attempts(runner_factory, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "noop_then_solve")
    monkeypatch.setenv("FAKE_PI_STATE_FILE", str(tmp_path / "attempt_state.txt"))
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert results[0].status == "pass_2"
    assert results[0].usage == {k: 2 * v for k, v in _ONE_ATTEMPT_USAGE.items()}


def test_self_reported_lesson_is_captured_end_to_end(runner_factory, tmp_path, monkeypatch):
    """Real gap this closes: the retry loop fed the next attempt raw test
    output but never solicited or captured an explicit self-reflection --
    Reflexion's actual validated technique (verbal self-reflection between
    retry attempts) was unavailable. Attempt 1 fails, attempt 2 states a
    LESSON: line then solves -- proves it's captured through the real
    subprocess/trajectory pipeline, not just the in-process unit tests."""
    monkeypatch.setenv("FAKE_PI_MODE", "fail_then_report_lesson_then_solve")
    monkeypatch.setenv("FAKE_PI_STATE_FILE", str(tmp_path / "attempt_state.txt"))
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    result = results[0]
    assert result.status == "pass_2"
    assert result.self_reported_lessons == ["needed clearer guidance"]


def test_exercise_subprocess_is_asked_to_request_lessons(runner_factory, monkeypatch):
    """aider_polyglot.py only asks retries for a LESSON: line under
    POLYGLOT_REQUEST_LESSONS=1, so live_eval must set it on the child even
    when the orchestrator's own environment says otherwise."""
    monkeypatch.setenv("POLYGLOT_REQUEST_LESSONS", "0")
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    real_popen = subprocess.Popen
    child_envs = []

    def spy(cmd, *a, **kw):
        if any(str(part).endswith("aider_polyglot.py") for part in cmd):
            child_envs.append(kw.get("env"))
        return real_popen(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", spy)
    for runner in runner_factory():
        runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert len(child_envs) == 1
    assert child_envs[0]["POLYGLOT_REQUEST_LESSONS"] == "1"


def test_exercise_subprocess_is_asked_for_strict_scoring(runner_factory, monkeypatch):
    """Both guards are off by default in aider_polyglot.py; a crash on a
    later attempt must not be cached as a fail, and an edited test file must
    not score as a pass, so live_eval turns them on for the child."""
    monkeypatch.setenv("POLYGLOT_CRASH_IS_ERROR", "0")
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "0")
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    real_popen = subprocess.Popen
    child_envs = []

    def spy(cmd, *a, **kw):
        if any(str(part).endswith("aider_polyglot.py") for part in cmd):
            child_envs.append(kw.get("env"))
        return real_popen(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", spy)
    for runner in runner_factory():
        runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert len(child_envs) == 1
    assert child_envs[0]["POLYGLOT_CRASH_IS_ERROR"] == "1"
    assert child_envs[0]["POLYGLOT_RESTORE_TESTS"] == "1"


def test_materialize_writes_candidate_text_preserving_frontmatter(runner_factory):
    for runner in runner_factory():
        runner.materialize({"skills_tools_bash": "Brand new guidance body.\n"})
        written = (runner.worktree.path / "skills" / "tools" / "bash.md").read_text()
    assert "name: bash" in written  # frontmatter preserved byte-for-byte
    assert "Brand new guidance body." in written
    assert "Original bash guidance." not in written


def test_candidate_text_actually_reaches_the_agent(runner_factory, tmp_path, monkeypatch):
    """The regression test for the mechanism itself: fake_pi echoes the
    REAL --system-prompt file it received back to a marker file, proving
    aider_polyglot.py (run as a subprocess FROM the scratch worktree) picked
    up rpc_client.REPO_ROOT as the worktree, not the source repo."""
    echo_file = tmp_path / "echo.txt"
    monkeypatch.setenv("FAKE_PI_MODE", "read_system_prompt_echo")
    monkeypatch.setenv("FAKE_PI_ECHO_FILE", str(echo_file))
    for runner in runner_factory():
        runner.run_batch({"agents_md": "DISTINCTIVE CANDIDATE TEXT 99887766\n"}, [ExerciseSpec("wordy")])
    assert "DISTINCTIVE CANDIDATE TEXT 99887766" in echo_file.read_text()


def test_two_candidates_that_differ_only_in_text_get_different_scores(runner_factory, monkeypatch):
    """Two candidates differing ONLY in instruction text must produce
    genuinely different scores -- fake_pi decides whether to solve based on
    what it reads from the live system prompt. A scoring path that never
    reads the candidate's text scores every candidate identically, and
    GEPA's acceptance criterion can then never fire."""
    monkeypatch.setenv("FAKE_PI_MODE", "solve_if_prompt_contains")
    monkeypatch.setenv("FAKE_PI_MAGIC_TOKEN", "OPEN-SESAME-42")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))

    for runner in runner_factory():
        good_candidate = {"agents_md": "Instructions containing OPEN-SESAME-42 the magic phrase.\n"}
        results_good = runner.run_batch(good_candidate, [ExerciseSpec("wordy")])

    for runner in runner_factory():
        bad_candidate = {"agents_md": "Unrelated instructions with no special phrase.\n"}
        results_bad = runner.run_batch(bad_candidate, [ExerciseSpec("wordy")])

    assert results_good[0].score != results_bad[0].score
    assert results_good[0].success is True
    assert results_bad[0].success is False


def test_cache_hit_skips_the_subprocess_entirely(runner_factory, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    cache = LiveResultCache(tmp_path / "cache")
    candidate = {"skills_tools_bash": "Some guidance.\n"}

    for runner in runner_factory(cache=cache):
        first = runner.run_batch(candidate, [ExerciseSpec("wordy")])

    for runner in runner_factory(cache=cache):
        # Prove the cache hit means "no subprocess at all", not just "a
        # subprocess ran and happened to agree" -- the strongest possible
        # assertion, stronger than comparing results alone.
        def _must_not_run(*a, **kw):
            raise AssertionError("cache hit should have skipped this exercise entirely")
        monkeypatch.setattr(runner, "_run_one_uncached", _must_not_run)
        second = runner.run_batch(candidate, [ExerciseSpec("wordy")])

    assert first[0].from_cache is False
    assert second[0].from_cache is True
    assert second[0].status == first[0].status
    assert second[0].score == first[0].score
    # usage is an output, memoized with the rest of the result.
    assert second[0].usage == first[0].usage == _ONE_ATTEMPT_USAGE


def test_materialize_raises_when_candidate_has_a_component_not_in_components_yaml(runner_factory):
    """Write_components_back() only logs a
    warning and silently skips a pred_name absent from the pinned
    components.yaml -- every candidate variant of an unmapped component
    would then materialize to IDENTICAL worktree content and score
    identically, the exact "score independent of candidate text" failure
    class this whole live-execution design exists to eliminate."""
    for runner in runner_factory():
        with pytest.raises(ValueError, match="not present in"):
            runner.materialize({"agents_md": "text", "totally_unmapped_pred_name": "other text"})


def test_run_config_includes_the_pi_binary(runner_factory):
    for runner in runner_factory():
        assert runner.run_config["pi_bin"] == str(FAKE_PI)


def test_run_config_includes_the_scoring_interpreter(runner_factory):
    """Gated scoring runs pytest under python_executable, so a different
    interpreter or pytest must not reuse cached scores."""
    for runner in runner_factory():
        config = runner.run_config
        assert config["python_executable"] == runner.python_executable
        assert config["python_version"] == sys.version.split()[0]
        assert config["pytest_version"] == pytest.__version__


def test_run_config_pi_bin_changes_with_a_different_binary(runner_factory, tmp_path):
    """The pi binary IS the agent under
    test -- omitting it from run_config meant a cache entry produced via
    LITTLE_CODER_PI_BIN_OVERRIDE (e.g. a fake_pi.py smoke run) could be read
    back as a genuine result by a later real run with the same model/etc."""
    other_pi = tmp_path / "other_fake_pi.py"
    other_pi.write_text(FAKE_PI.read_text())
    for runner in runner_factory():
        config_a = runner.run_config
    for runner in runner_factory(pi_bin=other_pi):
        config_b = runner.run_config
    assert config_a["pi_bin"] != config_b["pi_bin"]


def test_run_config_harness_hash_changes_when_the_worktree_executed_files_change(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """aider_polyglot.py/rpc_client.py run as a SUBPROCESS inside the
    scratch worktree, so the worktree's own (pinned-commit) copy is exactly
    what executes -- committing a change to them in source_repo, which the
    NEXT scratch_worktree checkout picks up, must change the cache key."""
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", str(FAKE_PI))

    def _hash():
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=FAKE_PI) as wt:
            runner = PolyglotLiveRunner(
                worktree=wt, components_yaml=source_repo / "config" / "components.yaml",
                model="fake/model", max_attempts=1, benchmark_root=fake_practice,
            )
            return runner.run_config["harness_hash"]

    hash_before = _hash()

    (source_repo / "benchmarks" / "rpc_client.py").write_text("SUBPROCESS_V2 = 2\n")
    subprocess.run(["git", "add", "-A"], cwd=source_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "retune rpc_client"], cwd=source_repo, check=True)

    hash_after = _hash()
    assert hash_before != hash_after


def test_run_config_harness_hash_changes_when_the_parent_processs_own_scoring_module_changes(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """Pass_n_score() (aider_polyglot_ingest.py,
    the compaction penalty formula) and write_components_back() (components.py,
    the token_cost estimator) run in the PARENT (orchestrator) process --
    imported once at the top of live_eval.py -- never inside the scratch
    worktree at all. An earlier version of run_config hashed the WORKTREE's
    copy of these two files, which meant an uncommitted retune in the real
    checkout the orchestrator actually imports from wouldn't change the
    cache key, even though the orchestrator's own process had already
    picked up the edit -- LiveResultCache would keep serving scores computed
    under the old formula. Hashing the module's own `__file__` (simulated
    here via monkeypatch, since editing this repo's real components.py
    mid-test would be its own kind of chaos) is what actually tracks that."""
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", str(FAKE_PI))

    fake_components_file = tmp_path / "fake_components.py"
    fake_components_file.write_text("SCORING_V1 = 1\n")
    monkeypatch.setattr(live_eval._components_module, "__file__", str(fake_components_file))

    def _hash():
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=FAKE_PI) as wt:
            runner = PolyglotLiveRunner(
                worktree=wt, components_yaml=source_repo / "config" / "components.yaml",
                model="fake/model", max_attempts=1, benchmark_root=fake_practice,
            )
            return runner.run_config["harness_hash"]

    hash_before = _hash()

    fake_components_file.write_text("SCORING_V2 = 2\n")  # no git commit -- exactly the uncommitted-retune case

    hash_after = _hash()
    assert hash_before != hash_after


def test_run_config_harness_hash_is_unaffected_by_editing_the_worktree_copy_of_the_scoring_files(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """The flip side of the fix above: aider_polyglot_ingest.py/components.py
    never execute from inside the scratch worktree, so committing an edit to
    the WORKTREE's copy of them (without touching what the parent process
    itself imports) must NOT spuriously change the cache key -- that would
    just cause needless re-runs of already-cached work for a file whose
    content never actually influenced the score."""
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", str(FAKE_PI))

    def _hash():
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=FAKE_PI) as wt:
            runner = PolyglotLiveRunner(
                worktree=wt, components_yaml=source_repo / "config" / "components.yaml",
                model="fake/model", max_attempts=1, benchmark_root=fake_practice,
            )
            return runner.run_config["harness_hash"]

    hash_before = _hash()

    (source_repo / "benchmarks" / "self_improve" / "components.py").write_text("SCORING_V2 = 2\n")
    subprocess.run(["git", "add", "-A"], cwd=source_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "retune scoring (worktree copy only)"], cwd=source_repo, check=True)

    hash_after = _hash()
    assert hash_before == hash_after


def test_budget_clamp_raises_instead_of_faking_a_timeout_score(runner_factory, monkeypatch, tmp_path):
    """Clamping the subprocess timeout to
    the remaining wall-clock budget, then treating the resulting kill as an
    ordinary subprocess timeout, fabricated a real-looking harness_error/0.0
    score for a run that was killed for budget reasons -- violating
    LiveBudget's own documented invariant that a budget refusal must never
    look like a genuine, scoreable outcome."""
    import time as time_module

    from benchmarks.self_improve.live_budget import LiveBudget, LiveEvalBudgetExceeded

    monkeypatch.setenv("FAKE_PI_MODE", "sleep_forever")
    monkeypatch.setenv("FAKE_PI_SLEEP_S", "30")
    # per_exercise_timeout_s is generously large; the BUDGET deadline is what
    # actually clamps the effective timeout down to ~2s, well short of the
    # fake_pi process's own 30s sleep.
    budget = LiveBudget(hard_deadline_monotonic=time_module.monotonic() + 2, max_live_runs=1000)
    for runner in runner_factory(budget=budget, per_exercise_timeout_s=120):
        with pytest.raises(LiveEvalBudgetExceeded):
            runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])


def test_on_result_still_fires_for_exercises_completed_before_a_later_budget_exceeded(
    runner_factory, fake_practice, monkeypatch,
):
    """The audit callback used to be invoked
    only after run_batch() returned the WHOLE batch, so a later exercise
    hitting LiveBudget's backstop lost the audit record for every exercise
    that already genuinely ran earlier in the same batch."""
    import time as time_module

    from benchmarks.self_improve.live_budget import LiveBudget, LiveEvalBudgetExceeded

    ex_dir = fake_practice / "python" / "exercises" / "practice" / "acronym"
    ex_dir.mkdir(parents=True)
    (ex_dir / "acronym.py").write_text(_WORDY_STUB)
    (ex_dir / "acronym_test.py").write_text(_WORDY_TEST.replace("wordy", "acronym"))

    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({
        "wordy.py": _b64(_WORDY_SOLUTION), "acronym.py": _b64(_WORDY_SOLUTION),
    }))

    seen = []
    budget = LiveBudget(hard_deadline_monotonic=time_module.monotonic() + 3600, max_live_runs=1)
    for runner in runner_factory(budget=budget, on_result=seen.append):
        with pytest.raises(LiveEvalBudgetExceeded):
            runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy"), ExerciseSpec("acronym")])

    assert len(seen) == 1
    assert seen[0].task_id == "python/wordy"


def test_per_exercise_timeout_default_tracks_attempt_timeout_s_env_var(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """This default used to hardcode a bare
    900 literal for aider_polyglot's own per-attempt budget, independent of
    the ATTEMPT_TIMEOUT_S env var that module actually reads. When
    aider_polyglot.py's own default was tripled to 2700s for a local
    reasoning model, this harness-level ceiling silently stayed at the old,
    now-too-short value -- the OUTER subprocess timeout would fire and kill
    an exercise via SIGTERM before even one INNER attempt's own (longer)
    budget had a chance to time out gracefully."""
    import benchmarks.aider_polyglot as aider_polyglot
    import benchmarks.self_improve.live_eval as live_eval
    from benchmarks.self_improve.live_eval import PolyglotLiveRunner, _attempt_timeout_s
    from benchmarks.self_improve.scratch_worktree import scratch_worktree

    # The first
    # fix here kept a duplicated literal and asserted against
    # aider_polyglot._positive_int_env("ATTEMPT_TIMEOUT_S", 2700) -- a
    # tautology, since with the env var unset that just returns the SAME
    # 2700 passed in as ITS OWN default argument. The second fix imported
    # aider_polyglot._ATTEMPT_TIMEOUT_S_DEFAULT directly -- structurally
    # sound, but importing the WHOLE module also runs its
    # CODEX_TIMEOUT_S = _positive_int_env(...) validation as a side
    # effect, which could crash this process over an unrelated, unused env
    # var. live_eval.py now regex-extracts the constant from
    # aider_polyglot.py's own file TEXT instead (see
    # _attempt_timeout_default_from_source()'s own docstring) -- this test
    # imports aider_polyglot directly ONLY here, as ground truth for
    # comparison, which live_eval.py itself deliberately avoids.
    real_default = aider_polyglot._ATTEMPT_TIMEOUT_S_DEFAULT
    assert live_eval._ATTEMPT_TIMEOUT_S_DEFAULT == real_default
    monkeypatch.delenv("ATTEMPT_TIMEOUT_S", raising=False)
    assert _attempt_timeout_s() == real_default

    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=FAKE_PI) as wt:
        runner = PolyglotLiveRunner(
            worktree=wt, components_yaml=source_repo / "config" / "components.yaml",
            model="fake/model", max_attempts=2, benchmark_root=fake_practice,
        )
        assert runner.per_exercise_timeout_s == 2 * (real_default + 90) + 180

    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    assert _attempt_timeout_s() == 30
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=FAKE_PI) as wt:
        runner = PolyglotLiveRunner(
            worktree=wt, components_yaml=source_repo / "config" / "components.yaml",
            model="fake/model", max_attempts=2, benchmark_root=fake_practice,
        )
        assert runner.per_exercise_timeout_s == 2 * (30 + 90) + 180


def test_per_exercise_timeout_default_uses_the_worktrees_own_pinned_copy(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """The outer per-exercise timeout
    default used to be read from the SOURCE checkout (this process' own
    aider_polyglot.py, at import time) -- but the subprocess that actually
    RUNS an exercise executes the scratch worktree's PINNED base_commit
    copy, which can diverge from the source checkout (an uncommitted local
    edit to aider_polyglot.py itself, mid-development on the harness while
    a live run is in progress). Using the wrong copy's default here could
    compute an outer timeout shorter than the pinned copy's actual
    per-attempt budget, killing the child before its own (longer) timeout
    fires gracefully. Simulates that divergence directly: edit the
    WORKTREE's pinned copy after checkout (as if it were pinned to an
    older/different commit) and confirm the runner picks up THAT value,
    not the source checkout's."""
    import benchmarks.aider_polyglot as aider_polyglot
    from benchmarks.self_improve.live_eval import PolyglotLiveRunner
    from benchmarks.self_improve.scratch_worktree import scratch_worktree

    # An earlier version of this test
    # hardcoded the bare 2700 literal in both this guard and the replace()
    # call below -- the exact duplication the whole change exists to
    # eliminate. Ground truth comes from aider_polyglot.py itself, same as
    # the sibling env-var test above.
    real_default = aider_polyglot._ATTEMPT_TIMEOUT_S_DEFAULT
    pinned_default = real_default - 2640  # deliberately different from real_default

    monkeypatch.delenv("ATTEMPT_TIMEOUT_S", raising=False)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=FAKE_PI) as wt:
        pinned_file = wt.path / "benchmarks" / "aider_polyglot.py"
        text = pinned_file.read_text()
        assert f"_ATTEMPT_TIMEOUT_S_DEFAULT = {real_default}" in text
        pinned_file.write_text(
            text.replace(
                f"_ATTEMPT_TIMEOUT_S_DEFAULT = {real_default}",
                f"_ATTEMPT_TIMEOUT_S_DEFAULT = {pinned_default}",
            )
        )

        runner = PolyglotLiveRunner(
            worktree=wt, components_yaml=source_repo / "config" / "components.yaml",
            model="fake/model", max_attempts=2, benchmark_root=fake_practice,
        )
        # Reflects the WORKTREE's pinned value, not the source checkout's.
        assert runner.per_exercise_timeout_s == 2 * (pinned_default + 90) + 180


@pytest.mark.parametrize("bad_value", ["not-a-number", "0", "-5"])
def test_attempt_timeout_s_raises_on_malformed_or_non_positive_value(bad_value, monkeypatch):
    """A malformed/non-positive
    ATTEMPT_TIMEOUT_S used to be silently swallowed and replaced with the
    default here, computing a per_exercise_timeout_s estimate as if the
    run would proceed normally -- but aider_polyglot.py's own
    _positive_int_env() raises SystemExit on the exact same value, so the
    actual subprocess crashes at import time instead. Must fail the same
    way, not silently substitute a misleading default."""
    from benchmarks.self_improve.live_eval import _attempt_timeout_s

    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", bad_value)
    with pytest.raises(SystemExit):
        _attempt_timeout_s()


def test_attempt_timeout_default_from_source_warns_when_regex_does_not_match(tmp_path, caplog):
    """A benign reformat of aider_polyglot.py
    (a type annotation, `= 2_700` with an underscore, a trailing comment,
    changed spacing) makes the regex stop matching -- must not silently
    degrade to the hardcoded fallback with no signal, or a real default
    change hidden behind a reformat goes completely undetected."""
    from benchmarks.self_improve.live_eval import (
        _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK,
        _attempt_timeout_default_from_source,
    )

    reformatted = tmp_path / "aider_polyglot.py"
    reformatted.write_text("_ATTEMPT_TIMEOUT_S_DEFAULT: int = 3000  # reformatted\n")

    with caplog.at_level("WARNING"):
        value = _attempt_timeout_default_from_source(reformatted)

    assert value == _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK
    assert "not found" in caplog.text


def test_attempt_timeout_default_from_source_warns_when_file_missing(tmp_path, caplog):
    from benchmarks.self_improve.live_eval import (
        _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK,
        _attempt_timeout_default_from_source,
    )

    with caplog.at_level("WARNING"):
        value = _attempt_timeout_default_from_source(tmp_path / "does_not_exist.py")

    assert value == _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK
    assert "could not read" in caplog.text


def test_attempt_timeout_default_from_source_warns_on_non_utf8_file(tmp_path, caplog):
    """UnicodeDecodeError is NOT an OSError
    subclass (it's a ValueError) -- a bare `except OSError` let it
    propagate straight out of this function, aborting live_eval's own
    module import (this function runs once at module level) on a
    non-UTF-8 locale/file, before even --estimate-only could run."""
    from benchmarks.self_improve.live_eval import (
        _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK,
        _attempt_timeout_default_from_source,
    )

    non_utf8 = tmp_path / "aider_polyglot.py"
    non_utf8.write_bytes(b"_ATTEMPT_TIMEOUT_S_DEFAULT = 2700\n\xff\xfe not utf-8 \x80")

    with caplog.at_level("WARNING"):
        value = _attempt_timeout_default_from_source(non_utf8)

    assert value == _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK
    assert "could not read" in caplog.text


def test_cache_is_keyed_on_the_sanitized_candidate(runner_factory, tmp_path, monkeypatch):
    """materialize() writes _sanitize_candidate(candidate), so two
    candidates differing only by a re-emitted frontmatter block run
    byte-identical files -- keying on the raw text paid twice for one outcome."""
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    cache = LiveResultCache(tmp_path / "cache")
    with_frontmatter = {"skills_tools_bash": "---\nname: bash\n---\nSame guidance.\n"}
    plain = {"skills_tools_bash": "Same guidance.\n"}

    for runner in runner_factory(cache=cache):
        first = runner.run_batch(with_frontmatter, [ExerciseSpec("wordy")])

    for runner in runner_factory(cache=cache):
        def _must_not_run(*a, **kw):
            raise AssertionError("the sanitized-equal candidate should have been a cache hit")
        monkeypatch.setattr(runner, "_run_one_uncached", _must_not_run)
        second = runner.run_batch(plain, [ExerciseSpec("wordy")])

    assert first[0].from_cache is False
    assert second[0].from_cache is True


def test_run_batch_sample_index_selects_a_distinct_cache_entry(runner_factory, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    cache = LiveResultCache(tmp_path / "cache")
    candidate = {"skills_tools_bash": "Some guidance.\n"}

    for runner in runner_factory(cache=cache):
        runner.run_batch(candidate, [ExerciseSpec("wordy")])
        calls = []
        real = runner._run_one_uncached
        monkeypatch.setattr(runner, "_run_one_uncached", lambda spec: calls.append(spec) or real(spec))
        again_default = runner.run_batch(candidate, [ExerciseSpec("wordy")])
        other_sample = runner.run_batch(candidate, [ExerciseSpec("wordy")], sample_index=1)

    assert again_default[0].from_cache is True
    assert other_sample[0].from_cache is False
    assert len(calls) == 1


def _canned(spec, status, score=0.0, error=None):
    return live_eval.LiveRunResult(
        task_id=spec.task_id, exercise=spec.exercise, language=spec.language,
        status=status, score=score, success=score > 0, error=error,
    )


#: Statuses run_batch() retries in place. harness_error then raises;
#: error/empty_response are returned scored 0.0 (kept out of live_cache). Membership is
#: pinned against live_cache.UNSCOREABLE_STATUSES in test_live_cache.py.
_RUNTIME_UNSCOREABLE = ["empty_response", "error"]
_ALL_UNSCOREABLE = ["harness_error", *_RUNTIME_UNSCOREABLE]


@pytest.mark.parametrize("status", _ALL_UNSCOREABLE)
def test_unscoreable_status_is_retried_in_place_and_never_cached(status, runner_factory, tmp_path, monkeypatch):
    cache = LiveResultCache(tmp_path / "cache")
    candidate = {"agents_md": "text"}
    outcomes = iter([status, "pass_1"])
    seen = []
    for runner in runner_factory(cache=cache, on_result=seen.append):
        def _next(spec):
            s = next(outcomes)
            return _canned(spec, s, score=1.0 if s == "pass_1" else 0.0, error="boom")
        monkeypatch.setattr(runner, "_run_one_uncached", _next)
        results = runner.run_batch(candidate, [ExerciseSpec("wordy")])
        run_config = runner.exercise_run_config(ExerciseSpec("wordy"))

    assert [r.status for r in results] == ["pass_1"]
    # Both genuine runs reach the audit trail; only the real outcome is scored.
    assert [r.status for r in seen] == [status, "pass_1"]
    cached = cache.get(live_eval._sanitize_candidate(candidate), run_config, "python/wordy")
    assert cached["status"] == "pass_1"


def test_persistent_harness_error_raises_instead_of_scoring_zero(runner_factory, tmp_path, monkeypatch):
    """A harness failure says nothing about the candidate: scoring it 0.0
    would poison GEPA's search, and EvaluationBatch.scores must stay
    index-aligned, so it can't simply be dropped either."""
    cache = LiveResultCache(tmp_path / "cache")
    seen = []
    for runner in runner_factory(cache=cache, on_result=seen.append):
        monkeypatch.setattr(runner, "_run_one_uncached", lambda spec: _canned(spec, "harness_error", error="results file missing"))
        with pytest.raises(live_eval.LiveEvalHarnessError, match="results file missing"):
            runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])

    assert len(seen) == 1 + live_eval.HARNESS_ERROR_RETRIES
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("status", _RUNTIME_UNSCOREABLE)
def test_persistent_runtime_unscoreable_is_scored_zero_and_never_cached(status, runner_factory, tmp_path, monkeypatch, caplog):
    """A candidate can cause these deterministically (an AGENTS.md bloated
    past n_ctx, a skill file that crashes pi), so once the in-place retries
    are spent the candidate gets the 0.0 it earned -- one bad proposal must
    not abort the whole GEPA run. Kept out of live_cache, so a transient
    fault that outlasted the retries is not pinned across runs (GEPA's own
    EvaluationCache still records the 0.0 for the rest of this optimize())."""
    caplog.set_level(logging.WARNING, logger="benchmarks.self_improve.live_eval")
    cache = LiveResultCache(tmp_path / "cache")
    seen = []
    for runner in runner_factory(cache=cache, on_result=seen.append):
        monkeypatch.setattr(runner, "_run_one_uncached", lambda spec: _canned(spec, status, error="boom"))
        results = runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])

    assert [r.status for r in results] == [status]
    assert results[0].score == 0.0
    assert [r.status for r in seen] == [status] * (1 + live_eval.HARNESS_ERROR_RETRIES)
    assert not list((tmp_path / "cache").rglob("*.json"))
    final = [r.getMessage() for r in caplog.records if "still hit" in r.getMessage()]
    assert len(final) == 1
    for part in (status, "boom", "python/wordy", "0.0"):
        assert part in final[0]


def test_persistent_runtime_unscoreable_without_a_reason_is_logged_as_such(runner_factory, monkeypatch, caplog):
    """A pi crash or an empty response usually records no reason; the
    warning must say so rather than print None."""
    caplog.set_level(logging.WARNING, logger="benchmarks.self_improve.live_eval")
    for runner in runner_factory():
        monkeypatch.setattr(runner, "_run_one_uncached", lambda spec: _canned(spec, "empty_response"))
        results = runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])

    assert results[0].score == 0.0
    final = [r.getMessage() for r in caplog.records if "still hit" in r.getMessage()]
    assert len(final) == 1
    assert "<no reason recorded>" in final[0]


@pytest.mark.parametrize("status", _ALL_UNSCOREABLE)
def test_unscoreable_retries_count_against_the_live_budget(status, runner_factory, monkeypatch):
    import time as time_module

    from benchmarks.self_improve.live_budget import LiveBudget, LiveEvalBudgetExceeded

    budget = LiveBudget(hard_deadline_monotonic=time_module.monotonic() + 3600, max_live_runs=2)
    for runner in runner_factory(budget=budget):
        monkeypatch.setattr(runner, "_run_one_uncached", lambda spec: _canned(spec, status))
        with pytest.raises(LiveEvalBudgetExceeded):
            runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])


def test_fail_timeout_is_scored_once_not_retried_and_not_cached(runner_factory, tmp_path, monkeypatch):
    """Guards against over-extending the retry set: fail_timeout is scored
    (a looping agent is a candidate outcome) and a retry would cost up to a
    full per-attempt wall clock, but it is still never cached."""
    cache = LiveResultCache(tmp_path / "cache")
    calls = []
    for runner in runner_factory(cache=cache):
        def _timeout(spec):
            calls.append(spec.task_id)
            return _canned(spec, "fail_timeout")
        monkeypatch.setattr(runner, "_run_one_uncached", _timeout)
        results = runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])

    assert calls == ["python/wordy"]
    assert results[0].status == "fail_timeout"
    assert results[0].score == 0.0
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("reason", [
    "exercise not found at /somewhere/wordy",
    "unknown agent 'nope'",
    "RuntimeError: shared JS deps missing at /x/node_modules; create them with ...",
])
def test_config_error_raises_immediately_without_retrying(reason, runner_factory, tmp_path, monkeypatch):
    """These can't change on retry, and aren't the candidate's doing."""
    cache = LiveResultCache(tmp_path / "cache")
    seen = []
    for runner in runner_factory(cache=cache, on_result=seen.append):
        monkeypatch.setattr(runner, "_run_one_uncached", lambda spec: _canned(spec, "error", error=reason))
        with pytest.raises(live_eval.LiveEvalHarnessError, match="config"):
            runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])

    assert [r.status for r in seen] == ["error"]
    assert not list((tmp_path / "cache").rglob("*.json"))


def _canned_by_exercise(statuses: dict):
    def _run(spec):
        status = statuses[spec.exercise]
        return _canned(spec, status, score=1.0 if status == "pass_1" else 0.0, error="boom")
    return _run


def test_many_consecutive_persistent_unscoreables_across_candidates_never_raise(runner_factory, tmp_path, monkeypatch):
    """Nothing tries to tell a dead model server from a broken candidate:
    however many exercises in a row end error/empty_response, across
    candidates and run_batch() calls, each is scored 0.0 and kept out of
    live_cache, and the LiveBudget is what bounds the spend. A scored
    outcome in the middle of the batch keeps its place."""
    first = {"ex0": "error", "ex1": "empty_response", "ex2": "fail", "ex3": "error"}
    second = {"ex4": "empty_response", "ex5": "error", "ex6": "empty_response"}
    statuses = {**first, **second}
    unscoreable = [n for n, s in statuses.items() if s != "fail"]
    cache = LiveResultCache(tmp_path / "cache")
    seen = []
    for runner in runner_factory(cache=cache, on_result=seen.append):
        monkeypatch.setattr(runner, "_run_one_uncached", _canned_by_exercise(statuses))
        results = runner.run_batch({"agents_md": "candidate A"}, [ExerciseSpec(n) for n in first])
        results += runner.run_batch({"agents_md": "candidate B"}, [ExerciseSpec(n) for n in second])

    assert len(unscoreable) == 6
    assert [r.status for r in results] == list(statuses.values())
    assert all(r.score == 0.0 for r in results)
    assert len(seen) == len(unscoreable) * (1 + live_eval.HARNESS_ERROR_RETRIES) + 1
    assert [r.task_id for r in results if r.status != "fail"] == [f"python/{n}" for n in unscoreable]
    # Only the scored "fail" is cached.
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 1


def test_pi_dying_on_attempt_one_is_retried_then_scored_zero_uncached(runner_factory, tmp_path, monkeypatch):
    """Real subprocess chain: fake_pi acks then exits, rpc_client reports
    process_exit, aider_polyglot's attempt loop breaks and classifies the run
    "error". Retried in place, then scored 0.0 and kept out of live_cache."""
    monkeypatch.setenv("FAKE_PI_MODE", "crash_after_ack")
    cache = LiveResultCache(tmp_path / "cache")
    seen = []
    for runner in runner_factory(cache=cache, on_result=seen.append):
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])

    assert [r.status for r in seen] == ["error"] * (1 + live_eval.HARNESS_ERROR_RETRIES)
    assert results[0].status == "error"
    assert results[0].score == 0.0
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_missing_exercise_is_a_config_error_and_raises_after_one_run(runner_factory, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "clean")
    cache = LiveResultCache(tmp_path / "cache")
    seen = []
    for runner in runner_factory(cache=cache, on_result=seen.append):
        with pytest.raises(live_eval.LiveEvalHarnessError, match="exercise not found"):
            runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("no_such_exercise")])

    assert [r.status for r in seen] == ["error"]
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_a_stale_results_file_is_not_read_back_as_this_runs_outcome(runner_factory):
    """A subprocess that dies before writing its own results must not be
    scored from whatever the previous try left in the worktree."""
    for runner in runner_factory():
        results_file = runner.worktree.path / "benchmarks" / "results_full_polyglot.json"
        results_file.write_text(json.dumps({"exercises": {"pi/python/wordy": {"status": "pass_1"}}}))
        runner.python_executable = shutil.which("false") or "/usr/bin/false"
        result = runner._run_one_uncached(ExerciseSpec("wordy"))

    assert result.status == "harness_error"
    assert "results file missing" in result.error


def test_non_utf8_results_file_is_retried_then_raises_harness_error(runner_factory, tmp_path):
    """A results file that is not UTF-8 used to raise UnicodeDecodeError out
    of _parse_result, past the harness_error retry path and before the
    budget and on_result bookkeeping for a run that did happen."""
    child = tmp_path / "bad_results_child.sh"
    # _interpreter_info() also runs this executable, without the variable.
    child.write_text(
        '#!/bin/sh\n'
        '[ -n "$POLYGLOT_RESULTS_FILE" ] || exit 1\n'
        "printf '{\"exercises\": {\"pi/python/wordy\": {\"status\": \"pass_1\\377\"}}}' "
        '> "$POLYGLOT_RESULTS_FILE"\n'
    )
    child.chmod(0o755)
    cache = LiveResultCache(tmp_path / "cache")
    seen = []
    for runner in runner_factory(cache=cache, on_result=seen.append):
        runner.python_executable = str(child)
        with pytest.raises(live_eval.LiveEvalHarnessError, match="malformed results file"):
            runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])

    assert [r.status for r in seen] == ["harness_error"] * (1 + live_eval.HARNESS_ERROR_RETRIES)
    assert "UnicodeDecodeError" in seen[0].error
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_a_non_utf8_agent_file_still_reaches_the_budget_and_on_result(runner_factory, tmp_path):
    """The child scores a pass and its snapshot holds an agent file with one
    latin-1 byte. _compute_diff used to raise UnicodeDecodeError after the
    run, so run_batch() never recorded it against the budget, never reported
    it to on_result, and never cached it."""
    import time as time_module

    from benchmarks.self_improve.live_budget import LiveBudget

    child = tmp_path / "latin1_snapshot_child.sh"
    child.write_text(
        '#!/bin/sh\n'
        '[ -n "$POLYGLOT_RESULTS_FILE" ] || exit 1\n'
        'd="$POLYGLOT_LOG_ROOT/pi/python/wordy"\n'
        'mkdir -p "$d/workdir_1"\n'
        "printf '{}' > \"$d/trajectory_1.json\"\n"
        "printf 'x = \\047\\351\\047\\n' > \"$d/workdir_1/wordy.py\"\n"
        "printf '{\"exercises\": {\"pi/python/wordy\": {\"status\": \"pass_1\"}}}' "
        '> "$POLYGLOT_RESULTS_FILE"\n'
    )
    child.chmod(0o755)
    cache = LiveResultCache(tmp_path / "cache")
    budget = LiveBudget(hard_deadline_monotonic=time_module.monotonic() + 3600, max_live_runs=5)
    seen = []
    for runner in runner_factory(cache=cache, budget=budget, on_result=seen.append):
        runner.python_executable = str(child)
        results = runner.run_batch({"agents_md": "text"}, [ExerciseSpec("wordy")])

    assert results[0].status == "pass_1"
    assert "agent/wordy.py" in results[0].diff_summary
    assert [r.status for r in seen] == ["pass_1"]
    assert budget.live_runs == 1
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 1


def test_live_run_result_from_dict_ignores_an_unknown_key():
    """A stray key in a cache entry must load, not raise TypeError -- a cache
    read can't be allowed to kill an in-flight run."""
    entry = _canned(ExerciseSpec("wordy"), "fail").to_dict()
    entry["unexpected"] = False
    loaded = live_eval.LiveRunResult.from_dict(entry)
    assert loaded == live_eval.LiveRunResult.from_dict(_canned(ExerciseSpec("wordy"), "fail").to_dict())


def _exercise_key(runner, name="wordy"):
    return run_config_hash(runner.exercise_run_config(ExerciseSpec(name)))


@pytest.mark.parametrize("relpath", ["wordy_test.py", "wordy.py", ".meta/config.json"])
def test_exercise_cache_key_changes_when_the_exercises_benchmark_inputs_change(
    runner_factory, fake_practice, relpath,
):
    """Same benchmark_root path, different stub/test/.meta content: a cached
    score measured against the old inputs must not be served."""
    ex_dir = fake_practice / "python" / "exercises" / "practice" / "wordy"
    (ex_dir / ".meta").mkdir(exist_ok=True)
    (ex_dir / ".meta" / "config.json").write_text('{"files": {"test": ["wordy_test.py"]}}')
    for runner in runner_factory():
        before = _exercise_key(runner)
    target = ex_dir / relpath
    target.write_text(target.read_text() + "\n# changed\n")
    for runner in runner_factory():
        after = _exercise_key(runner)
    assert before != after


def test_exercise_cache_key_is_per_exercise(runner_factory, fake_practice):
    """Editing one exercise must not invalidate every other exercise's entry."""
    practice = fake_practice / "python" / "exercises" / "practice"
    other = practice / "other"
    other.mkdir()
    (other / "other.py").write_text("x = 1\n")
    for runner in runner_factory():
        wordy_before, other_before = _exercise_key(runner), _exercise_key(runner, "other")
    (other / "other.py").write_text("x = 2\n")
    for runner in runner_factory():
        assert _exercise_key(runner) == wordy_before
        assert _exercise_key(runner, "other") != other_before


def test_exercise_inputs_are_hashed_once_per_runner(runner_factory, fake_practice, monkeypatch):
    calls = []
    real = live_eval._exercise_inputs_fingerprint
    monkeypatch.setattr(live_eval, "_exercise_inputs_fingerprint", lambda p: calls.append(p) or real(p))
    for runner in runner_factory():
        _exercise_key(runner)
        _exercise_key(runner)
    assert len(calls) == 1


def test_run_config_records_the_resolved_inner_timeout_even_with_an_explicit_outer_one(
    runner_factory, monkeypatch,
):
    """runner_factory passes per_exercise_timeout_s=60 explicitly, so the
    child's own ATTEMPT_TIMEOUT_S used to reach no part of the key."""
    for runner in runner_factory():
        assert runner.per_exercise_timeout_s == 60
        assert runner.run_config["attempt_timeout_s"] == 30
        before = run_config_hash(runner.run_config)
        monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "31")
        assert runner.run_config["attempt_timeout_s"] == 31
        assert run_config_hash(runner.run_config) != before


def test_run_batch_misses_the_cache_after_the_exercises_tests_change(
    runner_factory, fake_practice, tmp_path, monkeypatch,
):
    cache = LiveResultCache(tmp_path / "cache")
    candidate = {"agents_md": "text"}
    for runner in runner_factory(cache=cache):
        monkeypatch.setattr(runner, "_run_one_uncached", lambda spec: _canned(spec, "pass_1", score=1.0))
        runner.run_batch(candidate, [ExerciseSpec("wordy")])
    test_file = fake_practice / "python" / "exercises" / "practice" / "wordy" / "wordy_test.py"
    test_file.write_text(test_file.read_text() + "\ndef test_more():\n    assert False\n")
    calls = []
    for runner in runner_factory(cache=cache):
        monkeypatch.setattr(runner, "_run_one_uncached",
                            lambda spec: calls.append(spec) or _canned(spec, "fail"))
        results = runner.run_batch(candidate, [ExerciseSpec("wordy")])
    assert len(calls) == 1
    assert results[0].from_cache is False and results[0].status == "fail"
