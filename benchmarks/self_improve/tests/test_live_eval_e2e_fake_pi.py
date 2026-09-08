"""End-to-end live-eval pipeline test: REAL `git worktree add`, REAL
candidate file writes, a REAL subprocess invocation of aider_polyglot.py, a
REAL `pytest -x -q` scoring run, REAL result-JSON parsing -- with `pi`
routed through fake_pi.py so zero model calls happen. This is the centerpiece
test of the whole live-execution rewrite: it exercises the exact mechanism
(rpc_client.REPO_ROOT resolving to the SCRATCH worktree because
aider_polyglot.py is invoked as a subprocess whose own __file__ lives there)
that makes a candidate's text actually reach a live agent.
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import benchmarks.self_improve.live_eval as live_eval
from benchmarks.self_improve.exercises import ExerciseSpec
from benchmarks.self_improve.live_cache import LiveResultCache
from benchmarks.self_improve.live_eval import PolyglotLiveRunner
from benchmarks.self_improve.scratch_worktree import scratch_worktree

# This module unconditionally drives real `git init`/`git worktree add` and
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
    """Real gap, confirmed by review: compaction_events was discarded
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
    """THE regression test for the bug this entire live-execution rewrite
    exists to fix: under the old frozen-historical-data design, metric()
    never read the candidate's actual text, so every candidate scored
    identically and GEPA's acceptance criterion could never fire. Here, two
    candidates differing ONLY in instruction text produce genuinely
    different real outcomes, because fake_pi decides whether to solve based
    on what it actually reads from the live system prompt."""
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


def test_materialize_raises_when_candidate_has_a_component_not_in_components_yaml(runner_factory):
    """Real bug, confirmed by review: write_components_back() only logs a
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


def test_run_config_pi_bin_changes_with_a_different_binary(runner_factory, tmp_path):
    """Real bug, confirmed by review: the pi binary IS the agent under
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
    """Real gap, confirmed by review: pass_n_score() (aider_polyglot_ingest.py,
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
    """Real bug, confirmed by review: clamping the subprocess timeout to
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
    """Real bug, confirmed by review: the audit callback used to be invoked
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
    """Real bug, confirmed by review: this default used to hardcode a bare
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

    # Real gap, confirmed by review (twice, then a third round): the first
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
    """Real gap, confirmed by review: the outer per-exercise timeout
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

    # Real gap, confirmed by review: an earlier version of this test
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
    """Real gap, confirmed by review: a malformed/non-positive
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
    """Real gap, confirmed by review: a benign reformat of aider_polyglot.py
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
