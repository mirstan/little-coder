"""LiveRunResult + PolyglotLiveRunner: materializes a GEPA candidate into a
scratch worktree, runs a real aider_polyglot.py exercise as a subprocess per
exercise (not in-process -- see scratch_worktree.py's module docstring for
why), and parses the result into a graded score plus real feedback material
(diff, pytest output, transcript excerpt, reasoning excerpt) for the
reflection step.
"""
from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import yaml

import benchmarks.self_improve.components as _components_module
import benchmarks.self_improve.ingest.aider_polyglot_ingest as _aider_polyglot_ingest_module
from benchmarks.self_improve.components import split_frontmatter, write_components_back
from benchmarks.self_improve.exercises import ExerciseSpec, practice_dir
from benchmarks.self_improve.ingest.aider_polyglot_ingest import pass_n_score
from benchmarks.self_improve.ingest.common import summarize_for_reflection
from benchmarks.self_improve.live_budget import LiveEvalBudgetExceeded
from benchmarks.self_improve.live_cache import LiveResultCache
from benchmarks.self_improve.scratch_worktree import ScratchWorktree

logger = logging.getLogger(__name__)

#: Mirrors aider_polyglot.py's own _ATTEMPT_TIMEOUT_S_DEFAULT -- read
#: directly out of THAT module's own source text via regex, not a
#: duplicated bare literal here, so the two can never silently drift apart
#: the way this harness-level default already did once: it was still 900
#: when aider_polyglot.py's own default was tripled to 2700 for a local
#: reasoning model, meaning the OUTER subprocess timeout below would have
#: fired and killed the exercise via SIGTERM/SIGKILL before even ONE inner
#: attempt's own (now longer) budget had a chance to time out gracefully --
#: turning a clean, correctly-classified fail_timeout into an abrupt
#: harness_error instead.
#:
#: Regex-extracted rather than imported -- two real gaps, confirmed by
#: review, with an actual `import benchmarks.aider_polyglot`:
#: 1. A plain `import` executes that module's ENTIRE top level, including
#:    `CODEX_TIMEOUT_S = _positive_int_env("CODEX_TIMEOUT_S", 900)` --
#:    which raises SystemExit on a malformed CODEX_TIMEOUT_S even though
#:    this module never uses CODEX_TIMEOUT_S at all, crashing e.g. the
#:    free --estimate-only path (which spawns no subprocess) over an
#:    unrelated, unused env var.
#: 2. It reads the SOURCE checkout's copy at THIS process' import time --
#:    but a PolyglotLiveRunner's actual subprocess runs the WORKTREE's
#:    pinned base_commit copy, which can differ (an uncommitted local edit
#:    to aider_polyglot.py itself, mid-development on the harness while a
#:    live run is in progress). Computing the outer per-exercise timeout
#:    from the wrong copy's default could make it shorter than the pinned
#:    copy's actual per-attempt budget, killing the child before its own
#:    (longer) timeout fires gracefully.
#: One regex-based reader (_attempt_timeout_default_from_source) fixes
#: both: it never executes the file (no CODEX_TIMEOUT_S side effect), and
#: PolyglotLiveRunner.__init__ below calls it against `worktree.path`
#: specifically, not the source checkout, so the value actually used
#: always matches what will actually execute. `_ATTEMPT_TIMEOUT_S_DEFAULT`
#: itself (the source-checkout read, computed once below) remains only as
#: the --estimate-only / no-worktree fallback and this function's own
#: default argument. Mirrors components.py's own _TOKEN_COST_LINE_RE
#: precedent for reading a single declared value out of a companion file
#: without executing it.
_ATTEMPT_TIMEOUT_S_DEFAULT_RE = re.compile(r"(?m)^_ATTEMPT_TIMEOUT_S_DEFAULT = (\d+)$")
#: Last-resort fallback if even the source checkout's own aider_polyglot.py
#: can't be read/parsed (should not happen in a working checkout) -- a
#: budget ESTIMATE, not a correctness-critical read, so degrading to a
#: hardcoded value here is preferable to crashing an otherwise-healthy run.
_ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK = 2700


def _attempt_timeout_default_from_source(aider_polyglot_py_path: Path) -> int:
    """Regex-extract aider_polyglot.py's _ATTEMPT_TIMEOUT_S_DEFAULT
    constant from the given copy's file TEXT -- never imports/executes it
    (see the module-level comment above for why). Never raises."""
    try:
        text = aider_polyglot_py_path.read_text()
    except OSError:
        return _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK
    m = _ATTEMPT_TIMEOUT_S_DEFAULT_RE.search(text)
    return int(m.group(1)) if m else _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK


_ATTEMPT_TIMEOUT_S_DEFAULT = _attempt_timeout_default_from_source(
    Path(__file__).resolve().parent.parent / "aider_polyglot.py"
)


def _attempt_timeout_s(default: int = _ATTEMPT_TIMEOUT_S_DEFAULT) -> int:
    """Real gap, confirmed by review: a malformed or non-positive
    ATTEMPT_TIMEOUT_S used to be silently swallowed here and replaced with
    the default, computing a per_exercise_timeout_s estimate as if the run
    would proceed normally -- but aider_polyglot.py's own
    _positive_int_env() raises SystemExit on the exact same malformed/
    non-positive value, so the actual subprocess would crash at import
    time instead. Mirror that same validate-and-raise contract here so an
    orchestrator-side budget estimate can never be computed against a
    value that's actually going to blow up the exercise it's estimating
    for.

    `default` lets a caller with a specific worktree in hand (see
    _attempt_timeout_default_from_source()) pass that worktree's own
    pinned default instead of the source checkout's."""
    raw = os.environ.get("ATTEMPT_TIMEOUT_S")
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"ATTEMPT_TIMEOUT_S: expected a positive integer (seconds), got {raw!r}")
    if value <= 0:
        raise SystemExit(f"ATTEMPT_TIMEOUT_S: must be > 0, got {value}")
    return value


_MAX_TAIL_CHARS = 4_000
_MAX_TRANSCRIPT_CHARS = 4_000
_MAX_DIFF_CHARS = 6_000
#: Reasoning traces run long (a single real bowling attempt hit the 200-entry
#: non_text_deltas cap in aider_polyglot.py's own trajectory dump, almost
#: all of it thinking_delta) -- tail-truncate like transcript_excerpt so the
#: reflection prompt sees the model's LATEST reasoning, not its opening
#: thoughts truncated mid-sentence.
_MAX_REASONING_CHARS = 4_000
_ATTEMPT_NUM_RE = re.compile(r"_(\d+)$")


def _reasoning_excerpt_from_trajectory(traj_data: Mapping) -> str:
    """Reconstructs the model's reasoning stream from non_text_deltas the
    same way rpc_client.py's own prompt_and_collect() builds assistant_text
    from text_delta: concatenate each thinking_delta's "delta" field in
    order. Confirmed against a real gepa.optimize() run (2026-09-06,
    omlx/tiel-coder-oq4e, thinking=high) that "thinking_delta" is pi's real
    event type for this -- previously only a guessed stand-in (see
    PromptResult.non_text_deltas' own docstring in rpc_client.py).

    Entries can be non-dict here: aider_polyglot.py's _dump_trajectory._clip
    falls back to a truncated JSON string for any single delta that
    serializes past TRAJECTORY_FIELD_CHARS -- skip those defensively rather
    than crash on a malformed cache/trajectory entry."""
    chunks = [
        d.get("delta", "")
        for d in traj_data.get("non_text_deltas", [])
        if isinstance(d, dict) and d.get("type") == "thinking_delta"
    ]
    return "".join(chunks)[-_MAX_REASONING_CHARS:]


def _attempt_num(p: Path) -> int:
    m = _ATTEMPT_NUM_RE.search(p.stem)
    return int(m.group(1)) if m else -1


def _strip_leading_frontmatter_block(text: str) -> tuple[str, bool]:
    """If text starts with a `---\\n...\\n---\\n` block, strip it and report
    whether it did. A reflection LM might "helpfully" re-emit a YAML header
    even though load_components() already handed it frontmatter-stripped
    body text -- reattach_frontmatter() would then concatenate a SECOND
    header after the real one, and skill-inject's parseSkillFile() finds no
    target_tool, silently killing injection for every subsequent candidate
    while scores just look uniformly bad. Confirmed non-obvious failure mode
    from planning -- see the live-eval plan doc.

    Reuses components.split_frontmatter (same delimiter, same windowing) so
    the two frontmatter-stripping implementations can't drift apart."""
    frontmatter, body = split_frontmatter(text)
    return (body, True) if frontmatter is not None else (text, False)


def _sanitize_candidate(candidate: Mapping[str, str]) -> dict[str, str]:
    sanitized = {}
    for name, text in candidate.items():
        stripped, changed = _strip_leading_frontmatter_block(text)
        if changed:
            logger.warning(
                "live_eval: candidate %r's proposed text re-emitted a YAML "
                "frontmatter block -- stripped before writing, to avoid "
                "corrupting the real file's own frontmatter.", name,
            )
        sanitized[name] = stripped
    return sanitized


@dataclass
class LiveRunResult:
    task_id: str
    exercise: str
    language: str
    status: str
    score: float
    success: bool
    attempts: int = 0
    stop_reasons: list = field(default_factory=list)
    elapsed_s: float = 0.0
    turn_count: int = 0
    test_output_tail: str = ""
    transcript_excerpt: str = ""
    reasoning_excerpt: str = ""
    #: Summed across every attempt (mirrors turn_count), from
    #: aider_polyglot.py's own compaction_total -- a candidate whose
    #: injected text is large enough to force a mid-run context compaction
    #: is exhibiting real prompt bloat, factored into pass_n_score() above.
    compaction_total: int = 0
    #: One entry per attempt that had a retry (Reflexion-style: the agent
    #: reflects on why THAT attempt failed as part of its own next retry
    #: prompt -- see aider_polyglot.py's LESSON: convention). Unverified --
    #: the agent's own claim, never independently checked, never affects
    #: score. See polyglot_adapter.py's disclaimer text when this is surfaced.
    self_reported_lessons: list = field(default_factory=list)
    #: Error tool calls first ("[ERROR] name(args) -> result_text"), then a
    #: backfill of assistant_text's tail -- see
    #: ingest/common.py::summarize_for_reflection, reused verbatim here so a
    #: tool crash reaches reflection deliberately instead of only incidentally
    #: (via the model's own prose, or a downstream pytest failure).
    summarized_transcript: str = ""
    diff_summary: str = ""
    notifications: list = field(default_factory=list)
    error: str | None = None
    from_cache: bool = False
    exit_code: int | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "LiveRunResult":
        return cls(**dict(d))


class PolyglotLiveRunner:
    """Runs real aider_polyglot.py exercises, one subprocess per exercise,
    against a candidate materialized into a shared scratch worktree (reused
    across evaluations within one optimize() call -- GEPA's own
    default_batch_evaluate is confirmed sequential, so this is safe as long
    as no adapter-level batch_evaluate is ever defined on top of it)."""

    def __init__(
        self,
        *,
        worktree: ScratchWorktree,
        components_yaml: Path,
        model: str,
        language: str = "python",
        max_attempts: int = 2,
        retry: bool = True,
        thinking: str | None = None,
        benchmark_root: Path | None = None,
        cache: LiveResultCache | None = None,
        per_exercise_timeout_s: int | None = None,
        python_executable: str = sys.executable,
        budget=None,
        on_result=None,
    ):
        self.worktree = worktree
        self.components_yaml = Path(components_yaml)
        self.model = model
        self.language = language
        self.max_attempts = max_attempts
        self.retry = retry
        self.thinking = thinking
        self.benchmark_root = Path(benchmark_root) if benchmark_root else None
        self.cache = cache
        #: Optional live_budget.LiveBudget -- checked before every live run
        #: (never before a cache hit) and RAISES rather than letting a run
        #: it refuses to start silently score 0.0, which would poison the
        #: search with a reason that has nothing to do with the candidate.
        self.budget = budget
        # aider_polyglot's own ATTEMPT_TIMEOUT_S per attempt (see
        # _attempt_timeout_s() above) + 90s test budget, times max_attempts,
        # plus headroom -- a belt-and-braces ceiling ABOVE aider_polyglot's
        # own per-attempt budget so a wedged pi session can't stall a run
        # indefinitely. Default sourced from THIS worktree's own pinned
        # copy (_attempt_timeout_default_from_source), not the source
        # checkout's -- real gap, confirmed by review: the two can diverge
        # under an uncommitted local edit to aider_polyglot.py itself, and
        # using the wrong one here could compute an outer timeout shorter
        # than the pinned copy's actual per-attempt budget.
        worktree_default = _attempt_timeout_default_from_source(
            worktree.path / "benchmarks" / "aider_polyglot.py"
        )
        self.per_exercise_timeout_s = per_exercise_timeout_s or (
            max_attempts * (_attempt_timeout_s(default=worktree_default) + 90) + 180
        )
        self.python_executable = python_executable
        #: Optional callable(LiveRunResult) -- invoked as EACH result becomes
        #: available inside run_batch() (cache hits included), not after the
        #: whole batch returns. A caller auditing every live invocation
        #: (run_gepa.py's SpendLog) would otherwise lose every
        #: already-completed result in a batch that raises partway through
        #: (e.g. LiveBudget's own backstop firing on exercise N of M).
        self.on_result = on_result

    @property
    def run_config(self) -> dict:
        """Everything besides the candidate text that changes what a score
        means. Includes the harness files' own sha256 so a mid-project
        harness bugfix can't let a stale cache entry poison a later
        comparison.

        aider_polyglot.py/rpc_client.py are hashed from the WORKTREE copy
        (pinned at base_commit, which is itself already part of this
        config), not the source repo -- they run as a SUBPROCESS inside the
        scratch worktree (see this module's own docstring), so that copy is
        exactly what executes; hashing the source would make an uncommitted
        debug edit or a branch switch in the source checkout change the
        cache key without changing a single byte of what actually executes,
        causing spurious re-runs of already-cached (and possibly still
        in-flight) work.

        aider_polyglot_ingest.py/components.py/this module itself are the
        OPPOSITE case, hashed from the module `__file__` this SAME (parent,
        orchestrator) process actually imported -- confirmed real gap by
        review: pass_n_score() and write_components_back() run here, in the
        parent process (imported at the top of this module), never inside
        the scratch worktree at all, so hashing the worktree's copy of them
        (as an earlier version of this property did) couldn't detect an
        uncommitted retune of e.g. _COMPACTION_PENALTY or
        _estimate_token_cost -- LiveResultCache would keep serving scores
        computed under the old formula even though the orchestrator's own
        process had already picked up the edit. This module's own
        `__file__` is included for the identical reason, confirmed real
        gap by a second review round: _parse_result() below (the
        self_reported_lessons cap, pass_n_score() call site, etc.) also
        runs here in the parent process -- an uncommitted edit to this
        module's own processing logic used to leave a persistent cache
        entry computed under the OLD logic being served forever, since
        nothing in run_config depended on this file's own content."""
        worktree_executed_files = [
            self.worktree.path / "benchmarks" / "aider_polyglot.py",
            self.worktree.path / "benchmarks" / "rpc_client.py",
        ]
        parent_imported_files = [
            Path(_aider_polyglot_ingest_module.__file__),
            Path(_components_module.__file__),
            Path(__file__),
        ]
        hasher = hashlib.sha256()
        for f in worktree_executed_files + parent_imported_files:
            if f.exists():
                hasher.update(f.read_bytes())
        pi_bin = Path(self.worktree.pi_bin)
        try:
            pi_bin_mtime = pi_bin.stat().st_mtime
        except OSError:
            pi_bin_mtime = None
        return {
            "model": self.model,
            "language": self.language,
            "max_attempts": self.max_attempts,
            "retry": self.retry,
            "thinking": self.thinking,
            "base_commit": self.worktree.base_commit,
            "harness_hash": hasher.hexdigest(),
            # A different benchmark checkout can mean different stub/test
            # content for "the same" exercise name, and a different timeout
            # can turn a would-be timeout into a genuine pass (or vice
            # versa) -- both change what a cached score actually measures.
            "benchmark_root": str(self.benchmark_root) if self.benchmark_root else None,
            "per_exercise_timeout_s": self.per_exercise_timeout_s,
            # The pi binary IS the agent under test -- omitting it means a
            # smoke run through fake_pi.py (LITTLE_CODER_PI_BIN_OVERRIDE) can
            # populate the cache with fabricated results that a later real
            # run, with an identical candidate/model/etc, would then read
            # back as genuine. mtime (not a full content hash) also catches
            # an `npm install` bumping the real pi binary mid-project.
            "pi_bin": str(pi_bin),
            "pi_bin_mtime": pi_bin_mtime,
        }

    def materialize(self, candidate: Mapping[str, str]) -> list[Path]:
        """Reset the worktree to its pinned base commit, then write only the
        candidate's (sanitized) text into place, verifying nothing else in
        the tree changed."""
        self.worktree.reset()
        sanitized = _sanitize_candidate(candidate)
        mapping = yaml.safe_load(Path(self.components_yaml).read_text()) or {}
        unmapped = sorted(set(sanitized) - set(mapping))
        if unmapped:
            # write_components_back() only logs a warning and skips a
            # pred_name absent from the pinned components.yaml -- every GEPA
            # rewrite of an unmapped component would then be silently
            # dropped, so every candidate variant materializes to the SAME
            # worktree content and scores identically: the exact "score
            # independent of candidate text" failure class this whole
            # live-execution design exists to eliminate.
            raise ValueError(
                f"candidate has component(s) {unmapped} not present in "
                f"{self.components_yaml} at the pinned commit {self.worktree.base_commit} -- "
                "every proposed rewrite of these would be silently dropped and scored as a "
                "no-op. Commit a components.yaml entry for them first, or use --only-components "
                "to scope the candidate down to what's actually mapped."
            )
        changed = write_components_back(self.components_yaml, self.worktree.path, sanitized)
        self.worktree.assert_only_expected_dirty(changed)
        return changed

    def run_batch(self, candidate: Mapping[str, str], specs: Sequence[ExerciseSpec]) -> list[LiveRunResult]:
        """Cache-first, materialize-once: checks the on-disk memo for every
        requested exercise before touching the worktree at all; only if
        there's at least one miss does it reset+write the candidate, then
        runs each missed exercise. Returns results in the SAME order as
        `specs` (a hard requirement for the GEPA adapter built on top of
        this -- EvaluationBatch.scores must align index-for-index with the
        batch)."""
        run_config = self.run_config
        results: dict[str, LiveRunResult] = {}
        misses: list[ExerciseSpec] = []
        for spec in specs:
            cached = self.cache.get(candidate, run_config, spec.task_id) if self.cache else None
            if cached is not None:
                result = LiveRunResult.from_dict(cached)
                result.from_cache = True
                results[spec.task_id] = result
                if self.on_result is not None:
                    self.on_result(result)
            else:
                misses.append(spec)

        if misses:
            self.materialize(candidate)
            for spec in misses:
                if self.budget is not None:
                    self.budget.check_before_exercise(spec.task_id)  # raises rather than faking a score
                result = self._run_one_uncached(spec)
                if self.budget is not None:
                    self.budget.record_live_run()
                results[spec.task_id] = result
                # Emitted here, per-result, rather than after the whole batch
                # returns -- a later exercise in this same batch raising
                # (e.g. the budget backstop above) must not erase the audit
                # trail for exercises that already genuinely ran. Fired
                # BEFORE cache.put(): if the memo write itself raises (disk
                # I/O), the audit record for an exercise that DID genuinely
                # run must not be lost along with it.
                if self.on_result is not None:
                    self.on_result(result)
                if self.cache is not None:
                    self.cache.put(candidate, run_config, spec.task_id, result.to_dict())

        return [results[spec.task_id] for spec in specs]

    def _run_one_uncached(self, spec: ExerciseSpec) -> LiveRunResult:
        script = self.worktree.path / "benchmarks" / "aider_polyglot.py"
        cmd = [
            self.python_executable, str(script),
            "--exercise", spec.exercise, "--language", spec.language,
            "--model", self.model, "--max-attempts", str(self.max_attempts),
        ]
        if not self.retry:
            cmd.append("--no-retry")
        if self.thinking:
            cmd.extend(["--thinking", self.thinking])

        env = self.worktree.env()
        results_file = self.worktree.path / "benchmarks" / "results_full_polyglot.json"
        log_root = self.worktree.path / "benchmarks" / "full_polyglot_logs"
        env["POLYGLOT_RESULTS_FILE"] = str(results_file)
        env["POLYGLOT_LOG_ROOT"] = str(log_root)
        if self.benchmark_root:
            env["POLYGLOT_BENCHMARK_ROOT"] = str(self.benchmark_root)

        # check_before_exercise() only gates whether an exercise may START --
        # without also bounding THIS timeout, a single exercise's own (much
        # larger) per_exercise_timeout_s could let --max-wall-clock-s be
        # exceeded by up to one full exercise timeout once it's underway.
        effective_timeout = self.per_exercise_timeout_s
        budget_clamped = False
        if self.budget is not None:
            remaining = self.budget.remaining_seconds()
            if remaining < effective_timeout:
                effective_timeout = max(1.0, remaining)
                budget_clamped = True

        # Written BEFORE Popen() -- see mark_spawn_pending()'s own docstring
        # for the TOCTOU gap this closes (a SIGKILL between Popen() returning
        # and set_active_pid(proc.pid) below would otherwise leave no marker
        # evidence that a subprocess was ever started).
        self.worktree.mark_spawn_pending()
        proc = subprocess.Popen(
            cmd, cwd=str(self.worktree.path), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        # Recorded so gepa_scratch_gc.py can tell "the orchestrator died" apart
        # from "this exercise subprocess is still alive and using the
        # worktree" -- start_new_session=True means this child has its own
        # process group/pid and does NOT die when the orchestrator does.
        try:
            self.worktree.set_active_pid(proc.pid)
            try:
                _stdout, stderr = proc.communicate(timeout=effective_timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                self._kill_process_group(proc)
                _stdout, stderr = proc.communicate()
                if budget_clamped:
                    # This kill was a budget decision, not a genuine
                    # exercise timeout -- the exercise might well have
                    # passed given its full per_exercise_timeout_s. Raising
                    # (rather than returning a "harness_error"/0.0 result)
                    # honors LiveBudget's own documented invariant: a budget
                    # refusal must never look like a real, scoreable outcome.
                    raise LiveEvalBudgetExceeded(
                        f"wall-clock hard deadline reached while running {spec.task_id!r} "
                        f"(killed after {effective_timeout:.0f}s of remaining budget, "
                        f"short of its {self.per_exercise_timeout_s}s normal timeout)"
                    )
                return LiveRunResult(
                    task_id=spec.task_id, exercise=spec.exercise, language=spec.language,
                    status="harness_error", score=0.0, success=False,
                    error=f"subprocess timed out after {effective_timeout:.0f}s",
                )
        except BaseException:
            # Any other exception unwinding here (notably the
            # KeyboardInterrupt run_gepa.py's own signal handler raises on a
            # second Ctrl-C, but also one raised by set_active_pid()'s own
            # marker-file I/O before communicate() is even reached) must not
            # leave this detached (start_new_session=True) subprocess
            # running -- it never received the interrupt itself, and would
            # otherwise keep driving a real paid rollout against a worktree
            # the caller may be about to remove. Calling this twice (the
            # budget_clamped raise above already killed it) is safe --
            # _kill_process_group treats an already-dead process as a no-op.
            self._kill_process_group(proc)
            raise
        finally:
            self.worktree.set_active_pid(None)

        return self._parse_result(spec, results_file, log_root, stderr, exit_code)

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        """A plain subprocess timeout only kills the DIRECT child --
        aider_polyglot.py's own comments document that a spawned bash
        grandchild (from the agent's own tool calls) can still outlive
        that. start_new_session=True (in _run_one_uncached) makes this
        process its own session AND process group leader, so its pgid is
        exactly proc.pid at creation time -- used directly here rather than
        looked up via os.getpgid(proc.pid), which can raise ProcessLookupError
        (and previously caused this whole method to give up) once the direct
        child has exited even while its process group still has live
        descendants to reach.

        Escalation to SIGKILL is decided from the GROUP's own liveness, not
        proc.wait()'s return: if the direct child exits quickly but a
        descendant still holds the pipes open and ignores SIGTERM,
        proc.wait() returns immediately (only ever waiting on the direct
        child) -- naively treating that as "done" would skip SIGKILL
        entirely and leave the caller's subsequent communicate() call
        hanging forever on pipes that never close."""
        pgid = proc.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(pgid, 0)  # raises ProcessLookupError iff the WHOLE group is gone
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _parse_result(
        self, spec: ExerciseSpec, results_file: Path, log_root: Path, stderr: str, exit_code: int | None,
    ) -> LiveRunResult:
        base_kwargs = dict(task_id=spec.task_id, exercise=spec.exercise, language=spec.language, exit_code=exit_code)

        if not results_file.exists():
            return LiveRunResult(
                status="harness_error", score=0.0, success=False,
                error=f"results file missing: {results_file}\nstderr tail: {stderr[-2000:]}",
                **base_kwargs,
            )
        try:
            data = json.loads(results_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return LiveRunResult(
                status="harness_error", score=0.0, success=False,
                error=f"malformed results file: {e}", **base_kwargs,
            )

        record = data.get("exercises", {}).get(spec.results_key)
        if record is None:
            return LiveRunResult(
                status="harness_error", score=0.0, success=False,
                error=f"no record for {spec.results_key!r} in {results_file}", **base_kwargs,
            )

        status = record.get("status", "error")
        compaction_total = record.get("compaction_total", 0) or 0
        success, score = pass_n_score(status, compaction_events=compaction_total) or (False, 0.0)
        stop_reasons = record.get("stop_reasons") or []
        # aider_polyglot.py caps each individual LESSON: line at
        # LESSON_MAX_CHARS (500) when extracting it, but that's per-attempt --
        # with --max-attempts set high, the cumulative joined text this
        # adapter later inserts into GEPA reflection feedback (see
        # polyglot_adapter.py) had no overall cap. Real gap, confirmed by
        # review: one evaluated agent's chain of long lessons could still
        # overflow reflection context or inflate cost. Capped here at the
        # same budget other reflection-bound fields already use.
        raw_lessons = record.get("lessons")
        self_reported_lessons: list = []
        if isinstance(raw_lessons, list):
            remaining = _MAX_TRANSCRIPT_CHARS
            for lesson in raw_lessons:
                if not isinstance(lesson, str) or remaining <= 0:
                    continue
                clipped = lesson[:remaining]
                self_reported_lessons.append(clipped)
                remaining -= len(clipped)

        ex_log_dir = log_root / "pi" / spec.language / spec.exercise
        test_output_tail = ""
        final_output = ex_log_dir / "final_output.txt"
        if final_output.exists():
            test_output_tail = final_output.read_text()[-_MAX_TAIL_CHARS:]

        transcript_excerpt = ""
        reasoning_excerpt = ""
        summarized_transcript = ""
        # Unioned across EVERY attempt, not just the latest: a component
        # (e.g. a skill) can be injected on an earlier failed attempt whose
        # guidance still shapes a LATER attempt's success (or a retry can
        # simply happen not to re-trigger the same injection trigger) --
        # using only the latest trajectory's notifications would then
        # falsely report the component as "NOT injected" for a run it was
        # genuinely part of. Transcript/diff stay latest-only: those describe
        # one concrete attempt's content, not a set to union.
        notifications: list[str] = []
        diff_summary = ""
        traj_files = sorted(ex_log_dir.glob("trajectory_*.json"), key=_attempt_num)
        for traj_file in traj_files:
            try:
                traj_data = json.loads(traj_file.read_text())
                notifications.extend(
                    f"[{n.get('notifyType', 'info')}] {n.get('message', '')}"
                    for n in traj_data.get("notifications", [])
                )
            except (json.JSONDecodeError, OSError):
                pass
        if traj_files:
            latest = traj_files[-1]
            try:
                traj_data = json.loads(latest.read_text())
                full_assistant_text = traj_data.get("assistant_text") or ""
                transcript_excerpt = full_assistant_text[-_MAX_TRANSCRIPT_CHARS:]
                reasoning_excerpt = _reasoning_excerpt_from_trajectory(traj_data)
                summarized_transcript = summarize_for_reflection(
                    full_assistant_text, traj_data.get("tool_calls") or [],
                )
            except (json.JSONDecodeError, OSError):
                pass
            workdir = ex_log_dir / f"workdir_{_attempt_num(latest)}"
            if workdir.is_dir():
                diff_summary = self._compute_diff(spec, workdir)[:_MAX_DIFF_CHARS]

        return LiveRunResult(
            status=status, score=score, success=success,
            attempts=len(stop_reasons) if stop_reasons else (1 if status != "error" else 0),
            stop_reasons=stop_reasons, elapsed_s=record.get("elapsed_s", 0.0) or 0.0,
            turn_count=record.get("turn_count", 0) or 0, compaction_total=compaction_total,
            self_reported_lessons=self_reported_lessons,
            test_output_tail=test_output_tail, transcript_excerpt=transcript_excerpt,
            reasoning_excerpt=reasoning_excerpt, summarized_transcript=summarized_transcript,
            diff_summary=diff_summary, notifications=notifications,
            **base_kwargs,
        )

    def _compute_diff(self, spec: ExerciseSpec, workdir: Path) -> str:
        """Real diff between the agent's actual code and the pristine stub
        -- the single most useful thing a reflection LM can see, per the
        live-eval plan doc.

        Python-only: for any other --language this silently returns "" (the
        reflection LM just never sees a diff block for that exercise) rather
        than raising, since aider_polyglot.py itself supports other
        languages. Warn loudly so this gap is visible rather than a silent
        quality regression for non-Python runs."""
        if self.benchmark_root is None:
            return ""
        if spec.language != "python":
            logger.warning(
                "_compute_diff: no diff support for language %r (exercise %r) -- the "
                "reflective feedback for this run will be missing the code-change block.",
                spec.language, spec.exercise,
            )
            return ""
        pristine_dir = practice_dir(self.benchmark_root, spec.language) / spec.exercise
        if not pristine_dir.is_dir():
            return ""
        parts = []
        for py_file in sorted(workdir.glob("*.py")):
            if py_file.name.endswith("_test.py"):
                continue
            pristine_file = pristine_dir / py_file.name
            pristine_lines = pristine_file.read_text().splitlines(keepends=True) if pristine_file.exists() else []
            new_lines = py_file.read_text().splitlines(keepends=True)
            diff = difflib.unified_diff(
                pristine_lines, new_lines,
                fromfile=f"pristine/{py_file.name}", tofile=f"agent/{py_file.name}",
            )
            parts.append("".join(diff))
        return "\n".join(p for p in parts if p)
