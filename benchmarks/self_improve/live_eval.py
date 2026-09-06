"""LiveRunResult + PolyglotLiveRunner: materializes a GEPA candidate into a
scratch worktree, runs a real aider_polyglot.py exercise as a subprocess per
exercise (not in-process -- see scratch_worktree.py's module docstring for
why), and parses the result into a graded score plus real feedback material
(diff, pytest output, transcript excerpt) for the reflection step.
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

from benchmarks.self_improve.components import write_components_back
from benchmarks.self_improve.exercises import ExerciseSpec, practice_dir
from benchmarks.self_improve.ingest.aider_polyglot_ingest import pass_n_score
from benchmarks.self_improve.live_cache import LiveResultCache
from benchmarks.self_improve.scratch_worktree import ScratchWorktree

logger = logging.getLogger(__name__)

_FRONTMATTER_DELIM = "---\n"
_MAX_TAIL_CHARS = 4_000
_MAX_TRANSCRIPT_CHARS = 4_000
_MAX_DIFF_CHARS = 6_000
_ATTEMPT_NUM_RE = re.compile(r"_(\d+)$")


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
    from planning -- see the live-eval plan doc."""
    if not text.startswith(_FRONTMATTER_DELIM):
        return text, False
    end = text.find("\n---\n", len(_FRONTMATTER_DELIM))
    if end == -1:
        return text, False
    return text[end + len("\n---\n"):], True


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
        # 900s/attempt (aider_polyglot's own ATTEMPT_TIMEOUT_S default) + 90s
        # test budget, times max_attempts, plus headroom -- a belt-and-braces
        # ceiling ABOVE aider_polyglot's own per-attempt budget so a wedged
        # pi session can't stall a run indefinitely.
        self.per_exercise_timeout_s = per_exercise_timeout_s or (max_attempts * (900 + 90) + 180)
        self.python_executable = python_executable

    @property
    def run_config(self) -> dict:
        """Everything besides the candidate text that changes what a score
        means. Includes the harness files' own sha256 so a mid-project
        harness bugfix can't let a stale cache entry poison a later
        comparison."""
        harness_files = [
            self.worktree.source_repo_root / "benchmarks" / "aider_polyglot.py",
            self.worktree.source_repo_root / "benchmarks" / "rpc_client.py",
        ]
        hasher = hashlib.sha256()
        for f in harness_files:
            if f.exists():
                hasher.update(f.read_bytes())
        return {
            "model": self.model,
            "language": self.language,
            "max_attempts": self.max_attempts,
            "retry": self.retry,
            "thinking": self.thinking,
            "base_commit": self.worktree.base_commit,
            "harness_hash": hasher.hexdigest(),
        }

    def materialize(self, candidate: Mapping[str, str]) -> list[Path]:
        """Reset the worktree to its pinned base commit, then write only the
        candidate's (sanitized) text into place, verifying nothing else in
        the tree changed."""
        self.worktree.reset()
        sanitized = _sanitize_candidate(candidate)
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
            else:
                misses.append(spec)

        if misses:
            self.materialize(candidate)
            for spec in misses:
                result = self._run_one_uncached(spec)
                results[spec.task_id] = result
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

        proc = subprocess.Popen(
            cmd, cwd=str(self.worktree.path), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            _stdout, stderr = proc.communicate(timeout=self.per_exercise_timeout_s)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            self._kill_process_group(proc)
            _stdout, stderr = proc.communicate()
            return LiveRunResult(
                task_id=spec.task_id, exercise=spec.exercise, language=spec.language,
                status="harness_error", score=0.0, success=False,
                error=f"subprocess timed out after {self.per_exercise_timeout_s}s",
            )

        return self._parse_result(spec, results_file, log_root, stderr, exit_code)

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        """A plain subprocess timeout only kills the DIRECT child --
        aider_polyglot.py's own comments document that a spawned bash
        grandchild (from the agent's own tool calls) can still outlive
        that. start_new_session=True (in _run_one_uncached) puts this
        subprocess in its own process group, so os.killpg reaches the
        whole tree."""
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
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
        success, score = pass_n_score(status) or (False, 0.0)
        stop_reasons = record.get("stop_reasons") or []

        ex_log_dir = log_root / "pi" / spec.language / spec.exercise
        test_output_tail = ""
        final_output = ex_log_dir / "final_output.txt"
        if final_output.exists():
            test_output_tail = final_output.read_text()[-_MAX_TAIL_CHARS:]

        transcript_excerpt = ""
        notifications: list[str] = []
        diff_summary = ""
        traj_files = sorted(ex_log_dir.glob("trajectory_*.json"), key=_attempt_num)
        if traj_files:
            latest = traj_files[-1]
            try:
                traj_data = json.loads(latest.read_text())
                transcript_excerpt = (traj_data.get("assistant_text") or "")[-_MAX_TRANSCRIPT_CHARS:]
                notifications = [
                    f"[{n.get('notifyType', 'info')}] {n.get('message', '')}"
                    for n in traj_data.get("notifications", [])
                ]
            except (json.JSONDecodeError, OSError):
                pass
            workdir = ex_log_dir / f"workdir_{_attempt_num(latest)}"
            if workdir.is_dir():
                diff_summary = self._compute_diff(spec, workdir)[:_MAX_DIFF_CHARS]

        return LiveRunResult(
            status=status, score=score, success=success,
            attempts=len(stop_reasons) if stop_reasons else (1 if status != "error" else 0),
            stop_reasons=stop_reasons, elapsed_s=record.get("elapsed_s", 0.0) or 0.0,
            turn_count=record.get("turn_count", 0) or 0,
            test_output_tail=test_output_tail, transcript_excerpt=transcript_excerpt,
            diff_summary=diff_summary, notifications=notifications,
            **base_kwargs,
        )

    def _compute_diff(self, spec: ExerciseSpec, workdir: Path) -> str:
        """Real diff between the agent's actual code and the pristine stub
        -- the single most useful thing a reflection LM can see, per the
        live-eval plan doc."""
        if self.benchmark_root is None:
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
