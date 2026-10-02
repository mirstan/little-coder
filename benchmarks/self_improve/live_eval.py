"""LiveRunResult + PolyglotLiveRunner: materializes a GEPA candidate into a
scratch tree (a private git repo, see scratch_worktree.py), runs a real aider_polyglot.py exercise as a subprocess per
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
import tempfile
import time
import uuid
import weakref
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
from benchmarks.self_improve.live_cache import UNSCOREABLE_STATUSES, LiveResultCache
from benchmarks.self_improve.scratch_worktree import ScratchWorktree

logger = logging.getLogger(__name__)

#: Mirrors aider_polyglot.py's own _ATTEMPT_TIMEOUT_S_DEFAULT rather than
#: duplicating the literal: the two already drifted once (900 here vs 2700
#: there), which would have let the OUTER subprocess timeout below SIGKILL
#: an exercise before even one inner attempt's own budget expired --
#: turning a clean fail_timeout into an abrupt harness_error.
#:
#: Read by regex rather than `import benchmarks.aider_polyglot` because an
#: import would:
#: 1. execute that module's whole top level, including
#:    `CODEX_TIMEOUT_S = _positive_int_env("CODEX_TIMEOUT_S", 900)`, which
#:    raises SystemExit on a malformed CODEX_TIMEOUT_S this module never
#:    uses -- crashing even the subprocess-free --estimate-only path.
#: 2. read the SOURCE checkout at this process' import time, while the
#:    subprocess runs the WORKTREE's pinned base_commit copy; under an
#:    uncommitted local edit to aider_polyglot.py the two differ, and the
#:    outer timeout could come out shorter than the pinned copy's actual
#:    per-attempt budget.
#: PolyglotLiveRunner.__init__ therefore calls the reader against
#: `worktree.path`; _ATTEMPT_TIMEOUT_S_DEFAULT (the source-checkout read
#: below) is only the --estimate-only / no-worktree fallback.
_ATTEMPT_TIMEOUT_S_DEFAULT_RE = re.compile(r"(?m)^_ATTEMPT_TIMEOUT_S_DEFAULT = (\d+)$")
#: Last-resort fallback if even the source checkout's own aider_polyglot.py
#: can't be read/parsed (should not happen in a working checkout) -- a
#: budget ESTIMATE, not a correctness-critical read, so degrading to a
#: hardcoded value here is preferable to crashing an otherwise-healthy run.
_ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK = 2700


def _attempt_timeout_default_from_source(aider_polyglot_py_path: Path) -> int:
    """Regex-extract aider_polyglot.py's _ATTEMPT_TIMEOUT_S_DEFAULT
    constant from the given copy's file TEXT -- never imports/executes it
    (see the module-level comment above for why). Never raises.

    A benign reformat over there (`= 2_700`, a type annotation, a trailing
    comment) stops the regex matching, which would silently reintroduce
    exactly the drift this mechanism exists to prevent -- hence the warning
    on every fallback. Still doesn't raise: this feeds a budget estimate,
    not a correctness-critical read.

    UnicodeDecodeError is a ValueError, not an OSError, so a bare
    `except OSError` would let a non-UTF-8 file abort live_eval's own
    module import (this runs at module level)."""
    try:
        text = aider_polyglot_py_path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(
            "_attempt_timeout_default_from_source: could not read %s (%s) -- "
            "using hardcoded fallback %d", aider_polyglot_py_path, e,
            _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK,
        )
        return _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK
    m = _ATTEMPT_TIMEOUT_S_DEFAULT_RE.search(text)
    if m is None:
        logger.warning(
            "_attempt_timeout_default_from_source: _ATTEMPT_TIMEOUT_S_DEFAULT not "
            "found (or reformatted past what this regex matches) in %s -- using "
            "hardcoded fallback %d; the outer per-exercise timeout estimate may now "
            "be computed against a stale default", aider_polyglot_py_path,
            _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK,
        )
        return _ATTEMPT_TIMEOUT_S_HARDCODED_FALLBACK
    return int(m.group(1))


_ATTEMPT_TIMEOUT_S_DEFAULT = _attempt_timeout_default_from_source(
    Path(__file__).resolve().parent.parent / "aider_polyglot.py"
)


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _attempt_timeout_s(default: int = _ATTEMPT_TIMEOUT_S_DEFAULT) -> int:
    """Mirrors aider_polyglot.py's own _positive_int_env() validate-and-raise
    contract: a malformed or non-positive ATTEMPT_TIMEOUT_S crashes the
    subprocess at import time, so falling back to the default here would
    compute a budget estimate for an exercise that is going to blow up.

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


#: Directories skipped by _exercise_inputs_fingerprint: caches a local
#: pytest run can leave inside a benchmark checkout, which the child never
#: reads as an input.
_FINGERPRINT_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache"})


def _exercise_inputs_fingerprint(ex_dir: Path) -> str:
    """sha256 over one exercise directory's files (stub, tests, .meta,
    .docs): each file's relative path plus its bytes, a symlink's target
    rather than what it points at, and only the type of anything else (a
    FIFO is never opened). A missing directory hashes to a fixed marker, so
    the key still changes once the exercise appears."""
    hasher = hashlib.sha256()
    if not ex_dir.is_dir():
        hasher.update(b"missing")
        return hasher.hexdigest()
    for dirpath, dirnames, filenames in os.walk(ex_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in _FINGERPRINT_SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(ex_dir).as_posix().encode("utf-8", "surrogateescape")
            hasher.update(b"\0path\0" + rel)
            try:
                st = path.lstat()
                if path.is_symlink():
                    hasher.update(b"\0link\0" + os.fsencode(os.readlink(path)))
                elif path.is_file():
                    hasher.update(b"\0file\0" + path.read_bytes())
                else:
                    hasher.update(b"\0other\0" + str(st.st_mode).encode())
            except OSError as e:
                hasher.update(b"\0unreadable\0" + type(e).__name__.encode())
    return hasher.hexdigest()


#: In-place retries for an exercise whose run came back with any status in
#: live_cache.UNSCOREABLE_STATUSES, not only "harness_error" (a config
#: "error" is not retried; see _is_config_error) -- the name predates the
#: broader meaning; the tests still refer to it. Once they are spent,
#: "harness_error" raises LiveEvalHarnessError and "error"/"empty_response"
#: are returned scored 0.0 and kept out of live_cache (see
#: PolyglotLiveRunner.run_batch()). Each retry is a real run, charged to
#: LiveBudget.
HARNESS_ERROR_RETRIES = 2

#: record["reason"] prefixes for an "error" that retrying cannot change,
#: written by aider_polyglot.py's _run_exercise (missing exercise dir, unknown
#: agent) and by main()'s exception wrapper around it (the JS shared-deps
#: RuntimeError from _prepare_javascript, and the gated-scoring checks in
#: _python_preflight / _javascript_preflight). If that wording changes, such an
#: error is retried and then scored 0.0 like any other runtime error, which
#: wastes retries and hides the misconfiguration behind 0.0 scores.
_CONFIG_ERROR_REASON_PREFIXES = (
    "exercise not found at ",
    "unknown agent ",
    "RuntimeError: shared JS deps missing at ",
    "RuntimeError: scoring preflight",
)


def _is_config_error(result: "LiveRunResult") -> bool:
    return result.status == "error" and (result.error or "").startswith(_CONFIG_ERROR_REASON_PREFIXES)


class LiveEvalHarnessError(RuntimeError):
    """Raised by PolyglotLiveRunner.run_batch() when an exercise still comes
    back "harness_error" after HARNESS_ERROR_RETRIES in-place retries, or
    when an "error" is a config error that no retry can change (see
    _is_config_error). A persistent runtime "error"/"empty_response" never
    raises; it is scored 0.0.

    A harness failure (missing/malformed results file, outer subprocess
    timeout) says nothing about the candidate, so scoring it 0.0 would
    poison the search -- and EvaluationBatch.scores must stay index-aligned
    with the batch, so the exercise can't simply be dropped either. Mirrors
    live_budget.LiveEvalBudgetExceeded: raise rather than fabricate a score.
    Callers of gepa.optimize()/run_batch() (run_gepa.py) are expected to
    catch it alongside LiveEvalBudgetExceeded and persist partial results.
    Carries the last failing result as `.result`."""

    def __init__(self, message: str, result: "LiveRunResult | None" = None):
        super().__init__(message)
        self.result = result


#: Set to a fresh uuid4 hex in each live run's child environment (never in
#: os.environ). pi, its bash tool and the gated pytest all copy their
#: parent's environment, so whatever the run leaves behind carries it,
#: including processes in sessions of their own that a process-group kill
#: cannot reach. _sweep_tagged_processes finds them by it.
_RUN_TOKEN_ENV = "LITTLE_CODER_SELF_IMPROVE_RUN_TOKEN"
#: SIGTERM-to-SIGKILL grace for a run's process group: after a normal exit,
#: when only leftovers can be in it, and after the outer timeout.
_POST_RUN_GRACE_S = 2.0
_TIMEOUT_GRACE_S = 15.0
#: How long to wait for a SIGKILLed group to disappear before warning.
_KILL_CONFIRM_S = 5.0
#: Rescans after killing what a sweep found, bounding a respawning process.
_SWEEP_ROUNDS = 3
#: SIGTERM-to-SIGKILL grace for each process a sweep finds.
_SWEEP_GRACE_S = 1.0
_PS_TIMEOUT_S = 10
#: Linux reads /proc/<pid>/environ; elsewhere (macOS) `ps -E` prints it.
_USE_PROC = sys.platform.startswith("linux")


def _ps_environ_listing() -> bytes:
    """`ps` output with each process's environment after its command line.
    Bytes, not text: one process with a non-UTF-8 value must not disable
    the sweep. Raises on a failed or timed-out ps."""
    r = subprocess.run(
        ["ps", "-A", "-E", "-ww", "-o", "pid=", "-o", "command="],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=_PS_TIMEOUT_S,
    )
    if r.returncode != 0:
        raise OSError(f"ps exited {r.returncode}: {r.stderr[-300:]!r}")
    return r.stdout


def _tagged_pids_from_ps(listing: bytes, token: str) -> set[int]:
    """Pids whose `ps -E` line has the exact entry NAME=token, bounded by
    whitespace or the line's ends so a longer value does not match."""
    entry = re.compile(rb"(?:^|\s)" + re.escape(f"{_RUN_TOKEN_ENV}={token}".encode()) + rb"(?:\s|$)")
    pids = set()
    for line in listing.splitlines():
        head = line.split(None, 1)
        if head and head[0].isdigit() and entry.search(line):
            pids.add(int(head[0]))
    return pids


def _list_tagged_pids(token: str) -> set[int]:
    """Every process, other than this one, whose environment carries this
    run's token. Raises if the process list cannot be read at all.

    What it cannot see: a process that cleared or rewrote its environment,
    one owned by another user, on Linux a same-user process whose
    /proc/<pid>/environ the kernel refuses to read (non-dumpable or with
    changed credentials: ssh-agent, anything run through sudo, a setuid or
    file-capability binary), and on macOS any Apple platform binary
    (/bin/sh, /bin/bash, /bin/zsh, sleep, tail, perl and the rest), whose
    environment ps -E does not show. Unreadable processes are skipped
    silently."""
    if _USE_PROC:
        needle = f"{_RUN_TOKEN_ENV}={token}".encode()
        pids = set()
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/environ", "rb") as fh:
                    environ = fh.read()
            except OSError:  # gone, another user's, unreadable (see above), or a kernel thread
                continue
            if needle in environ.split(b"\0"):
                pids.add(int(name))
    else:
        pids = _tagged_pids_from_ps(_ps_environ_listing(), token)
    pids.discard(os.getpid())
    return pids


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _group_gone(pgid: int) -> bool:
    """macOS answers EPERM for a group whose only members are zombies."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return True
    return False


def _signal_quietly(send, target: int, sig: int) -> None:
    try:
        send(target, sig)
    except (ProcessLookupError, PermissionError):
        pass


_MAX_TAIL_CHARS = 4_000
_MAX_TRANSCRIPT_CHARS = 4_000
_MAX_DIFF_CHARS = 6_000
#: Mirrors aider_polyglot.py's _SAFE_READ_LIMIT.
_MAX_SNAPSHOT_FILE_BYTES = 4 << 20
#: Reasoning traces run long (one real bowling attempt filled the 200-entry
#: non_text_deltas cap in aider_polyglot.py's trajectory dump, nearly all
#: thinking_delta) -- truncated from the TAIL so reflection sees the
#: model's latest reasoning, not its opening thoughts.
_MAX_REASONING_CHARS = 4_000
_ATTEMPT_NUM_RE = re.compile(r"_(\d+)$")


def _reasoning_excerpt_from_trajectory(traj_data: Mapping) -> str:
    """Reconstructs the model's reasoning stream from non_text_deltas the
    same way rpc_client.py's own prompt_and_collect() builds assistant_text
    from text_delta: concatenate each thinking_delta's "delta" field in
    order. "thinking_delta" is confirmed to be pi's real event type here,
    against a live run with thinking=high.

    Entries can be non-dict: aider_polyglot.py's _dump_trajectory._clip
    falls back to a truncated JSON string for any single delta that
    serializes past TRAJECTORY_FIELD_CHARS."""
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
    while scores just look uniformly bad.

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
    #: {input_tokens, cache_read_tokens, output_tokens} summed across every
    #: attempt, from aider_polyglot.py's per-exercise "usage" (input_tokens
    #: includes the cached part -- see its _usage_tokens). An output only:
    #: never part of run_config or the cache key. None means "not recorded"
    #: (a results file or memo entry from before usage was carried, or a
    #: harness_error with no results), distinct from a genuine zero.
    usage: dict | None = None
    #: From aider_polyglot.py's gated scoring (POLYGLOT_RESTORE_TESTS):
    #: True when some attempt was forced to fail for changing files the
    #: scorer protects (tests, runner configuration). tamper_reasons also
    #: carries information-only findings, marked "info:". The paths in them
    #: are chosen by the agent.
    tests_tampered: bool = False
    tamper_reasons: list = field(default_factory=list)
    #: Same source: True when some attempt was forced to fail because a
    #: solution file broke the solution policy (not a regular file, or a
    #: tripwire hit). Not test tampering; the paths are chosen by the agent.
    solution_rejected: bool = False
    rejection_reasons: list = field(default_factory=list)
    error: str | None = None
    from_cache: bool = False
    exit_code: int | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "LiveRunResult":
        # Unknown keys are ignored rather than raising TypeError, for the same
        # reason live_cache.get() treats a malformed entry as a miss: a stray
        # key in a cache file must not kill an in-flight run. (Entries from a
        # build with different fields never get here: run_config hashes this
        # file, so they sit under another cfg_hash.)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class PolyglotLiveRunner:
    """Runs real aider_polyglot.py exercises, one subprocess per exercise,
    against a candidate materialized into a shared scratch tree (reused
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
        # Read once, here, never again by path: the tree is agent-writable
        # after any run, and run_batch() checks the mapping before its first
        # reset. See _pinned_bytes() for where each copy comes from.
        pinned_yaml = self._pinned_bytes(self.components_yaml)
        if pinned_yaml is None:
            raise ValueError(f"{self.components_yaml} is not a file at the pinned commit {worktree.base_commit}")
        self._component_mapping: dict = yaml.safe_load(pinned_yaml) or {}
        # write_components_back() takes a path, so it gets a private copy
        # of the pinned bytes outside the tree, removed with the runner.
        fd, copy_path = tempfile.mkstemp(prefix="components-", suffix=".yaml")
        with os.fdopen(fd, "wb") as f:
            f.write(pinned_yaml)
        self._pinned_components_yaml = Path(copy_path)
        weakref.finalize(self, _unlink_quietly, copy_path)
        #: aider_polyglot.py/rpc_client.py as the subprocess runs them (see
        #: run_config), from the base commit; a file the commit lacks is
        #: left out of the hash.
        self._worktree_executed_bytes = [
            data for data in (
                worktree.base_file_bytes("benchmarks/aider_polyglot.py"),
                worktree.base_file_bytes("benchmarks/rpc_client.py"),
            ) if data is not None
        ]
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
        # indefinitely. Default read from THIS worktree's pinned copy, not
        # the source checkout's: an uncommitted local edit makes them
        # diverge, and the source's value could yield an outer timeout
        # shorter than the pinned copy's own per-attempt budget.
        self._worktree_attempt_timeout_default = _attempt_timeout_default_from_source(
            worktree.path / "benchmarks" / "aider_polyglot.py"
        )
        self.per_exercise_timeout_s = per_exercise_timeout_s or (
            max_attempts * (_attempt_timeout_s(default=self._worktree_attempt_timeout_default) + 90) + 180
        )
        #: (benchmark root, language, exercise) -> _exercise_inputs_fingerprint,
        #: so each exercise directory is hashed once per runner.
        self._exercise_fingerprints: dict[tuple[str, str, str], str] = {}
        self.python_executable = python_executable
        #: Optional callable(LiveRunResult) -- invoked as EACH result becomes
        #: available inside run_batch() (cache hits included), not after the
        #: whole batch returns. A caller auditing every live invocation
        #: (run_gepa.py's SpendLog) would otherwise lose every
        #: already-completed result in a batch that raises partway through
        #: (e.g. LiveBudget's own backstop firing on exercise N of M).
        self.on_result = on_result

    def _pinned_bytes(self, path: Path) -> bytes | None:
        """`path`'s content as of construction. Inside the scratch tree:
        its content at base_commit, from the scratch repo's object store
        (ScratchWorktree.base_file_bytes), or None when the commit has no
        file there. Outside the tree (tests pass the source repo's copy):
        read by path, once, now."""
        try:
            rel = path.relative_to(self.worktree.path)
        except ValueError:
            return path.read_bytes()
        return self.worktree.base_file_bytes(rel.as_posix())

    @property
    def run_config(self) -> dict:
        """Everything besides the candidate text that changes what a score
        means. Includes the harness files' own sha256 so a mid-project
        harness bugfix can't let a stale cache entry poison a later
        comparison.

        Each file is hashed from whichever copy actually executes, which
        differs by file:

        aider_polyglot.py/rpc_client.py run as a SUBPROCESS inside the
        scratch worktree, so the worktree's pinned copy is hashed -- its
        content at base_commit, read once in __init__, since reset() puts
        exactly that back before every run while the tree itself is
        whatever the last run's agent left. Hashing
        the source repo instead would let an uncommitted debug edit or a
        branch switch there change the cache key without changing a byte of
        what executes, spuriously re-running already-cached work.

        aider_polyglot_ingest.py/components.py/this module run in the
        parent orchestrator process (imported at the top of this file), so
        the `__file__` this process imported is hashed. Hashing the
        worktree's copy would miss an uncommitted retune of e.g.
        _COMPACTION_PENALTY, _estimate_token_cost, or _parse_result's own
        scoring logic, and LiveResultCache would keep serving scores
        computed under the superseded formula."""
        parent_imported_files = [
            Path(_aider_polyglot_ingest_module.__file__),
            Path(_components_module.__file__),
            Path(__file__),
        ]
        hasher = hashlib.sha256()
        for data in self._worktree_executed_bytes:
            hasher.update(data)
        for f in parent_imported_files:
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
            # The child's own per-attempt budget, resolved the way it will
            # resolve it (ATTEMPT_TIMEOUT_S from the environment it inherits,
            # else the worktree copy's default). An explicit outer timeout
            # above leaves this free to change on its own.
            "attempt_timeout_s": _attempt_timeout_s(default=self._worktree_attempt_timeout_default),
            # The pi binary IS the agent under test -- omitting it means a
            # smoke run through fake_pi.py (LITTLE_CODER_PI_BIN_OVERRIDE) can
            # populate the cache with fabricated results that a later real
            # run, with an identical candidate/model/etc, would then read
            # back as genuine. mtime (not a full content hash) also catches
            # an `npm install` bumping the real pi binary mid-project.
            "pi_bin": str(pi_bin),
            "pi_bin_mtime": pi_bin_mtime,
            # The child runs under this interpreter, and its gated scoring
            # runs pytest under it too (aider_polyglot._run_python_gated).
            **self._interpreter_info(),
        }

    def _effective_benchmark_root(self) -> Path:
        """The root the child reads exercises from: benchmark_root when set,
        else the POLYGLOT_BENCHMARK_ROOT it inherits, else aider_polyglot.py's
        own default (its BENCHMARK_ROOT line)."""
        if self.benchmark_root:
            return self.benchmark_root
        inherited = self.worktree.env().get("POLYGLOT_BENCHMARK_ROOT")
        return Path(inherited) if inherited else Path.home() / "Documents" / "polyglot-benchmark"

    def exercise_run_config(self, spec: ExerciseSpec, run_config: Mapping | None = None) -> dict:
        """run_config plus a fingerprint of this exercise's benchmark inputs,
        the key run_batch() reads and writes live_cache under. Per exercise,
        so editing one exercise's tests leaves every other entry valid."""
        root = self._effective_benchmark_root()
        memo_key = (str(root), spec.language, spec.exercise)
        fingerprint = self._exercise_fingerprints.get(memo_key)
        if fingerprint is None:
            fingerprint = _exercise_inputs_fingerprint(practice_dir(root, spec.language) / spec.exercise)
            self._exercise_fingerprints[memo_key] = fingerprint
        return {**(self.run_config if run_config is None else run_config),
                "exercise_inputs_sha256": fingerprint}

    def _interpreter_info(self) -> dict:
        """python_executable plus its Python and pytest versions, probed
        once per runner. A failed probe records None rather than raising."""
        cached = getattr(self, "_interpreter_info_cache", None)
        if cached is None:
            python_version = pytest_version = None
            try:
                r = subprocess.run(
                    [self.python_executable, "-I", "-c",
                     "import sys, pytest; print(sys.version.split()[0], pytest.__version__)"],
                    capture_output=True, text=True, timeout=60,
                )
                parts = r.stdout.split()
                if r.returncode == 0 and len(parts) == 2:
                    python_version, pytest_version = parts
            except (OSError, subprocess.TimeoutExpired):
                pass
            cached = {"python_executable": str(self.python_executable),
                      "python_version": python_version, "pytest_version": pytest_version}
            self._interpreter_info_cache = cached
        return dict(cached)

    def materialize(self, candidate: Mapping[str, str]) -> list[Path]:
        """Reset the worktree to its pinned base commit, then write only the
        candidate's (sanitized) text into place, verifying nothing else in
        the tree changed."""
        self.worktree.reset()
        return self._write_sanitized(_sanitize_candidate(candidate))

    def _write_sanitized(self, sanitized: Mapping[str, str]) -> list[Path]:
        """materialize() minus the reset and the sanitize step --
        _sanitize_candidate strips only ONE leading block, so it must not be
        re-applied to text that is already sanitized (run_batch() sanitizes
        once and uses that same dict for both the cache key and the write)."""
        self._check_mapped(sanitized)
        changed = write_components_back(self._pinned_components_yaml, self.worktree.path, sanitized)
        self.worktree.assert_only_expected_dirty(changed)
        return changed

    def _prepare_clean_tree(self, sanitized: Mapping[str, str]) -> None:
        """Before every live run, retries included: back to the pinned base
        commit (reset() also verifies the checkout), then the candidate's
        text, so no run sees what an earlier run's agent left in the tree --
        except `node_modules` directories, which reset() keeps (`clean -e
        node_modules`). Nothing a later run executes reads them today: pi
        is the source checkout's (resolve_pi_bin) and the exercise works in
        a temp dir outside the tree (aider_polyglot.py)."""
        self.worktree.reset()
        self._write_sanitized(sanitized)

    def _check_mapped(self, sanitized: Mapping[str, str]) -> None:
        """Raises ValueError for a component the pinned components.yaml does
        not map (the mapping __init__ read, never the tree's copy)."""
        unmapped = sorted(set(sanitized) - set(self._component_mapping))
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

    def run_batch(
        self, candidate: Mapping[str, str], specs: Sequence[ExerciseSpec], *, sample_index: int = 0,
    ) -> list[LiveRunResult]:
        """Cache-first, reset-per-run: checks the on-disk memo for every
        requested exercise before touching the worktree at all, so a batch
        of cache hits never touches it. Each missed exercise then runs live,
        and before every run, retries included, the tree is reset to the
        base commit and the candidate written again (_prepare_clean_tree):
        an agent can change anything in the tree, and no run may be scored
        against what an earlier one left there (`node_modules` directories
        survive the reset; see _prepare_clean_tree). Returns results in the SAME order as
        `specs` (a hard requirement for the GEPA adapter built on top of
        this -- EvaluationBatch.scores must align index-for-index with the
        batch).

        The memo is keyed on the SANITIZED candidate, i.e. exactly what gets
        written, plus `sample_index` (which of k repeated samples this is).

        A run with any status in UNSCOREABLE_STATUSES is retried in place up
        to HARNESS_ERROR_RETRIES times, and is never written to live_cache.
        If it persists: "harness_error" raises LiveEvalHarnessError and is
        never returned as a scored result; "error"/"empty_response" are
        returned scored 0.0 with a WARNING log naming the status, reason and
        task_id. The environment or the candidate can cause them (e.g. a
        candidate overflowing the context window), and nothing here tries to
        tell which. Kept out of live_cache only: GEPA's own EvaluationCache
        (valset and minibatch put_batch) and its Pareto valset scores still
        record that 0.0 for the rest of this optimize() call, so a dead
        model server scores every exercise 0.0 until the LiveBudget
        wall-clock or run cap stops the run. A config "error" (see
        _is_config_error) raises without retrying."""
        run_config = self.run_config
        sanitized = _sanitize_candidate(candidate)
        results: dict[str, LiveRunResult] = {}
        misses: list[ExerciseSpec] = []
        for spec in specs:
            cached = (
                self.cache.get(sanitized, self.exercise_run_config(spec, run_config), spec.task_id,
                               sample_index=sample_index)
                if self.cache else None
            )
            if cached is not None:
                result = LiveRunResult.from_dict(cached)
                result.from_cache = True
                results[spec.task_id] = result
                if self.on_result is not None:
                    self.on_result(result)
            else:
                misses.append(spec)

        if misses:
            # Before any budget check, so an unmapped component still fails
            # fast instead of behind a budget refusal.
            self._check_mapped(sanitized)
            for spec in misses:
                result = self._run_with_retries(spec, sanitized)
                if result.status in UNSCOREABLE_STATUSES:
                    logger.warning(
                        "live_eval: %s still hit %s after %d retries (reason: %s) -- "
                        "scoring it 0.0, kept out of live_cache",
                        spec.task_id, result.status, HARNESS_ERROR_RETRIES,
                        result.error or "<no reason recorded>",
                    )
                results[spec.task_id] = result
                if self.cache is not None:
                    self.cache.put(sanitized, self.exercise_run_config(spec, run_config), spec.task_id,
                                   result.to_dict(), sample_index=sample_index)

        return [results[spec.task_id] for spec in specs]

    def _run_with_retries(self, spec: ExerciseSpec, sanitized: Mapping[str, str]) -> LiveRunResult:
        """Runs one exercise live, retrying an unscoreable status in place,
        each try in a freshly reset tree with `sanitized` written into it.
        Returns the first scoreable result, or the last unscoreable one once
        the retries are spent; raises for a config error or a persistent
        harness_error.

        The order within a try matters: the budget check comes first, so a
        refusal touches nothing; _run_one_uncached parses its result
        (reading log_root inside the tree) before returning, and the next
        try's reset deletes log_root with every other untracked file."""
        for _try in range(1 + HARNESS_ERROR_RETRIES):
            if self.budget is not None:
                self.budget.check_before_exercise(spec.task_id)  # raises rather than faking a score
            self._prepare_clean_tree(sanitized)
            result = self._run_one_uncached(spec)
            if self.budget is not None:
                self.budget.record_live_run()
            # Emitted here, per-run, rather than after the whole batch
            # returns -- a later exercise in this same batch raising
            # (e.g. the budget backstop above) must not erase the audit
            # trail for exercises that already genuinely ran. Fired for
            # an unscoreable run too: it genuinely ran and spent budget,
            # even when it is never scored. Fired BEFORE
            # run_batch()'s cache.put(): if the memo write itself raises
            # (disk I/O), the audit record for an exercise that DID
            # genuinely run must not be lost along with it.
            if self.on_result is not None:
                self.on_result(result)
            if result.status not in UNSCOREABLE_STATUSES:
                return result
            if _is_config_error(result):
                raise LiveEvalHarnessError(
                    f"{spec.task_id!r} hit a config error that no retry can change -- "
                    f"refusing to score it: {result.error}",
                    result=result,
                )
            logger.warning(
                "live_eval: %s hit %s (try %d/%d): %s",
                spec.task_id, result.status, _try + 1, 1 + HARNESS_ERROR_RETRIES, result.error,
            )
        if result.status == "harness_error":
            raise LiveEvalHarnessError(
                f"{spec.task_id!r} still hit harness_error after {HARNESS_ERROR_RETRIES} "
                f"retries -- refusing to score a harness failure: {result.error}",
                result=result,
            )
        return result

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
        # Off by default in aider_polyglot.py (it changes the retry prompt);
        # self-improve needs the LESSON: lines for self_reported_lessons.
        env["POLYGLOT_REQUEST_LESSONS"] = "1"
        # Also off by default there. A pi crash on a later attempt would
        # otherwise record "fail", which is scored and cached; and an agent
        # that edits its tests or adds a runner hook could score a pass
        # (RESTORE_TESTS scores a harness-built tree with verified reports).
        env["POLYGLOT_CRASH_IS_ERROR"] = "1"
        env["POLYGLOT_RESTORE_TESTS"] = "1"
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

        token = uuid.uuid4().hex
        env[_RUN_TOKEN_ENV] = token

        results_file.unlink(missing_ok=True)
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
        # Recorded (in the marker beside the scratch tree) so
        # gepa_scratch_gc.py can tell "the orchestrator died" apart from
        # "this exercise subprocess is still alive and using the scratch
        # tree" -- start_new_session=True means this child has its own
        # process group/pid and does NOT die when the orchestrator does.
        try:
            self.worktree.set_active_pid(proc.pid)
            try:
                _stdout, stderr = proc.communicate(timeout=effective_timeout)
                exit_code = proc.returncode
                # Before _parse_result: an agent's `cmd &` outlives the run
                # and could otherwise still be writing while it is scored.
                self._reap_leftovers(proc, token, grace_s=_POST_RUN_GRACE_S)
            except subprocess.TimeoutExpired:
                # Both before communicate(): a leftover can hold the pipes.
                self._kill_process_group(proc)
                self._sweep_tagged_processes(token)
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
            # otherwise keep driving a real paid rollout against a scratch
            # tree the caller may be about to remove. Calling this twice (the
            # budget_clamped raise above already killed it) is safe --
            # _kill_process_group treats an already-dead process as a no-op.
            self._kill_process_group(proc)
            try:
                self._sweep_tagged_processes(token)
            except Exception:
                logger.exception("live_eval: leftover-process sweep failed while unwinding")
            raise
        finally:
            self.worktree.set_active_pid(None)

        return self._parse_result(spec, results_file, log_root, stderr, exit_code)

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen, grace_s: float = _TIMEOUT_GRACE_S) -> bool:
        """A plain subprocess timeout only kills the DIRECT child --
        aider_polyglot.py's own comments document that a spawned bash
        grandchild (from the agent's own tool calls) can still outlive
        that. start_new_session=True (in _run_one_uncached) makes this
        process its own session AND process group leader, so its pgid is
        exactly proc.pid at creation time -- used directly here rather than
        looked up via os.getpgid(proc.pid), which raises ProcessLookupError
        once the direct child has exited even while its process group still
        has live descendants to reach.

        Escalation to SIGKILL is decided from the GROUP's own liveness, not
        proc.wait()'s return: if the direct child exits quickly but a
        descendant still holds the pipes open and ignores SIGTERM,
        proc.wait() returns immediately (only ever waiting on the direct
        child) -- naively treating that as "done" would skip SIGKILL
        entirely and leave the caller's subsequent communicate() call
        hanging forever on pipes that never close. The grace period is
        timed from the group's liveness too: after a normal exit the leader
        is already reaped, so waiting on it would end the grace at once.

        Refuses a pgid of 0 or 1 or this process's own group. A group whose
        only members are zombies (macOS answers EPERM for it) or another
        user's processes counts as done: nothing in it can be signalled.
        Never blocks past grace_s + _KILL_CONFIRM_S; warns if the group
        outlives that. Returns whether the group still existed, i.e.
        whether anything had to be signalled."""
        pgid = proc.pid
        if pgid <= 1 or pgid == os.getpgrp():
            logger.warning("live_eval: refusing to signal process group %d", pgid)
            return False
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False

        def _gone_within(seconds: float) -> bool:
            deadline = time.monotonic() + seconds
            while True:
                proc.poll()  # reaps the leader if it is a zombie
                try:
                    os.killpg(pgid, 0)
                except (ProcessLookupError, PermissionError):
                    return True
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.05)

        if _gone_within(grace_s):
            return True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return True
        if not _gone_within(_KILL_CONFIRM_S):
            logger.warning(
                "live_eval: process group %d still exists %.0fs after SIGKILL", pgid, _KILL_CONFIRM_S,
            )
        return True

    @staticmethod
    def _sweep_tagged_processes(token: str, _lister=None) -> list[int]:
        """Kills every process carrying this run's token (see
        _RUN_TOKEN_ENV), and every process group one of them is in: SIGTERM,
        up to _SWEEP_GRACE_S, then SIGKILL, then a rescan, for at most
        _SWEEP_ROUNDS rounds. Returns the tagged pids it signalled.

        The group kill is what reaches a leftover the listing cannot see. A
        non-interactive shell keeps its `&` jobs in its own process group,
        so on macOS, where ps -E hides the environment of Apple binaries, a
        /bin/bash loop sharing a group with a visible tagged process (python,
        node) still dies. It is safe because setpgid() cannot move a process
        into a group in another session, and every tagged process is in a
        session the run's child (start_new_session=True) or one of its
        descendants created. Group 0, group 1 and this process's own group
        are never signalled.

        Best effort, fail-open: if the process list cannot be read, logs one
        warning and kills nothing. Never raises except KeyboardInterrupt.
        Stands down if this process's own environment carries the token,
        which would make the orchestrator's other children targets.
        `_lister` replaces _list_tagged_pids (tests)."""
        lister = _lister or _list_tagged_pids
        if os.environ.get(_RUN_TOKEN_ENV) == token:
            logger.warning(
                "live_eval: this process's own environment carries %s for this run -- "
                "skipping the leftover-process sweep", _RUN_TOKEN_ENV,
            )
            return []
        killed: set[int] = set()
        try:
            for _round in range(_SWEEP_ROUNDS):
                try:
                    pids = set(lister(token)) - {os.getpid()}
                except Exception as e:
                    logger.warning(
                        "live_eval: could not list processes for the leftover-process sweep "
                        "(%s: %s) -- anything this run left behind is still running",
                        type(e).__name__, e,
                    )
                    break
                if not pids:
                    break
                own_group = os.getpgrp()
                pgids = set()
                for pid in pids:  # before any signal: a reaped pid has no group to ask about
                    try:
                        pgids.add(os.getpgid(pid))
                    except OSError:
                        pass
                pgids = {g for g in pgids if g > 1 and g != own_group}
                live_pids, live_pgids = set(pids), set(pgids)
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    for pid in live_pids:
                        _signal_quietly(os.kill, pid, sig)
                    for pgid in live_pgids:
                        _signal_quietly(os.killpg, pgid, sig)
                    deadline = time.monotonic() + _SWEEP_GRACE_S
                    while True:
                        live_pids = {p for p in live_pids if not _pid_gone(p)}
                        live_pgids = {g for g in live_pgids if not _group_gone(g)}
                        if not (live_pids or live_pgids) or time.monotonic() >= deadline:
                            break
                        time.sleep(0.05)
                    if not (live_pids or live_pgids):
                        break
                killed.update(pids)
            else:
                logger.warning(
                    "live_eval: tagged processes still appearing after %d sweep rounds", _SWEEP_ROUNDS,
                )
        except Exception as e:  # never let cleanup hide the run's own outcome
            logger.warning("live_eval: leftover-process sweep failed: %s: %s", type(e).__name__, e)
        if killed:
            logger.warning("live_eval: killed processes this run left behind: %s", sorted(killed))
        return sorted(killed)

    def _reap_leftovers(self, proc: subprocess.Popen, token: str, grace_s: float) -> None:
        """The run's process group, then everything carrying its token. Runs
        before _parse_result and before set_active_pid(None), so nothing the
        run started can still write into the tree while it is scored, and
        the GC treats the tree as busy until it is quiet."""
        if self._kill_process_group(proc, grace_s=grace_s):
            logger.warning("live_eval: the run's process group outlived its leader; killed it")
        self._sweep_tagged_processes(token)

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
        # Read as bytes so the locale's encoding does not matter (the child
        # writes ASCII JSON). ValueError covers JSONDecodeError and
        # UnicodeDecodeError; RecursionError is a deeply nested file. Each
        # used to escape run_batch()'s harness_error retry path.
        try:
            data = json.loads(results_file.read_bytes())
        except (ValueError, OSError, RecursionError) as e:
            return LiveRunResult(
                status="harness_error", score=0.0, success=False,
                error=f"malformed results file: {type(e).__name__}: {e}", **base_kwargs,
            )

        # Valid JSON of the wrong shape is a harness failure too: an
        # AttributeError here would escape run_batch()'s retry path.
        exercises = data.get("exercises", {}) if isinstance(data, dict) else None
        if not isinstance(exercises, dict):
            return LiveRunResult(
                status="harness_error", score=0.0, success=False,
                error=f"malformed results file: expected an object with an \"exercises\" object "
                      f"in {results_file}", **base_kwargs,
            )
        record = exercises.get(spec.results_key)
        if record is None:
            return LiveRunResult(
                status="harness_error", score=0.0, success=False,
                error=f"no record for {spec.results_key!r} in {results_file}", **base_kwargs,
            )
        status = record.get("status", "error") if isinstance(record, dict) else None
        if not isinstance(status, str):
            return LiveRunResult(
                status="harness_error", score=0.0, success=False,
                error=f"malformed record for {spec.results_key!r} in {results_file}: expected an "
                      "object with a string \"status\"", **base_kwargs,
            )

        compaction_total = record.get("compaction_total", 0) or 0
        success, score = pass_n_score(status, compaction_events=compaction_total) or (False, 0.0)
        stop_reasons = record.get("stop_reasons") or []
        # aider_polyglot.py caps each individual LESSON: line at
        # LESSON_MAX_CHARS (500) when extracting it, but that's per-attempt:
        # with --max-attempts set high, the joined text polyglot_adapter.py
        # inserts into GEPA reflection feedback is otherwise unbounded.
        # Capped at the same budget other reflection-bound fields use.
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

        def _clip_reasons(raw) -> list:
            return [r[:200] for r in raw if isinstance(r, str)][:10] if isinstance(raw, list) else []

        tamper_reasons = _clip_reasons(record.get("tamper_reasons"))
        rejection_reasons = _clip_reasons(record.get("rejection_reasons"))

        usage = None
        raw_usage = record.get("usage")
        if isinstance(raw_usage, dict):
            usage = {
                key: int(val) if isinstance(val := raw_usage.get(key, 0), (int, float)) else 0
                for key in ("input_tokens", "cache_read_tokens", "output_tokens")
            }

        ex_log_dir = log_root / "pi" / spec.language / spec.exercise
        test_output_tail = ""
        final_output = ex_log_dir / "final_output.txt"
        if final_output.exists():
            test_output_tail = final_output.read_text(errors="replace")[-_MAX_TAIL_CHARS:]

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
                traj_data = json.loads(traj_file.read_bytes())
                notifications.extend(
                    f"[{n.get('notifyType', 'info')}] {n.get('message', '')}"
                    for n in traj_data.get("notifications", [])
                )
            except (ValueError, OSError, RecursionError):
                pass
        if traj_files:
            latest = traj_files[-1]
            try:
                traj_data = json.loads(latest.read_bytes())
                full_assistant_text = traj_data.get("assistant_text") or ""
                transcript_excerpt = full_assistant_text[-_MAX_TRANSCRIPT_CHARS:]
                reasoning_excerpt = _reasoning_excerpt_from_trajectory(traj_data)
                summarized_transcript = summarize_for_reflection(
                    full_assistant_text, traj_data.get("tool_calls") or [],
                )
            except (ValueError, OSError, RecursionError):
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
            diff_summary=diff_summary, notifications=notifications, usage=usage,
            tests_tampered=record.get("tests_tampered") is True, tamper_reasons=tamper_reasons,
            solution_rejected=record.get("solution_rejected") is True,
            rejection_reasons=rejection_reasons,
            error=reason if isinstance(reason := record.get("reason"), str) else None, **base_kwargs,
        )

    def _compute_diff(self, spec: ExerciseSpec, workdir: Path) -> str:
        """Real diff between the agent's actual code and the pristine stub.

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
            # The snapshot keeps the agent's symlinks as symlinks; following
            # one (to /dev/zero, say) would hang this process.
            if py_file.is_symlink() or not py_file.is_file() or py_file.stat().st_size > _MAX_SNAPSHOT_FILE_BYTES:
                continue
            pristine_file = pristine_dir / py_file.name
            # errors="replace": the snapshot holds the agent's raw bytes, and
            # a UnicodeDecodeError here would escape after the paid run,
            # before run_batch() records it against the budget or reports it.
            pristine_lines = (
                pristine_file.read_text(errors="replace").splitlines(keepends=True)
                if pristine_file.exists() else []
            )
            new_lines = py_file.read_text(errors="replace").splitlines(keepends=True)
            diff = difflib.unified_diff(
                pristine_lines, new_lines,
                fromfile=f"pristine/{py_file.name}", tofile=f"agent/{py_file.name}",
            )
            parts.append("".join(diff))
        return "\n".join(p for p in parts if p)
