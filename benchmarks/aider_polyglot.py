#!/usr/bin/env python3
"""Aider Polyglot runner for little-coder.

Drives `pi --mode rpc` per exercise via benchmarks/rpc_client.py::PiRpc, or
`codex exec` per exercise via a plain subprocess call -- --agent selects
which. Per-language transforms (xit-strip, @Disabled-strip, cpp CMakeLists
named dirs, cargo --include-ignored, EXERCISM_RUN_ALL_TESTS) are copied
verbatim from little-coder's original aider_polyglot.py — the only real
change is that the agent call site uses PiRpc instead of agent.run().

Usage:
    python benchmarks/aider_polyglot.py              # full run, default model
    python benchmarks/aider_polyglot.py --language python
    python benchmarks/aider_polyglot.py --exercise hello-world --language python
    python benchmarks/aider_polyglot.py --model llamacpp/qwen3.5-9b
    python benchmarks/aider_polyglot.py --agent codex --model gpt-5.6-luna
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import uuid
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rpc_client import PiRpc, PromptResult, capture_environment_snapshot  # noqa: E402

# `or`, not os.environ.get(name, default): an exported-but-empty override
# must still fall back -- Path("") normalizes to Path("."), whose .exists()
# is True, silently pointing at the wrong directory instead of the intended
# default (same trap documented for LITTLE_CODER_PI_BIN_OVERRIDE in
# rpc_client.py). These three exist so a live-eval test harness can point
# an isolated subprocess invocation of this script at a synthetic benchmark
# root/results file/log dir, without which none of that machinery is
# testable without a real paid model run (BENCHMARK_ROOT feeds
# LANG_DESCRIPTORS below at IMPORT time).
BENCHMARK_ROOT = Path(os.environ.get("POLYGLOT_BENCHMARK_ROOT") or (Path.home() / "Documents" / "polyglot-benchmark"))
REPO_ROOT = Path(__file__).parent.parent
RESULTS_FILE = Path(os.environ.get("POLYGLOT_RESULTS_FILE") or (Path(__file__).parent / "results_full_polyglot.json"))
LOG_ROOT = Path(os.environ.get("POLYGLOT_LOG_ROOT") or (Path(__file__).parent / "full_polyglot_logs"))
#: Identifies one invocation, so records and artifacts can be traced back to the
#: run that produced them. RESULTS_FILE and LOG_ROOT are both deterministic and
#: shared across runs, which has already caused artifacts from different runs to
#: be compared as if they came from one.
RUN_ID = f"{datetime.datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
#: Caps applied to persisted trajectories -- see _dump_trajectory.
TRAJECTORY_TEXT_CHARS = 200_000
TRAJECTORY_FIELD_CHARS = 20_000
#: Give up if this many exercises fail in a row -- a broken environment,
#: not broken exercises.
MAX_CONSECUTIVE_ERRORS = 3
def _positive_int_env(name: str, default: int) -> int:
    """Parse a positive-integer env var, failing with a readable message.

    A bare ValueError traceback (ATTEMPT_TIMEOUT_S=30m) is unhelpful, and a
    silently-accepted 0 is worse: _drain_events_until returns immediately with
    no events, so every exercise would record fail_timeout with a 0-turn
    trajectory.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name}: expected a positive integer (seconds), got {raw!r}")
    if value <= 0:
        raise SystemExit(f"{name}: must be > 0, got {value}")
    return value


#: Per-attempt RPC budget, seconds. 900 suited a fast hosted model, but a
#: local model with a genuine large thinking budget (PR #17 fixed
#: omlx/rapidmlx's thinking_budget profile, previously silently stuck at
#: 4096) needs real headroom -- wordy/transpose already hit 691-722s even
#: at that broken smaller budget, and GAIA runs under the real 32768 budget
#: showed single completions taking 200-900+s. Tripled to 2700, matching
#: the same 3x used for GAIA's --timeout, so a hard multi-turn exercise is
#: capability-limited rather than clock-limited.
ATTEMPT_TIMEOUT_S = _positive_int_env("ATTEMPT_TIMEOUT_S", 2700)
#: Per-attempt budget for `codex exec`, seconds.
CODEX_TIMEOUT_S = _positive_int_env("CODEX_TIMEOUT_S", 900)
DEFAULT_MODEL = "llamacpp/qwen3.6-35b-a3b"

# Allowed tools for Polyglot — the core filesystem + bash toolbox. Ports
# the whitepaper's Polyglot configuration (no TB-style ShellSession, no
# GAIA-style Browser/Evidence).
ALLOWED_TOOLS = [
    "read", "Read",
    "write", "Write",
    "edit", "Edit",
    "bash", "Bash",
    "glob", "Glob",
    "grep", "Grep",
    "webfetch", "WebFetch",
]


# ── Per-language descriptors ──────────────────────────────────────────────
#
# Keep the structure identical to little-coder's aider_polyglot.py so the
# transforms (smoke-tested over the full 225-exercise run that produced the
# 78.67% headline) carry over unchanged.

def _copy_exercise(src: Path, dst: Path):
    """Copy exercise tree, excluding .meta/ (solutions live there)."""
    def _ignore(_dir, names):
        return [".meta"] if ".meta" in names else []
    shutil.copytree(src, dst, ignore=_ignore)


def _prepare_python(src: Path, work: Path):
    _copy_exercise(src, work)
    stubs = list(work.glob("*.py"))
    stubs = [p for p in stubs if not p.name.endswith("_test.py")]
    tests = list(work.glob("*_test.py"))
    return stubs, tests


def _run_python(work: Path, timeout: int):
    try:
        r = subprocess.run(
            ["python3", "-m", "pytest", "-x", "-q"],
            cwd=work, capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"


LANG_DESCRIPTORS = {
    "python": {
        "score_in_copy": True,
        "practice_dir": BENCHMARK_ROOT / "python" / "exercises" / "practice",
        "prepare": _prepare_python,
        "run_tests": _run_python,
        "syntax_hint": "Use Python 3. Run tests with `python -m pytest -x -q`.",
        "timeout_s": 90,
    },
    # go/rust/cpp/javascript/java descriptors omitted from this scaffold;
    # copy them verbatim from the Python repo's aider_polyglot.py when
    # running the full benchmark. Stub:
    # "rust":      {..., "prepare": _prepare_rust, ...},
    # "go":        {..., "prepare": _prepare_go, ...},
    # "cpp":       {..., "prepare": _prepare_cpp, ...},
    # "javascript": {..., "prepare": _prepare_js, ...},
    # "java":      {..., "prepare": _prepare_java, ...},
}


# ── Result file helpers ────────────────────────────────────────────────────

def _load_results() -> dict:
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text())
        except Exception:
            pass
    return {"exercises": {}, "meta": {}}


def _save_results(data: dict):
    tmp = RESULTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(RESULTS_FILE)


# ── Core loop ──────────────────────────────────────────────────────────────


def _purge_log_dir(log_dir: Path):
    """Remove artifacts from previous runs of this exercise.

    LOG_ROOT/<lang>/<exercise> is deterministic and nothing cleaned it, so a
    one-attempt rerun left the previous run's trajectory_2/workdir_2 sitting
    next to a fresh trajectory_1. Those pairs look like one run and are not:
    comparing them produced a confident, wrong conclusion during review.
    """
    for pattern in ("trajectory_*", "workdir_*", "final_output*"):
        for path in log_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except OSError:
                    pass


def _clip(value, limit: int):
    """Bound a field before it is serialised into the trajectory."""
    if isinstance(value, str):
        return value[:limit]
    if value is None:
        return value
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit]


def _dump_trajectory(log_dir, attempt_name, result, work=None, notifications=None):
    """Persist what the harness otherwise discards.

    Only the final pytest output survives a run today, so a failure can be seen
    but not explained. PromptResult already carries the assistant text and every
    tool call; the work dir is a TemporaryDirectory destroyed on scope exit,
    taking the model's actual code with it. `notifications` (ctx.ui.notify
    events -- skill/knowledge injections, and critically the thinking-budget
    extension's "the model has thought long enough" harness intervention) was
    the other thing this dropped: an attempt that read files and stopped
    without writing anything looked identical whether the model chose to stop
    or the harness force-aborted its thinking, unless you happened to re-run
    it live with rpc.notifications() to compare (this is exactly how the
    thinking-budget intervention was first discovered, on a real `bowling`
    failure -- nothing in the persisted trajectory said so).

    notifications: this attempt's OWN rpc.notifications(). Each attempt opens
    a fresh PiRpc session (see _run_exercise), so this is already scoped to
    just this attempt -- no delta-slicing across attempts needed or possible
    anymore. Extension activity (skill-inject, knowledge-inject,
    quality-monitor, etc.) otherwise invisible in this dump.

    Writes per attempt: trajectory_<n>.json, trajectory_<n>.txt, workdir_<n>/.
    """
    try:
        payload = {
            "attempt": attempt_name,
            "agent_ended": getattr(result, "agent_ended", None),
            "turn_count": getattr(result, "turn_count", None),
            "compaction_events": getattr(result, "compaction_events", None),
            "assistant_text": (getattr(result, "assistant_text", "") or "")[:TRAJECTORY_TEXT_CHARS],
            # write/Write args carry whole file bodies and bash results whole
            # command output; uncapped this reaches hundreds of MB per exercise.
            "tool_calls": [
                {
                    **tc,
                    "args": _clip(tc.get("args"), TRAJECTORY_FIELD_CHARS),
                    "result_text": _clip(tc.get("result_text"), TRAJECTORY_FIELD_CHARS),
                }
                for tc in getattr(result, "tool_calls", [])
            ],
            # Same policy as tool_calls above -- extension notify text is
            # normally short, but it's authored by extensions (including
            # third-party ones a run may load), and this is otherwise the
            # one field in this payload exempt from the file's own size cap.
            "notifications": [
                {**n, "message": _clip(n.get("message"), TRAJECTORY_FIELD_CHARS)}
                for n in (notifications or [])
            ],
        }
        (log_dir / f"trajectory_{attempt_name}.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8", errors="replace")
        lines = []
        for i, tc in enumerate(payload["tool_calls"], 1):
            lines.append(f"--- [{i}] {tc.get('name')} err={tc.get('is_error')} ---")
            lines.append(f"    args: {json.dumps(tc.get('args', {}), default=str)[:1500]}")
            lines.append(f"    result: {str(tc.get('result_text', ''))[:1500]}")
        if payload["notifications"]:
            lines.append("=== notifications ===")
            for n in payload["notifications"]:
                lines.append(f"    [{n.get('notifyType')}] {str(n.get('message', ''))[:1500]}")
        lines.append("=== assistant text ===")
        lines.append(payload["assistant_text"][:20000])
        (log_dir / f"trajectory_{attempt_name}.txt").write_text(
            "\n".join(lines), encoding="utf-8", errors="replace")
        if work is not None and Path(work).exists():
            snap = log_dir / f"workdir_{attempt_name}"
            if snap.exists():
                shutil.rmtree(snap, ignore_errors=True)
            # A full 225-exercise run copies go/rust/cpp/js/java trees too;
            # without this a snapshot pulls in target/, build/, node_modules/
            # and compiled binaries, twice per exercise.
            ignore = shutil.ignore_patterns(
                "__pycache__", ".pytest_cache", "target", "build", "node_modules",
                ".gradle", "CMakeFiles", "*.o", "*.so", "*.class", "*.rlib")
            # A file can vanish mid-walk while the agent is still active. One
            # retry beats losing the snapshot to the blanket except below,
            # which would leave only a .ERROR file for the very case the
            # snapshot exists to explain.
            for attempt_i in range(2):
                try:
                    shutil.copytree(work, snap, ignore=ignore)
                    break
                except FileNotFoundError:
                    shutil.rmtree(snap, ignore_errors=True)
                    if attempt_i == 1:
                        raise
                    time.sleep(0.5)
    except Exception as exc:
        # Diagnostics must never fail the exercise they are diagnosing.
        try:
            (log_dir / f"trajectory_{attempt_name}.ERROR").write_text(repr(exc))
        except Exception:
            pass


def _build_prompt(exercise_name: str, stub_paths, test_paths, syntax_hint: str) -> str:
    stubs_list = "\n".join(f"  - {p}" for p in stub_paths)
    tests_list = "\n".join(f"  - {p}" for p in test_paths)
    return (
        f"Implement the Exercism exercise `{exercise_name}`.\n\n"
        f"Stub file(s) to implement:\n{stubs_list}\n\n"
        f"Test file(s) (for reference only — DO NOT edit):\n{tests_list}\n\n"
        f"{syntax_hint}\n\n"
        "Read the stubs + any `.docs/instructions.md` in the workspace, "
        "then implement the solution. When you believe the code is correct, "
        "stop calling tools."
    )



#: Environment knobs that change how the agent behaves, and therefore what a
#: pass/fail means. Recorded so a result set is self-describing.
_ENV_KNOBS = (
    "PI_REASONING_MAX_TOKENS", "LITTLE_CODER_BASH_ALLOW",
    "LITTLE_CODER_PERMISSION_MODE", "LITTLE_CODER_INJECT_MODE",
    "LITTLE_CODER_SUBCODER_CONCURRENCY", "LITTLE_CODER_COMPACT_AT_PERCENT",
    # Now that --thinking is a first-class scoring parameter: the
    # thinking-budget extension can force the level to "off" mid-attempt
    # once this budget is exceeded, so a run recorded at a high --thinking
    # level may spend much of its budget at "off" in practice. This knob
    # directly gates that, and PI_REASONING_MAX_TOKENS (already listed
    # above) is the analogous budget for the reasoning-token cap itself.
    "LITTLE_CODER_THINKING_BUDGET",
)


def _scoring_params(model: str, language: str, retry: bool, desc: dict, *,
                    max_attempts: int = 2, thinking: str | None = None,
                    agent: str = "pi") -> dict:
    """Everything that affects whether an exercise passes.

    Absent from results until now, so `--resume` could silently blend runs made
    under different budgets into one file that looked homogeneous.
    ATTEMPT_TIMEOUT_S/CODEX_TIMEOUT_S are module constants read from the
    environment at import, not argparse fields, so they are captured here
    rather than from `args`. Recorded per-agent, not unconditionally as pi's
    values: allowed_tools and the pi-only env knobs describe nothing on the
    codex path, and codex is budgeted by CODEX_TIMEOUT_S, not
    ATTEMPT_TIMEOUT_S -- recording the wrong one would make _param_mismatches
    blind to exactly the kind of change (a different per-attempt budget)
    this function exists to catch.
    """
    return {
        "agent": agent,
        "model": model,
        "language": language,
        "attempt_timeout_s": ATTEMPT_TIMEOUT_S if agent == "pi" else CODEX_TIMEOUT_S,
        "test_timeout_s": desc.get("timeout_s"),
        "retry": retry,
        "max_attempts": max_attempts if retry else 1,
        "thinking": thinking,
        "score_in_copy": bool(desc.get("score_in_copy")),
        "allowed_tools": sorted(set(ALLOWED_TOOLS)) if agent == "pi" else None,
        "env": {k: os.environ[k] for k in _ENV_KNOBS if k in os.environ} if agent == "pi" else {},
    }


def _param_mismatches(recorded: dict, current: dict) -> list[str]:
    """Which scoring parameters differ between a stored run and this one."""
    if not recorded:
        return ["<no parameters recorded>"]
    out = []
    for key in sorted(set(recorded) | set(current)):
        was, now = recorded.get(key), current.get(key)
        if was != now:
            out.append(f"{key}: {was!r} -> {now!r}")
    return out


def _exit_code(records_written: dict) -> int:
    """Non-zero when THIS invocation recorded a harness error.

    Scoped to what this run wrote: a resumed run must not fail for an error it
    inherited from an earlier file and never touched.
    """
    return 1 if any(str(r.get("status")) == "error" for r in records_written.values()) else 0


def _stop_reason(result) -> str:
    """Why an attempt ended: agent_end | deadline | process_exit.

    Shim, deliberately: if rpc_client predates PromptResult.stop_reason (or that
    change is reverted -- it has been once already), fall back to the old signal
    so scoring degrades instead of breaking.
    """
    reason = getattr(result, "stop_reason", None)
    if reason:
        return reason
    return "agent_end" if getattr(result, "agent_ended", False) else "deadline"


def _is_empty_response(result) -> bool:
    """pi finished the turn without doing anything.

    An empty completion from the provider: agent_end arrives, but no tool call,
    no turn of work and no assistant text. Six of sixteen recorded attempts in
    this repo's log tree look like this. Classified as agent_end they read as
    clean failures, hiding a provider-side fault behind a model-quality number.
    """
    return (
        getattr(result, "agent_ended", False)
        and (getattr(result, "turn_count", 0) or 0) <= 1
        and not getattr(result, "tool_calls", None)
        and not (getattr(result, "assistant_text", "") or "").strip()
    )


def _attempt_outcome(result) -> str:
    """One attempt's outcome, independent of whether the tests passed."""
    reason = _stop_reason(result)
    if reason in ("process_exit", "deadline"):
        return reason
    if _is_empty_response(result):
        return "empty_response"
    return "completed"


def _classify_status(passed: bool, attempt: str | None, outcomes: list[str]) -> str:
    """Precedence for the recorded status. Pure, so it can be table-tested.

    `passed` wins over everything: a pi that exits right after writing a
    correct solution must not be downgraded from a pass to an error.
    Generalized from a hardcoded two attempts to however many `outcomes`
    actually ran, so --max-attempts can be raised without touching this
    function.
    """
    if passed:
        return attempt or "pass_1"
    if not outcomes:
        return "error"
    last = outcomes[-1]
    if last == "process_exit" and len(outcomes) == 1:
        # Only one attempt ran, so a dead process means the run itself
        # failed, not that the exercise was genuinely attempted and failed.
        return "error"
    if last == "deadline":
        return "fail_timeout"
    if last == "empty_response":
        return "empty_response"
    return "fail"


def _score(desc, work: Path, timeout: int):
    """Run the tests against a COPY, so a live agent cannot influence the score.

    prompt_and_collect returning does not mean pi is finished: measured on three
    real exercises, pi still had a live child process at the moment the tests
    would start (the process is now closed by the `with PiRpc(...)` block before
    this runs, but a spawned bash grandchild can still outlive that close()).
    Scoring a copy avoids that leftover process's writes landing mid-score, and
    stops the test runner's own droppings landing in the agent's tree. Not
    enabled for cpp/rust, whose build dirs are absolute-path-bound and would
    force a full rebuild per scoring pass.
    """
    if not desc.get("score_in_copy"):
        return desc["run_tests"](work, timeout)
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / work.name
        shutil.copytree(work, target)
        return desc["run_tests"](target, timeout)


def _run_codex_turn(
    model: str, work: Path, prompt: str, session_id: str | None, log_dir: Path, attempt_name: str,
) -> tuple[PromptResult, str | None]:
    """Drive one codex exec turn. Returns (result, session_id_to_resume_next).

    Sandbox: workspace-write + --approve-for-me -- auto-approves within the
    workspace, blocks writes outside it and (per Codex's OS-level sandbox)
    outbound network by default. This is a categorically different
    restriction from little-coder's shell-command whitelist -- document,
    don't conflate, when comparing failures against the pi agent.

    session_id is None for a fresh attempt 1 and the id returned by the
    previous call for a retry. `codex exec resume` is a DIFFERENT subcommand
    from `codex exec` with a different, smaller flag set: --approve-for-me
    and --cd are rejected outright (confirmed against codex-cli 0.153.0:
    "unexpected argument '--approve-for-me' found"), and there is no `--last`
    used here either, since --last is a global "newest session, filtered by
    the *process's* cwd" selector over ~/.codex/sessions/ -- shared across
    every codex invocation on the machine, so it can silently resume a
    DIFFERENT exercise's session (if this one's attempt 1 never recorded a
    session) or a concurrent/earlier run's. Explicit SESSION_ID (captured
    from attempt 1's own `thread.started` event) avoids that ambiguity
    entirely. Verified empirically that resuming by SESSION_ID needs neither
    --approve-for-me nor --cd: the sandbox/approval policy and working
    directory both carry over from the session being resumed.

    Returns a PromptResult (not a codex-specific type) so the rest of
    _run_exercise's loop -- _attempt_outcome, _stop_reason, _dump_trajectory
    -- works unchanged for both agents. codex has no equivalent of pi's
    per-tool-call introspection, so tool_calls is always empty; turn_count
    is a coarser unit than pi's (see below).
    """
    # Under log_dir, not work: work is the sandbox's writable root, so a file
    # there is visible to the model (who could delete or overwrite it) and
    # gets copied into every _score() snapshot. A fixed name under `work`
    # also went stale across attempts when resume never actually ran (the
    # bug this whole rewrite fixes) -- attempt 2's read would silently pick
    # up attempt 1's leftover file. Naming it per-attempt removes that hazard
    # even now that resume is fixed.
    out_file = log_dir / f"codex_last_message_{attempt_name}.txt"

    if session_id is None:
        args = [
            "codex", "exec",
            "--approve-for-me",  # auto-approves within workspace-write sandbox (mutually
                                  # exclusive with --sandbox; --approve-for-me already implies it)
            "--skip-git-repo-check",
            "--json",
            "--model", model,
            "--cd", str(work),
            "-o", str(out_file),
            "--",
            prompt,
        ]
    else:
        args = [
            "codex", "exec", "resume", session_id,
            "--skip-git-repo-check",
            "--json",
            "-o", str(out_file),
            "--",
            prompt,
        ]

    # cwd=work is passed explicitly (not relied on via --cd, which `resume`
    # doesn't accept) so a retry's relative-path operations resolve inside
    # the exercise's own tree regardless of where this harness process
    # itself was launched from.
    #
    # start_new_session=True + killing the whole process group on timeout,
    # not just the direct child: codex's own sandboxed subprocesses can
    # otherwise survive a plain kill() and keep writing into `work` while
    # _score() reads that tree right after.
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        stdin=subprocess.DEVNULL, cwd=str(work), start_new_session=True,
    )
    try:
        # stdin MUST be closed/redirected: codex exec reads an appended <stdin> block
        # whenever stdin isn't already closed, even when a prompt is given as an
        # argument (per its own --help) -- inherited-but-open stdin from a parent
        # shell hangs forever waiting for EOF. Confirmed by direct testing: a run
        # with inherited stdin sat at 0% CPU / zero network connections for 17+
        # minutes on a trivial task, blocked in a stdin read, not "thinking."
        stdout, stderr = proc.communicate(timeout=CODEX_TIMEOUT_S)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        returncode = None

    if stderr:
        (log_dir / f"codex_stderr_{attempt_name}.txt").write_text(
            stderr, encoding="utf-8", errors="replace")

    # NOTE on turn_count comparability: a `turn.completed` event fires once per
    # `codex exec` invocation regardless of how many tool calls happened inside
    # it (confirmed empirically: a 3-tool-call edit task produced exactly one
    # turn.completed) -- this is a coarser unit than pi's turn_count, which
    # increments per tool-call round. Counting actionable item.completed events
    # instead (command_execution / file_change / mcp_tool_call / patch_apply) is
    # a much closer analog to what pi's turn_count actually measures: distinct
    # steps taken.
    action_types = {"command_execution", "file_change", "mcp_tool_call", "patch_apply"}
    action_count = 0
    new_session_id = None
    turn_failed = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("type")
        if ev_type == "thread.started":
            new_session_id = ev.get("thread_id") or new_session_id
        elif ev_type == "turn.failed":
            # A turn that errors mid-flight (stream error, context overflow,
            # moderation, or -- before this fix -- an argv the "resume"
            # subcommand rejects outright) must not be recorded as a clean
            # completion: _attempt_outcome/_classify_status treat
            # "completed" as license to retry with a fresh prompt, and an
            # error that never touched the model has nothing to retry from.
            turn_failed = True
        elif ev_type == "item.completed":
            item = ev.get("item") or {}
            if isinstance(item, dict) and item.get("type") in action_types:
                action_count += 1

    # returncode is None only on the timeout path above.
    if returncode is None:
        return (
            PromptResult(turn_count=action_count, agent_ended=False, stop_reason="deadline"),
            new_session_id or session_id,
        )
    if returncode != 0 or turn_failed:
        # Checked, not ignored: a non-zero exit (argv rejected, auth
        # failure, invalid model, HTTP error) used to be indistinguishable
        # from a clean turn that did nothing -- stderr above is what
        # actually explains it.
        return (
            PromptResult(turn_count=action_count, agent_ended=False, stop_reason="process_exit"),
            new_session_id or session_id,
        )

    assistant_text = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else ""
    return (
        PromptResult(
            turn_count=max(action_count, 1),
            agent_ended=True,
            stop_reason="agent_end",
            assistant_text=assistant_text,
        ),
        new_session_id or session_id,
    )


def _run_exercise(
    lang: str,
    ex_name: str,
    model: str,
    verbose: bool,
    retry: bool,
    *,
    max_attempts: int = 2,
    thinking: str | None = None,
    agent: str = "pi",
    thinking_confirmation: dict | None = None,
):
    desc = LANG_DESCRIPTORS.get(lang)
    if desc is None:
        return {"status": "skipped", "reason": f"no descriptor for language {lang}"}

    src = desc["practice_dir"] / ex_name
    if not src.exists():
        return {"status": "error", "reason": f"exercise not found at {src}"}

    # Namespaced by agent: two agents run against the same exercise names,
    # and an un-namespaced log_dir let a later agent's run silently
    # overwrite an earlier agent's raw diagnostic text (trajectory/workdir/
    # final_output artifacts). The scored JSON results key is namespaced by
    # agent too (see main()) for the same reason -- otherwise --resume
    # against a pi results file with --agent codex would skip every
    # exercise pi had already passed and overwrite the rest, producing one
    # results file that silently blends the two agents' scores.
    log_dir = LOG_ROOT / agent / lang / ex_name
    log_dir.mkdir(parents=True, exist_ok=True)
    _purge_log_dir(log_dir)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / ex_name
        stubs, tests = desc["prepare"](src, work)
        prompt = _build_prompt(ex_name, stubs, tests, desc["syntax_hint"])

        t0 = time.time()
        # Generalized from a hardcoded one retry (2 attempts) into N attempts,
        # so a higher --max-attempts can be tested without changing anything
        # else about the loop (result-JSON shape beyond stop_reasons/
        # turn_count below, pass_N labeling convention). --no-retry
        # (retry=False) still means exactly 1 attempt regardless of
        # --max-attempts, matching the previous behavior of that flag.
        #
        # Each attempt opens a FRESH PiRpc session rather than reusing one
        # continuous conversation across attempts. Reusing one session lets
        # context balloon with every attempt's full verbose reasoning trail
        # (worse at higher --thinking effort), and on a real multi-attempt
        # failure this drove pi's own client to cancel the stream mid-turn --
        # after which pi's internal turn-state never recovered ("Agent is
        # already processing" on every subsequent prompt in that session, no
        # matter how long you wait). A fresh session per attempt keeps each
        # prompt's context bounded to "current file state + latest failure,"
        # sidestepping the bug entirely: the model still sees its own previous
        # edit by reading the file (which persists in `work` across
        # attempts), just not its own prior reasoning transcript.
        effective_attempts = max_attempts if retry else 1
        attempt = None
        passed, out = False, ""
        outcomes: list[str] = []
        stop_reasons: list[str] = []
        turn_total = 0
        current_prompt = prompt
        codex_session_id = None
        for i in range(1, effective_attempts + 1):
            # Reset every iteration, not just declared once before the loop:
            # if a future branch (or an exception) ever skipped reassigning
            # it, this attempt would otherwise be dumped carrying the PRIOR
            # attempt's notifications -- silently wrong forensics in exactly
            # the artifact this exists to make trustworthy.
            attempt_notifications: list[dict] = []
            if agent == "pi":
                with PiRpc(model=model, cwd=str(work), allowed_tools=ALLOWED_TOOLS,
                           session_id=f"poly-{lang}-{ex_name}-attempt{i}",
                           env={"LITTLE_CODER_PERMISSION_MODE": "accept-all"},
                           thinking=thinking) as rpc:
                    if i == 1 and thinking_confirmation is not None and not thinking_confirmation["attempted"]:
                        # Best-effort, and only once per RUN, ever -- gated on
                        # "attempted", not "got a value back", so a FAILED
                        # confirmation also stops later exercises from
                        # re-probing (config doesn't change mid-run; a second,
                        # third, ... attempt after a first failure would just
                        # be more of the same redundant/failing RPC round-trip).
                        # Mutates the caller's dict directly and immediately,
                        # rather than returning through `record` below: if
                        # prompt_and_collect() on this same attempt then
                        # raises, the exception unwinds out of _run_exercise
                        # before any `return` -- a value threaded through the
                        # return dict would be lost, but this mutation already
                        # happened and is visible to the caller regardless.
                        thinking_confirmation["attempted"] = True
                        try:
                            state = rpc.get_state()
                            level = state.get("thinkingLevel")
                            if level:
                                thinking_confirmation["confirmed_live"] = level
                            else:
                                # A successful response with no usable
                                # thinkingLevel must not look identical to
                                # "never attempted" -- both confirmed_live
                                # and error staying None would be exactly
                                # that ambiguity.
                                thinking_confirmation["error"] = (
                                    f"get_state succeeded but returned no thinkingLevel: {state!r}"
                                )
                        except Exception as exc:
                            thinking_confirmation["error"] = f"{type(exc).__name__}: {exc}"
                    # prompt_and_collect() returns partial events *silently*
                    # when agent_end never arrives (_drain_events_until
                    # returns `collected`, it does not raise), so a
                    # budget-capped attempt is otherwise indistinguishable
                    # from a completed one.
                    r = rpc.prompt_and_collect(current_prompt, timeout=ATTEMPT_TIMEOUT_S)
                # ctx.ui.notify events -- skill/knowledge injections, and
                # critically the thinking-budget extension's "the model has
                # thought long enough" intervention. Read AFTER the `with`
                # block (not inside it): PiRpc.close() joins the reader
                # thread before returning, so by the time __exit__ finishes,
                # everything the reader ever drained is guaranteed to be in
                # self._notifications -- reading before close() started (as
                # an earlier version of this code did) had no such guarantee.
                # Harmless if this list ends up empty.
                attempt_notifications = rpc.notifications()
            elif agent == "codex":
                # Unlike pi's fresh-session-per-attempt (see above), codex
                # resumes its own prior session from attempt 2 onward --
                # a different tool with a different retry convention, kept
                # as-is rather than forced into pi's shape. Because it
                # remembers its own conversation, its retry prompt (below)
                # is shorter than pi's, but still restates the DO-NOT-edit-
                # tests guard rather than relying solely on the resumed
                # session's memory of it.
                r, codex_session_id = _run_codex_turn(
                    model, work, current_prompt, codex_session_id, log_dir, str(i))
            else:
                return {"status": "error", "reason": f"unknown agent {agent!r}"}
            turn_total += r.turn_count
            outcome = _attempt_outcome(r)
            outcomes.append(outcome)
            stop_reasons.append(_stop_reason(r))
            # Snapshot BEFORE the tests run and before any retry prompt is
            # sent, so the artifact reflects what THIS attempt produced.
            _dump_trajectory(log_dir, str(i), r, work, notifications=attempt_notifications)
            passed, out = _score(desc, work, desc["timeout_s"])
            (log_dir / f"final_output_{i}.txt").write_text(out)
            if passed:
                attempt = f"pass_{i}"
                break
            if outcome in ("deadline", "process_exit"):
                # A capped or crashed attempt has nothing more for a retry to
                # build on -- stop rather than spend the remaining budget
                # repeating the same failure. empty_response is deliberately
                # NOT included here: per _is_empty_response, it means the
                # provider returned an empty completion with the work tree
                # untouched -- a transient provider-side fault, not a failed
                # attempt with nothing to build on. It's the single most
                # retryable outcome (6 of 16 logged attempts in this repo's
                # own log tree look like this), so let it fall through to a
                # normal retry instead of aborting the rest of the budget on
                # a failure mode that didn't actually consume an attempt's
                # worth of the model's effort.
                break
            if agent == "pi":
                # Repeats the original prompt in full, not just the failure
                # -- each attempt is a fresh session (see above) with no
                # memory of this one, so a retry prompt built from the
                # failure alone drops the exercise name, the stub/test file
                # paths, the "tests are for reference only -- DO NOT edit"
                # constraint, and the syntax hint. Without those, an agent
                # scoring itself against a copy of the tree (_score()) has
                # nothing stopping it from "fixing" the failure by editing
                # the test file instead of the implementation and recording
                # a fabricated pass.
                current_prompt = (
                    prompt
                    + "\n\n---\n\nThis is a retry: the file(s) already contain "
                      "your previous attempt's code (read the current state "
                      "before editing). The tests failed with this output:\n\n```\n"
                    + out[-4000:]
                    + "\n```\n\nFix the implementation and try again."
                )
            else:
                # codex resumes its own session (see above), so it already
                # has the exercise's stub/test paths and syntax hint in
                # context and doesn't need the full prompt restated the way
                # pi's fresh session does. The DO-NOT-edit-tests constraint
                # is restated anyway (cheap insurance): relying solely on
                # the resumed session remembering it is unsound if the
                # session ever gets summarized/compacted, or if a future
                # change resumes the wrong session entirely.
                current_prompt = (
                    "The tests failed. Output:\n\n```\n"
                    + out[-4000:]
                    + "\n```\n\nThe test file(s) are for reference only -- "
                      "do not edit them. Fix the implementation and try again."
                )

        elapsed = time.time() - t0
        (log_dir / "final_output.txt").write_text(out)
        if verbose:
            print(f"[{lang}/{ex_name}] {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s on {attempt or 'fail'}")

        record = {
            "run_id": RUN_ID,
            "status": _classify_status(passed, attempt, outcomes),
            "stop_reasons": stop_reasons,
            "elapsed_s": round(elapsed, 2),
            "turn_count": turn_total,
        }
        return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["pi", "codex"], default="pi")
    ap.add_argument("--model", default=None)
    ap.add_argument("--language", default="python")
    ap.add_argument("--exercise", default=None, help="Run a single exercise")
    ap.add_argument("--exercises", type=int, default=0, help="Run first N exercises (0 = all)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=2,
                     help="Total attempts including the first; default 2 matches "
                          "the original hardcoded one-retry behavior")
    ap.add_argument("--thinking", default=None,
                     choices=["off", "minimal", "low", "medium", "high", "xhigh", "max"],
                     help="--thinking level passed to the pi CLI. Unset means "
                          "whatever pi itself resolves to with no flag -- its "
                          "compiled default is 'medium', but a user- or "
                          "machine-local ~/.pi/agent/settings.json can override "
                          "that, so 'unset' is not a fixed, reproducible level.")
    ap.add_argument("--config-label", default=None,
                     help="Recorded in meta.config_label for result-file bookkeeping when the "
                          "same --agent/--model is run under different tuning configs (e.g. "
                          "sampling params changed on the server, not visible to this script) "
                          "-- avoids collisions in compare_agents.py's (agent, model) grouping. "
                          "Not itself a scoring parameter -- unlike --thinking/--max-attempts, "
                          "it doesn't affect how an attempt runs, so a --resume with a different "
                          "--config-label is not flagged as a mismatch.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.max_attempts < 1:
        sys.exit(f"--max-attempts must be >= 1, got {args.max_attempts}")

    # DEFAULT_MODEL is a pi/llamacpp id; using it as a silent fallback for
    # codex would point at a model that doesn't exist for that agent, so
    # codex requires an explicit --model instead of inheriting pi's default.
    model = args.model or (DEFAULT_MODEL if args.agent == "pi" else None)
    if model is None:
        sys.exit("--model is required for --agent codex (e.g. --model gpt-5.1-codex-max)")

    results = _load_results() if args.resume else {"exercises": {}, "meta": {}}
    if args.config_label:
        results["meta"]["config_label"] = args.config_label

    desc = LANG_DESCRIPTORS.get(args.language)
    if desc is None:
        sys.exit(f"No descriptor for language '{args.language}'. Supported: {list(LANG_DESCRIPTORS)}")

    params = _scoring_params(model, args.language, not args.no_retry, desc,
                             max_attempts=args.max_attempts, thinking=args.thinking,
                             agent=args.agent)
    # Diagnostic only -- deliberately NOT part of _scoring_params/_param_mismatches,
    # same reasoning as config_label: this doesn't affect how an attempt runs,
    # so a --resume under a different sampling temperature isn't a mismatch.
    env_snapshot = capture_environment_snapshot(model, cli_thinking=args.thinking, agent=args.agent)
    if args.resume:
        mismatches = _param_mismatches(results["meta"].get("scoring_params", {}), params)
        if mismatches and results["exercises"]:
            # Warn rather than refuse: the documented canonical run deliberately
            # changed a scoring parameter mid-run and resumed
            # (docs/benchmark-qwen3.6-35b-a3b.md). Refusing would have blocked
            # it. Every record carries its own parameters, so a mixed file stays
            # analysable instead of merely looking homogeneous.
            print("WARNING: resuming with different scoring parameters --", file=sys.stderr)
            for line in mismatches:
                print(f"  {line}", file=sys.stderr)
            print("  Existing results were produced under the old values; new ones "
                  "will carry the new values.", file=sys.stderr)

    # runs[] keeps every invocation's parameters instead of overwriting them,
    # so meta.model no longer describes only the last run of a mixed file.
    results["meta"].setdefault("runs", []).append({
        "run_id": RUN_ID,
        "started_at": datetime.datetime.now().isoformat(),
        "scoring_params": params,
        "environment_snapshot": env_snapshot,
    })
    results["meta"]["scoring_params"] = params
    results["meta"]["environment_snapshot"] = env_snapshot
    results["meta"]["run_id"] = RUN_ID
    # Otherwise a --resume that skips every already-passed exercise (or an
    # empty exercise set) never reaches the per-exercise _save_results()
    # below, and this run's metadata -- including which config/environment
    # it ran under -- is silently never persisted at all.
    _save_results(results)

    practice = desc["practice_dir"]
    if args.exercise:
        names = [args.exercise]
    else:
        names = sorted(p.name for p in practice.iterdir() if p.is_dir())
        if args.exercises:
            names = names[:args.exercises]

    consecutive_errors = 0
    written_this_run: dict[str, dict] = {}
    # Owned by main(), mutated directly by _run_exercise on the first
    # exercise's first attempt (see there for why: a dict mutation survives
    # even if _run_exercise raises afterward, unlike a value threaded through
    # its return dict). "attempted" gates on ATTEMPTED, not "got a value" --
    # a failed confirmation must also stop every later exercise from
    # re-probing and re-appending the same error, not just a successful one.
    thinking_confirmation = {"attempted": False, "confirmed_live": None, "error": None}
    confirmation_error_recorded = False
    for name in names:
        key = f"{args.agent}/{args.language}/{name}"
        if args.resume and str(results["exercises"].get(key, {}).get("status", "")).startswith("pass_"):
            continue
        # Results are checkpointed atomically per exercise one line below, but
        # an exception anywhere in _run_exercise still discarded every remaining
        # exercise. Record and continue -- while still failing loudly if the
        # environment itself is broken (missing pi CLI, model server down),
        # which would otherwise write "error" 225 times and print a summary
        # that reads like a finished 0% run.
        try:
            r = _run_exercise(
                args.language, name, model,
                verbose=args.verbose,
                retry=not args.no_retry,
                max_attempts=args.max_attempts,
                thinking=args.thinking,
                agent=args.agent,
                thinking_confirmation=thinking_confirmation,
            )
        except Exception as exc:
            r = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400]}
            print(f"[{args.language}/{name}] ERROR {r['reason']}")

        # Idempotent after the first exercise (thinking_confirmation stops
        # changing once "attempted" is True), so applying it unconditionally
        # every iteration is harmless -- simpler than trying to detect "this
        # was the exercise that just attempted it".
        if thinking_confirmation["confirmed_live"] is not None:
            env_snapshot["thinking"]["confirmed_live"] = thinking_confirmation["confirmed_live"]
        elif thinking_confirmation["error"] is not None and not confirmation_error_recorded:
            # A failed confirmation attempt must not just vanish -- otherwise
            # confirmed_live staying null is indistinguishable from "never
            # tried" versus "tried and pi's get_state rejected the request".
            # confirmation_error_recorded (a local, not part of env_snapshot
            # itself) is the dedup flag, so this stays out of the persisted
            # JSON -- env_snapshot only ever holds data meant to be written.
            env_snapshot.setdefault("errors", []).append(
                {"source": "confirmed_thinking", "error": thinking_confirmation["error"]})
            confirmation_error_recorded = True

        r["scoring_params"] = params
        results["exercises"][key] = r
        written_this_run[key] = r
        _save_results(results)
        # Counts BOTH raised exceptions and returned {"status": "error"} --
        # _run_exercise returns that for a missing exercise directory, so a
        # stale benchmark checkout would otherwise never trip the guard.
        if str(r.get("status")) == "error":
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(
                    f"aborting after {consecutive_errors} consecutive exercise "
                    f"errors -- the environment looks broken, not the exercises"
                )
        else:
            consecutive_errors = 0

    print(json.dumps({
        k: v["status"]
        for k, v in results["exercises"].items()
    }, indent=2))

    code = _exit_code(written_this_run)
    if code:
        bad = [k for k, v in written_this_run.items() if str(v.get("status")) == "error"]
        print(f"harness errors in this run: {', '.join(bad)}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
