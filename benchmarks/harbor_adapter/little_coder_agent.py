"""Harbor (Terminal-Bench 2.0) adapter for little-coder.

Subclasses harbor.agents.base.BaseAgent (the TB 2.0 counterpart of TB 1.0's
terminal_bench.agents.base_agent.BaseAgent). The heavy lifting — pi RPC
subprocess, extension stack, ShellSession proxy — is shared with the TB 1.0
adapter via benchmarks/rpc_client.py::PiRpc.

The one moving part that differs from TB 1.0:

  TB 1.0:  agent gets a TmuxSession.send_keys(...) interface — sync.
  TB 2.0:  agent gets environment.exec(command, ...) — *async*.

My PiRpc reader thread invokes the shell-proxy callback synchronously when
an extension_ui_request with the __LC_TB_SHELL__ prefix arrives. To call
harbor's async env.exec from that sync context, we stash the event loop
in run() and use asyncio.run_coroutine_threadsafe().

Launch:

    harbor run \
      --dataset terminal-bench@2.0 \
      --agent-import-path benchmarks.harbor_adapter.little_coder_agent:LittleCoderAgent \
      --model llamacpp/qwen3.6-35b-a3b \
      --n-concurrent 1
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

# Repo root, derived the same way _read_version_from_package_json() finds
# package.json -- benchmarks/harbor_adapter/little_coder_agent.py is two
# levels below it.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_version_from_package_json() -> str:
    """Read agent version from the repo's package.json at import time.

    Avoids hardcoded version drift between the adapter's version() return
    and the actual released tag. Falls back to "unknown" if the file is
    missing or malformed.
    """
    try:
        pkg = _REPO_ROOT / "package.json"
        return json.load(open(pkg)).get("version", "unknown")
    except Exception:
        return "unknown"


def _read_code_sha() -> str:
    """Best-effort short git SHA of this repo's HEAD, computed once at
    module IMPORT time -- not per-trial. Import time is exactly when this
    process's copy of the adapter code was frozen, so a Harbor job that has
    been running for hours on stale code still reports its OWN stale SHA
    here, unaffected by anything landing on disk afterward -- without this,
    a long-running job silently keeps enforcing whatever behavior was
    current when it started, with nothing on disk recording which code was
    actually active. Falls back to "unknown" for a non-git checkout or
    missing git binary, mirroring _read_version_from_package_json()'s
    fallback.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        sha = out.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


_AGENT_VERSION = _read_version_from_package_json()
_CODE_SHA = _read_code_sha()
try:
    _ADAPTER_MTIME = Path(__file__).resolve().stat().st_mtime
except Exception:
    _ADAPTER_MTIME = None


from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# benchmarks/ isn't a package — let the importer resolve by sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rpc_client import (  # noqa: E402
    PiRpc,
    capture_environment_snapshot,
    preview_tool_result,
    prompt_with_error_retry,
)


DEFAULT_ALLOWED_TOOLS = ["ShellSession", "ShellSessionCwd", "ShellSessionReset", "ShellRecall"]
DEFAULT_MODEL = "llamacpp/qwen3.6-35b-a3b"

# Fallback when the real per-task timeout can't be derived (see
# _resolve_trial_timeout_sec) -- e.g. a manual/local run with no Harbor
# trial_dir, or a future Harbor version with a different cache layout.
DEFAULT_PROMPT_TIMEOUT_SEC = 3600.0
# Safety margin below Harbor's own enforced timeout: leaves headroom for
# our own graceful cutoff (and finalize-warn's wall-clock nudge) to land
# before Harbor's external asyncio.wait_for kills the process outright.
DEADLINE_SAFETY_MARGIN = 0.9
HARBOR_TASK_CACHE = Path.home() / ".cache" / "harbor" / "tasks"

# A deadline snapshot of files the model changed since trial start, staged
# under /tmp/.lc-snapshot in the container.
# Insurance/recovery only -- grading reads the live container files, never
# this snapshot; it's useful only via the finalize-message pointer telling
# the model it exists, for the case where a late mistake destroys earlier
# good state (the motivating trial: overfull-hbox, where a debug script
# clobbered a graded file that had already reached the goal state).
#
# Fires SNAPSHOT_LEAD_SEC before the trial's own effective_timeout_sec --
# deliberately mirrors finalize-warn's own WARN_REMAINING_MS (see
# .pi/extensions/finalize-warn/index.ts) so the snapshot and the "you're
# running low" nudge land around the same moment; if either constant is
# ever changed there, consider changing it here too (no shared import --
# this is Python, that's TypeScript -- so the two are only in comment-level
# lockstep, same as tb-finalize-guard's own WARN_REMAINING_MS is with
# finalize-warn's).
SNAPSHOT_LEAD_SEC = 600.0
# Below this total trial budget, skip the snapshot entirely: the computed
# delay (effective_timeout_sec - SNAPSHOT_LEAD_SEC) would already be clamped
# to 0 (snapshotting almost immediately at trial start), and for a trial this
# short there's nothing meaningful yet to recover that the model couldn't
# just redo from scratch.
SNAPSHOT_MIN_BUDGET_SEC = 300.0
SNAPSHOT_START_MARKER = "/tmp/.lc-start"
SNAPSHOT_PUBLISH_PATH = "/tmp/.lc-snapshot"
# Stage dirs are "<prefix>.$$" and the cleanup glob is "<prefix>.*" --
# _build_snapshot_command derives both from this one prefix, so a command
# can never create a stage dir under a name its own cleanup line doesn't
# reap.
SNAPSHOT_STAGE_PREFIX = "/tmp/.lc-snapshot.stage"
# Feeds `head -z -n` below, and is the threshold that tells a complete
# snapshot from one truncated at the cap.
SNAPSHOT_MAX_FILES = 500

# Start-of-trial snapshot of the task's PRE-EXISTING /app files. Catches the
# class no write-detector can see -- a program the model spawned overwriting
# a task-provided input from inside its own process (the motivating trial:
# overfull-hbox, whose generated Perl script opened /app/input.tex for write).
# Its own publish path and stage prefix, never the deadline snapshot's: the
# two are published at different times under different scopes, and a shared
# name would have one command's cleanup reap the other's directory.
INITIAL_SNAPSHOT_PUBLISH_PATH = "/tmp/.lc-initial"
INITIAL_SNAPSHOT_STAGE_PREFIX = "/tmp/.lc-initial.stage"
# Host-side destination, under the per-trial logs dir: per-trial by
# construction, sitting next to environment_snapshot.json where a human doing
# post-mortem recovery already looks, and download_dir lands it as a real
# directory tree with the container's own paths preserved.
INITIAL_SNAPSHOT_DIR_NAME = "initial_state"

# Bounded, atomically-staged copy of files under /app, shared by both
# snapshots (see _build_snapshot_command): the deadline one, scoped to files
# modified since SNAPSHOT_START_MARKER was touched, and the start-of-trial
# one, scoped to everything. Every cap here answers a specific failure mode:
#   - per-file size cap (-size -10M) and an aggregate file-count cap
#     (head -z -n 500) and an aggregate byte cap (209715200 = 200MB, via the
#     `du --files0-from` sum) together bound total copy volume regardless of
#     how many small files changed -- a per-file cap alone is insufficient
#     (thousands of small files could still copy gigabytes).
#   - a free-space reserve check (FREE >= TOTAL + 524288000, i.e. 500MB)
#     protects storage_mb-tight task containers (10240MB typical in TB2.1
#     task.tomls) from being pushed over their quota by the snapshot itself.
#   - staging into $STAGE and only `mv`-ing it to the publish path once
#     fully populated means a half-copied snapshot is never visible at the
#     published path (atomic publish).
#   - the whole body runs under an internal `timeout 20`, and the trailing
#     `rm -rf <stage prefix>.*` (outside that timeout) reaps a stage dir
#     orphaned if the 20s kill lands mid-copy ("cleanup-on-timeout" duty).
#     Documented here rather than as a trailing inline comment on the command
#     string's own last line: _exec_async appends a wrapper epilogue
#     (`; }} ; __rc=$? ; printf ... ; pwd`) directly onto whatever this string
#     ends with, so a trailing `#` comment with no newline after it silently
#     swallows that whole epilogue. Do not add a trailing inline comment to
#     this constant's last line.
#   - `set -e` plus every failure being swallowed by the caller means this
#     command degrades to "no snapshot" on any error (missing GNU coreutils
#     like `head -z`/`du --files0-from` on a BusyBox-ish image, `find -newer`
#     failing because the model deleted the start marker, etc.) -- never to
#     an unbounded copy.
#   - the publish branch is additionally gated on `[ -s "$STAGE/.list" ]`:
#     with zero candidate files (e.g. the model deleted /tmp/.lc-start, so
#     `find -newer` errors -- not fatal under `set -e` since the pipeline's
#     exit status is `head`'s, which is 0 -- or the model simply hasn't
#     touched anything yet), GNU `xargs` still runs its command once with no
#     input unless told not to, so `cp --parents -t "$STAGE"` would run with
#     no operands, fail, get swallowed by `|| true`, and publish an empty
#     $STAGE anyway. `-s` is the portable fix and also directly covers "find
#     matched nothing". Deliberately NOT also passing `xargs -r`
#     (--no-run-if-empty) as belt-and-braces: that flag is GNU-specific, and
#     under `set -e` an unsupported flag on a non-GNU findutils image would
#     abort the whole script instead of merely running xargs once -- turning
#     a portability gap into a silent full-feature outage, which is worse
#     than the bug the `-s` gate already fixes on its own. This codebase has
#     no confirmed inventory of every TB task container's base image/findutils
#     provenance, so rather than bet on GNU everywhere, `-s` alone is the
#     whole fix here -- it's portable and sufficient on its own.


def _build_snapshot_command(
    *,
    scope_comment: str,
    find_predicate: str,
    publish_path: str,
    stage_prefix: str,
) -> str:
    """Instantiate the bounded-copy command documented above.

    Parameterized on exactly what differs between the two snapshots -- the
    extra `find` predicate that sets the scope (and the comment naming it),
    the publish path, and the stage-dir prefix -- so the second snapshot
    inherits the first's hardening instead of growing a second copy
    discipline of its own.

    stage_prefix drives both `STAGE=<prefix>.$$` and the trailing
    `rm -rf <prefix>.*`: passing them separately is how an instantiation ends
    up leaking a stage dir per timeout kill under a name nothing reaps.

    find_predicate is inserted after the shared caps and before -print0; ""
    means no extra predicate. _SNAPSHOT_COMMAND's instantiation below is
    byte-identical to the literal this constant held before the template
    existed, pinned by test_harbor_snapshot.py -- that assertion is the whole
    safety argument for refactoring a command that already runs in real
    trials.
    """
    predicate = f" {find_predicate}" if find_predicate else ""
    return (
        "timeout 20 sh -c '\n"
        "  set -e\n"
        f"  STAGE={stage_prefix}.$$\n"
        "  rm -rf \"$STAGE\" && mkdir -p \"$STAGE\"\n"
        f"  # {scope_comment}\n"
        f"  find /app -xdev -maxdepth 3 -type f -size -10M{predicate} -print0 2>/dev/null \\\n"
        f"    | head -z -n {SNAPSHOT_MAX_FILES} > \"$STAGE/.list\"           # aggregate file-count cap\n"
        "  TOTAL=$(du -cb --files0-from=\"$STAGE/.list\" 2>/dev/null | tail -1 | cut -f1)\n"
        "  FREE=$(df -B1 --output=avail /tmp | tail -1)\n"
        "  # non-empty candidate list AND aggregate byte cap 200MB AND leave >=500MB free space reserve\n"
        "  if [ -s \"$STAGE/.list\" ] && [ \"${TOTAL:-0}\" -le 209715200 ] && [ \"${FREE:-0}\" -ge $((TOTAL + 524288000)) ]; then\n"
        "    xargs -0 -a \"$STAGE/.list\" cp --parents -t \"$STAGE\" 2>/dev/null || true\n"
        "    rm -f \"$STAGE/.list\"\n"
        f"    rm -rf {publish_path} && mv \"$STAGE\" {publish_path}   # atomic publish\n"
        "  else\n"
        "    rm -rf \"$STAGE\"                                             # refuse oversize or empty\n"
        "  fi\n"
        f"' ; rm -rf {stage_prefix}.* 2>/dev/null"
    )


_SNAPSHOT_COMMAND = _build_snapshot_command(
    scope_comment="candidate list: files under /app changed since trial start, per-file <10M",
    find_predicate=f"-newer {SNAPSHOT_START_MARKER}",
    publish_path=SNAPSHOT_PUBLISH_PATH,
    stage_prefix=SNAPSHOT_STAGE_PREFIX,
)

# How many files the start-of-trial snapshot actually published, appended to
# that command only -- outside the shared template, so the deadline
# instantiation stays byte-identical. The rc cannot carry this: the template
# ends in a best-effort cleanup `rm`, so rc reports that rm, and the refuse
# branch exits 0 exactly like the publish branch does.
_INITIAL_SNAPSHOT_COUNT_PREFIX = "lc-initial-files="
_INITIAL_SNAPSHOT_COUNT_PROBE = (
    f" ; printf '{_INITIAL_SNAPSHOT_COUNT_PREFIX}%s\\n' "
    f'"$(find {INITIAL_SNAPSHOT_PUBLISH_PATH} -type f 2>/dev/null | wc -l)"'
)

# The same bounded copy with the freshness filter dropped: at trial start
# "every file under /app" is exactly "every pre-existing file", which is the
# scope this snapshot exists to preserve.
#
# Every cap carries over unchanged but means something different against that
# wider scope -- flagged here, not resolved: -maxdepth 3, the file-count cap
# and -size -10M were sized for a modified-file delta, so against a whole tree
# they can truncate silently (rc is 0 either way), and a >200MB aggregate makes
# the command refuse outright rather than copy part of the tree. Hence the
# three-way outcome logging in _classify_initial_snapshot; a bare "succeeded"
# would be a lie on two of those paths.
_INITIAL_SNAPSHOT_COMMAND = (
    # Unconditional cleanup of the publish path itself before staging even
    # starts -- outside the shared template (which only clears it on its
    # own success path), so a refusal always finds nothing there and the
    # probe below correctly reports 0 files, not whatever a previous
    # invocation happened to leave behind. Not expected to ever matter in
    # practice (a fresh container has nothing at this path the first time
    # this command ever runs), but "nothing at this path" should not be
    # allowed to depend on that assumption holding.
    f"rm -rf {INITIAL_SNAPSHOT_PUBLISH_PATH} ; "
    + _build_snapshot_command(
        scope_comment="candidate list: every file under /app at trial start, per-file <10M",
        find_predicate="",
        publish_path=INITIAL_SNAPSHOT_PUBLISH_PATH,
        stage_prefix=INITIAL_SNAPSHOT_STAGE_PREFIX,
    )
    + _INITIAL_SNAPSHOT_COUNT_PROBE
)


def _compute_snapshot_delay_sec(effective_timeout_sec: float) -> float | None:
    """Pure helper (split out for testability) for run()'s scheduling of
    _snapshot_at_deadline: returns the sleep delay to pass it, or None to
    skip scheduling the snapshot task at all.

    Below SNAPSHOT_MIN_BUDGET_SEC, returns None -- a short-task edge case:
    the naive delay (effective_timeout_sec - SNAPSHOT_LEAD_SEC) would
    already be negative and get clamped to 0 (snapshotting almost
    immediately), and for a trial this short there's nothing meaningful yet
    to recover that the model couldn't just redo from scratch, so the whole
    task is skipped rather than merely delay-clamped.
    """
    if effective_timeout_sec < SNAPSHOT_MIN_BUDGET_SEC:
        return None
    return max(0.0, effective_timeout_sec - SNAPSHOT_LEAD_SEC)


_HARNESS_EXIT_CODE_RE = re.compile(r"^\[exit=(-?\d+)", re.MULTILINE)


def _extract_exit_code(formatted_output: str) -> int | None:
    """Pulls the rc _format_output embeds in its footer line (e.g.
    `[exit=0 cwd=/app timed_out=false backend=harbor-env]`) back out of
    run_harness's returned string. Pure/module-level so it's directly
    testable without a fake proxy.

    Returns None only if the footer itself is missing/malformed (should not
    happen in practice -- _format_output always emits it); callers should
    treat that the same as "did not succeed".
    """
    # Last match, not the first: a command's own output can contain a line
    # shaped like the footer, and a mid-line byte cut can even create one.
    # The real footer is always last.
    last = None
    for last in _HARNESS_EXIT_CODE_RE.finditer(formatted_output):
        pass
    return int(last.group(1)) if last else None


async def _snapshot_at_deadline(proxy: "_HarborShellProxy", delay_sec: float, logger: logging.Logger) -> None:
    """Sleeps until SNAPSHOT_LEAD_SEC before the trial deadline, then fires
    the one bounded snapshot command via run_harness -- never proxy.run()
    (see _HarborShellProxy.run_harness's docstring for why that would
    deadlock this same event loop). Meant to be wrapped in
    asyncio.create_task() by run() and cancelled in run()'s own finally.

    Insurance only: must never raise into the trial. Every failure --
    including the container simply not having the snapshot command's
    required coreutils -- is swallowed here; asyncio.CancelledError is the
    one exception let through, so run()'s task.cancel() actually cancels
    promptly instead of being silently caught by the broad handler below.

    Logs an outcome derived from the actual rc instead of an unconditional
    "attempted" message -- that unconditional message is exactly what let
    2.1's unterminated-comment syntax error (rc=2, every single time) ship
    and run undetected: "attempted" is true whether or not anything actually
    happened, so it told us nothing.
    """
    try:
        await asyncio.sleep(delay_sec)
        out = await proxy.run_harness(_SNAPSHOT_COMMAND, timeout=25)
        rc = _extract_exit_code(out)
        if rc == 0:
            logger.info("LittleCoderAgent: deadline snapshot succeeded")
        else:
            logger.info(f"LittleCoderAgent: deadline snapshot did not produce a snapshot (rc={rc})")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.info(f"LittleCoderAgent: deadline snapshot failed (non-fatal): {e}")


_INITIAL_SNAPSHOT_COUNT_RE = re.compile(
    rf"^{_INITIAL_SNAPSHOT_COUNT_PREFIX}\s*(\d+)", re.MULTILINE
)
# Bounds only the docker-cp download. TimeoutError is an Exception subclass,
# so this is caught by the same try/except that already preserves the stage
# outcome on a raised download error.
_INITIAL_SNAPSHOT_DOWNLOAD_TIMEOUT_SEC = 35.0
# Backstop only: the stage (run_harness's own 25s) and the download (above)
# are each individually bounded and already preserve the stage outcome on
# their own timeout. Sized for slack above both, not to race them.
_INITIAL_SNAPSHOT_TIMEOUT_SEC = 75.0


def _parse_initial_snapshot_file_count(formatted_output: str) -> int | None:
    """Pull _INITIAL_SNAPSHOT_COUNT_PROBE's count back out of run_harness's
    returned string. Pure/module-level so it's directly testable without a
    fake proxy.

    None means the probe line never arrived -- a killed command, a `find`/`wc`
    the image doesn't have, output truncated ahead of it -- which callers
    treat the same as "did not succeed", never as zero files.
    """
    # Last match, for the same reason _extract_exit_code takes the last
    # footer: a command's own output can contain a line shaped like this one.
    last = None
    for last in _INITIAL_SNAPSHOT_COUNT_RE.finditer(formatted_output):
        pass
    return int(last.group(1)) if last else None


class _InitialSnapshotOutcome(NamedTuple):
    outcome: str
    download: bool
    message: str


def _classify_initial_snapshot(rc: int | None, file_count: int | None) -> _InitialSnapshotOutcome:
    """Pure helper (split out for testability, like _compute_snapshot_delay_sec)
    turning the start-of-trial snapshot's two observable results into the
    outcome to log and whether there is anything worth downloading.

    Four outcomes rather than "attempted"/"succeeded", because two real cases
    are neither: a copy truncated at the file-count cap, and a refusal that
    published nothing at all. Both exit 0, so rc alone cannot tell them from a
    complete snapshot -- hence the file-count probe.

    "partial" here means the file-count cap specifically. -maxdepth 3 and the
    per-file -size -10M drop files with no observable trace, so even a
    "succeeded" snapshot can be missing a large or deeply-nested original;
    anything pointing the model at this copy has to say so.

    rc is accepted and logged for visibility only, never branched on: the
    wrapped command's own last statement is always its trailing cleanup
    `rm`, which exits 0 whether staging succeeded, refused, or crashed
    partway through, so the sentinel _wrap_command captures can never
    actually distinguish those cases. The same is true of the pre-existing
    deadline snapshot's rc, just never load-bearing there since it only
    feeds a log message. A genuine crash before the probe's own printf
    ever ran is instead caught below, correctly, by the probe line simply
    never arriving.
    """
    if file_count is None:
        return _InitialSnapshotOutcome(
            "failed",
            False,
            f"failed -- stage command reported no file count (rc={rc}); skipping download",
        )
    if file_count == 0:
        return _InitialSnapshotOutcome(
            "refused",
            False,
            f"refused -- nothing published at {INITIAL_SNAPSHOT_PUBLISH_PATH} "
            "(aggregate over the 200MB cap, free-space reserve unmet, or no "
            "candidate files); skipping download",
        )
    if file_count >= SNAPSHOT_MAX_FILES:
        return _InitialSnapshotOutcome(
            "partial",
            True,
            f"partial -- {file_count} files, at the {SNAPSHOT_MAX_FILES}-file cap; "
            "files past the cap have no start-of-trial copy",
        )
    return _InitialSnapshotOutcome(
        "succeeded", True, f"succeeded -- {file_count} files staged"
    )


def _initial_snapshot_is_present(outcome: _InitialSnapshotOutcome | None) -> bool:
    """True when staging actually published files the model can read.

    The single gate for everything that points the model at the copy, so the
    prompt paragraph and the extensions' env var can never disagree about
    whether the path exists.
    """
    return outcome is not None and outcome.outcome not in ("failed", "refused")


def _initial_snapshot_advertisement(
    outcome: _InitialSnapshotOutcome | None,
) -> str | None:
    """The prompt paragraph telling the model the start-of-trial copy exists,
    or None to say nothing. Pure/module-level, like _classify_initial_snapshot.

    Silent on None/failed/refused: those are exactly the cases where the copy
    may not be there, and pointing the model at a path that does not exist
    costs it turns for nothing -- worse than never mentioning it.

    The text is deliberately reactive, not a standing instruction to diff
    against this copy as routine practice: it earns its place only when the
    model already suspects a specific file was clobbered. And the restore
    restraint is spelled out because the dangerous misreading is the obvious
    one -- a model near its deadline "restoring" pristine originals over the
    solution it just finished writing.
    """
    if not _initial_snapshot_is_present(outcome):
        return None
    app_copy = f"{INITIAL_SNAPSHOT_PUBLISH_PATH}/app"
    text = (
        "Recovery note: a reference copy of this task's starting files (taken "
        "at trial start, before any of your changes) exists inside the "
        f"container under `{app_copy}/` (e.g. `{app_copy}/somefile` mirrors "
        "`/app/somefile`). If you ever suspect a task-provided file was "
        "corrupted or overwritten — by a killed command, a buggy script, or "
        "your own edit — diff against or restore from that copy instead of a "
        "backup you made later. Restore from there only a file you believe you "
        "corrupted — never over your own completed solution. Treat it as "
        f"read-only and never write into `{INITIAL_SNAPSHOT_PUBLISH_PATH}`. It "
        "may not contain very large (>10MB) or deeply nested files."
    )
    if outcome.outcome == "partial":
        text += (
            f" The copy hit its {SNAPSHOT_MAX_FILES}-file cap, so some starting "
            "files are absent from it — absence there does not mean the file "
            "didn't exist."
        )
    return text


def _pi_env(
    *,
    budget_start_epoch_ms: int,
    deadline_epoch_ms: int,
    initial_snapshot: _InitialSnapshotOutcome | None,
) -> dict[str, str]:
    """The env pi's extensions are handed for this trial. Pure/module-level,
    like _initial_snapshot_advertisement, so its contents are assertable
    without standing up a whole run.

    permission-gate's SAFE_PREFIXES whitelist is meant to guard a real user's
    own machine during interactive use, and its own header documents this
    opt-out for benchmark runs. Docker is the actual isolation boundary for a
    TB trial, and every other TB agent (bare pi, codex) already runs here with
    unrestricted tool access. Observed directly: fix-git blocked on `cd`/`git
    -C`, prove-plus-comm blocked on `coqc`, configure-git-webserver blocked on
    `setsid`/`nc`/`socat`/`crontab` -- three different tools across three
    unrelated tasks, not a pattern fixable by allow-listing one command at a
    time.

    LITTLE_CODER_BUDGET_START_EPOCH_MS is the other end of the same interval
    the deadline closes. Only the pair says what FRACTION of the trial is
    gone, which is what tb-finalize-guard's progress checkpoints fire on; the
    deadline alone says how long is left and nothing about how long that was
    out of. Absolute epoch ms on both ends rather than a duration on one, so
    a reader resolves them by one rule.

    LITTLE_CODER_INITIAL_SNAPSHOT is what lets an extension mention the
    start-of-trial copy: tb-finalize-guard runs on TB1.0 too, whose adapter
    stages no such copy, so an unconditional pointer there would send the
    model after a path that does not exist. Carrying the classified outcome
    rather than a bare flag also lets the reader pass on the same
    cap-truncation caveat the prompt paragraph carries.

    Set to empty, not omitted, when no copy was staged: PiRpc builds the
    child env as dict(os.environ) updated with this dict, so an omitted key
    does not clear one already present in THIS process's own environment
    (e.g. leaked from an earlier trial in the same worker/shell). Empty
    still reads as "no copy" to the var's one reader,
    initialSnapshotOutcome(), whose exact match on "succeeded"/"partial"
    treats it the same as absent.
    """
    env = {
        "LITTLE_CODER_PERMISSION_MODE": "accept-all",
        "LITTLE_CODER_BUDGET_START_EPOCH_MS": str(budget_start_epoch_ms),
        "LITTLE_CODER_DEADLINE_EPOCH_MS": str(deadline_epoch_ms),
    }
    env["LITTLE_CODER_INITIAL_SNAPSHOT"] = (
        initial_snapshot.outcome if _initial_snapshot_is_present(initial_snapshot) else ""
    )
    return env


async def _snapshot_initial_state(
    proxy: "_HarborShellProxy",
    environment: BaseEnvironment,
    logs_dir: Path | None,
    logger: logging.Logger,
) -> _InitialSnapshotOutcome | None:
    """Stage a bounded copy of the task's pre-existing /app files inside the
    container at trial start, then pull it onto the host under
    logs_dir/INITIAL_SNAPSHOT_DIR_NAME.

    Insurance only: must never raise into the trial. A failed stage -- a
    container without GNU coreutils, say -- degrades to "no snapshot" and
    one log line; a download the environment backend can't do, or a slow
    docker-cp, costs only the host-side copy. That same swallowing is why
    the outcome is logged from the real rc and real file count: a silent
    nothing here looks identical to success.

    The container-side copy is left in place afterwards rather than deleted.
    It gives the model an in-container restore source from turn 1 (`cp
    /tmp/.lc-initial/app/input.tex /app/input.tex` would have recovered the
    motivating trial outright). Accepted cost, flagged not fixed: it holds up
    to 200MB of the same /tmp the deadline snapshot measures its own 500MB
    free-space reserve against, so on a disk-tight container it can be what
    makes that later snapshot refuse.

    Returns the CONTAINER-side stage outcome (None when there isn't one), for
    _initial_snapshot_advertisement to decide what the model gets told. That
    is the copy the model can actually reach, so the host-side download
    failing does not nullify it -- see the inner function.
    """
    if logs_dir is None:
        logger.info(
            "LittleCoderAgent: skipping initial snapshot -- no per-trial logs dir"
        )
        return None
    try:
        return await asyncio.wait_for(
            _snapshot_initial_state_inner(proxy, environment, logs_dir, logger),
            timeout=_INITIAL_SNAPSHOT_TIMEOUT_SEC,
        )
    except Exception as e:
        logger.info(f"LittleCoderAgent: initial snapshot failed (non-fatal): {e}")
        return None


async def _snapshot_initial_state_inner(
    proxy: "_HarborShellProxy",
    environment: BaseEnvironment,
    logs_dir: Path,
    logger: logging.Logger,
) -> _InitialSnapshotOutcome:
    """The two-step body _snapshot_initial_state wraps in its timeout and
    catch-all: stage in the container, then download what got published.

    The two steps have independent value, so the download gets its own
    try/except rather than riding the caller's: the model is pointed at the
    CONTAINER-side copy, which a failed host-side docker-cp neither removes
    nor invalidates. Letting that failure discard the stage outcome would
    silence the recovery note over a problem the model never sees."""
    out = await proxy.run_harness(_INITIAL_SNAPSHOT_COMMAND, timeout=25)
    result = _classify_initial_snapshot(
        _extract_exit_code(out), _parse_initial_snapshot_file_count(out)
    )
    logger.info(f"LittleCoderAgent: initial snapshot {result.message}")
    if not result.download:
        return result
    target = logs_dir / INITIAL_SNAPSHOT_DIR_NAME
    try:
        # Required, not defensive: docker's download_dir runs `docker compose
        # cp service:SRC/. DEST`, and `docker cp SRC/. DEST` needs DEST to
        # already exist -- only download_dir_with_exclusions' base
        # implementation mkdirs its own target. Without this the download
        # fails on every trial, and the catch-all would swallow it.
        target.mkdir(parents=True, exist_ok=True)
        await asyncio.wait_for(
            environment.download_dir(INITIAL_SNAPSHOT_PUBLISH_PATH, target),
            timeout=_INITIAL_SNAPSHOT_DOWNLOAD_TIMEOUT_SEC,
        )
    except Exception as e:
        logger.info(
            f"LittleCoderAgent: initial snapshot download failed (non-fatal): {e}"
        )
        return result
    logger.info(
        f"LittleCoderAgent: initial snapshot downloaded to {target} "
        f"(container-side copy kept at {INITIAL_SNAPSHOT_PUBLISH_PATH})"
    )
    return result


# Start-of-trial toolchain probe. The image a TB task ships is minimal and
# varies per task: write-compressor burned ~35 turns discovering by trial and
# error that python3 was absent, then pivoting to Perl.
#
# Closed candidate list by design -- it is both what the probe asks about and
# what _parse_toolchain_probe will accept back, so nothing outside it can
# reach the model's prompt. Hence the prompt line's "others may exist".
_TOOLCHAIN_CANDIDATES = (
    "python3",
    "python",
    "perl",
    "awk",
    "gcc",
    "cc",
    "g++",
    "make",
    "node",
)
_TOOLCHAIN_PROBE_TIMEOUT_SEC = 10
# Built from the tuple above so the two can never drift apart. Only stdout is
# ever read: the loop's rc is non-zero whenever the LAST candidate happens to
# be absent, the same rc-is-unusable trap _classify_initial_snapshot documents.
_TOOLCHAIN_PROBE_COMMAND = (
    "for c in "
    + " ".join(_TOOLCHAIN_CANDIDATES)
    + "; do command -v $c >/dev/null 2>&1 && printf '%s ' $c; done"
)


def _parse_toolchain_probe(out: str) -> list[str] | None:
    """Pull the detected tools back out of run_harness's returned string.
    Pure/module-level so it's directly testable without a fake proxy.

    Filtered against _TOOLCHAIN_CANDIDATES rather than taken verbatim: the
    string also carries _format_output's own footer line, and on a broken
    image whatever the shell printed instead. This list is interpolated
    straight into the model's prompt, so a fabricated tool name here is worse
    than no line at all.

    Returns None, never [], when nothing usable is found -- an empty result,
    unparseable garbage and "no candidate present" are one case to every
    caller: there is nothing safe to tell the model.
    """
    tokens = set((out or "").split())
    found = [c for c in _TOOLCHAIN_CANDIDATES if c in tokens]
    return found or None


def _toolchain_probe_note(tools: list[str] | None) -> str | None:
    """The prompt line a successful probe earns, or None to say nothing."""
    if not tools:
        return None
    return (
        "Toolchain probe — available in this container: "
        + " ".join(tools)
        + " (others may exist; probe before assuming)."
    )


class _ToolchainProbeResult(NamedTuple):
    tools: list[str] | None
    status: str


async def _probe_toolchain(
    proxy: "_HarborShellProxy", logger: logging.Logger
) -> _ToolchainProbeResult:
    """Run the probe once at trial start.

    Best-effort like the snapshots either side of it: any failure degrades to
    no prompt line, never to a wrong one. The returned `status` is what makes
    that degradation legible after the fact: `tools` alone serializes to the
    same JSON `null` whether the container call raised or the probe ran
    cleanly and genuinely found none of the candidates -- `status` is the
    raw record of which of those actually happened, for
    environment_snapshot.json. (A run_harness timeout does not raise here --
    _exec_async catches it in either shape, the docker backend's RuntimeError
    or a raw asyncio.TimeoutError, and returns a normal error string -- so
    that case is disclosed inside the raw output captured by the
    non-exception branch below, not by the except.)
    """
    try:
        out = await proxy.run_harness(
            _TOOLCHAIN_PROBE_COMMAND, timeout=_TOOLCHAIN_PROBE_TIMEOUT_SEC
        )
    except Exception as e:
        logger.info(f"LittleCoderAgent: toolchain probe failed (non-fatal): {e}")
        return _ToolchainProbeResult(None, f"probe failed: {e}")
    tools = _parse_toolchain_probe(out)
    logger.info(f"LittleCoderAgent: toolchain probe -> {tools}")
    # Sliced, not the full string: on a broken image this is _format_output's
    # whole footer-and-all output, up to the ~48KB per-call cap -- more than
    # a status field needs to disclose "what actually happened" and needless
    # bulk in a JSON file meant for a quick post-mortem read.
    return _ToolchainProbeResult(
        tools, f"probe ran, raw output: {out[:2000]!r}"
    )


# Stated up front rather than left to the ShellSession tool description
# alone, which is demonstrably too weak: overfull-hbox's model never once
# passed `timeout`, so every long command ran under the 30s default.
#
# The not-killed sentence is measured, not assumed: on timeout, harbor
# terminates only the host-side docker-exec client -- reproduced against a
# live container, the in-container command survived and its statements past
# the timeout still ran.
_HARD_LIMITS_PARAGRAPH = (
    "Hard limits of this environment: each ShellSession call fails at its "
    "timeout (default 30s — pass `timeout: <seconds>` up to 600 for "
    "compiles/installs/long scripts) and everything the command printed is "
    "discarded with it. The command itself is not killed — it may still be "
    "running in the container and finish later, with none of its output "
    "ever shown — so inspect actual state (files, processes) before "
    "re-running anything non-idempotent. Output is capped at 200 "
    "lines / ~48KB per call. The container image is minimal: check which "
    "interpreters and tools exist (`command -v python3 perl gcc ...`) before "
    "designing an approach around one."
)


def _compose_prompt(
    prefix: str, task_block: str, notes: Sequence[str | None] = ()
) -> str:
    """Assemble run()'s prompt from its two fixed halves plus whatever the
    start-of-trial container probes produced.

    A seam, not decoration: the notes are only known after those probes run,
    many lines below where the prefix literal is written. Notes land between
    the prefix and TASK deliberately: appended after the closing "say 'done'"
    sentence, they would displace the model's last instruction.

    _HARD_LIMITS_PARAGRAPH is unconditional and lives here rather than at the
    call site so no caller can compose a prompt without it.
    """
    parts = [prefix, _HARD_LIMITS_PARAGRAPH, "\n\n"]
    for note in notes:
        if note:
            parts.append(note)
            parts.append("\n\n")
    parts.append(task_block)
    return "".join(parts)


def _fallback_timeout_info() -> dict:
    return {
        "cache_layout": None,
        "resolution": "fallback-default",
        "selected_task_toml": None,
        "base_timeout_sec": None,
        "multiplier": None,
        "effective_timeout_sec": DEFAULT_PROMPT_TIMEOUT_SEC,
    }


_MAX_CANDIDATES_RECORDED = 10


def _resolve_pkg_candidates(matches: list[Path], base_resolution: str):
    """Resolve >=1 surviving package-cache task.toml candidates by reading
    each one's [agent].timeout_sec instead of trusting mtime (every real
    package-cache task.toml on disk has an EPOCH mtime, so mtime carries no
    signal here at all -- see this module's cache-layout docstring).

    Returns (match_path, resolution, ambiguous, candidates_considered,
    candidate_task_tomls, candidate_timeouts_sec). Raises on a malformed
    candidate; the caller's broad except degrades that to the default
    timeout like any other unexpected shape.
    """
    timeouts = [float(tomllib.loads(p.read_text())["agent"]["timeout_sec"]) for p in matches]
    considered = len(matches)
    if considered == 1 or len(set(timeouts)) == 1:
        resolution = base_resolution if considered == 1 else f"{base_resolution}-agreed"
        return matches[0], resolution, False, considered, None, None
    # Disagreement: an under-estimate merely finalizes early; an
    # over-estimate risks the trial getting hard-killed by Harbor mid-write
    # -- so the minimum is the safer pick.
    min_idx = min(range(considered), key=lambda i: timeouts[i])
    return (
        matches[min_idx],
        base_resolution,
        True,
        considered,
        [str(p) for p in matches[:_MAX_CANDIDATES_RECORDED]],
        timeouts[:_MAX_CANDIDATES_RECORDED],
    )


def _resolve_trial_timeout_info(logs_dir: Path | None) -> dict:
    """Best-effort derivation of this trial's real Harbor-enforced timeout
    (task.toml's [agent].timeout_sec x the job's timeout_multiplier), so our
    internal deadline tracks Harbor's actual budget instead of a blind guess.

    Harbor gives a custom agent no direct/supported way to read this value --
    it's a private Trial attribute enforced purely via an external
    asyncio.wait_for() one level above the agent call. Every step here
    opportunistically reads files Harbor already writes/caches for other
    reasons (the trial's own config.json, and its local task-download cache).
    Falls back to DEFAULT_PROMPT_TIMEOUT_SEC on any failure -- missing files,
    unexpected shape, ambiguous cache match, etc. -- rather than raising.

    Returns a provenance dict (see _fallback_timeout_info for the shape) so a
    caller can log/record exactly which file was used and how it was found,
    instead of just the bare float.

    Harbor caches a task's files differently depending on which registry
    resolved the dataset, and the two layouts must NEVER be mixed within one
    lookup: a legacy name@version dataset (e.g. "terminal-bench@2.0") caches
    at <hash>/<task_name>/task.toml (task.toml's mtime is the real download
    time), while a newer org/name package dataset (e.g. "terminal-bench/
    terminal-bench-2-1") caches one level deeper, at packages/<org>/
    <task_name>/<content-hash>/task.toml -- and every package-cache task.toml
    on disk has an EPOCH mtime (Harbor's package extractor doesn't restore
    timestamps). A naive "union both globs, newest mtime wins" therefore
    always prefers the legacy file whenever a task name happens to exist in
    both layouts, silently reading the wrong dataset generation's timeout for
    any task that shares a name with a predecessor from the other layout. The
    trial's own config.json shape (task.name+ref vs task.path) tells us
    unambiguously which layout THIS trial belongs to -- use that instead of
    mtime.
    """
    try:
        if logs_dir is None:
            return _fallback_timeout_info()
        trial_dir = logs_dir.parent
        config = json.loads((trial_dir / "config.json").read_text())
        multiplier = float(config["timeout_multiplier"])
        task = config["task"]
        # A legacy name@version trial config's task dict has "path" (a bare
        # task name, e.g. "configure-git-webserver"); a newer org/name
        # package dataset's has "name" instead (namespaced, e.g.
        # "terminal-bench/some-task") and no "path" key at all.
        task_name = task.get("name")
        if task_name is not None:
            # Package shape. task_name is typically namespaced
            # ("<org>/<bare>"); the package cache directory is keyed by
            # (org, bare) separately, plus a content-hash directory equal to
            # task["ref"]'s sha256 hex when present -- giving an exact,
            # unambiguous path with no heuristics needed.
            org, _, bare = task_name.rpartition("/")
            org = org or "*"
            ref = task.get("ref") or ""
            match_path = None
            candidates_considered = 1
            ambiguous = False
            candidate_task_tomls = None
            candidate_timeouts_sec = None
            if ref.startswith("sha256:"):
                ref_hex = ref.split(":", 1)[1]
                candidate = HARBOR_TASK_CACHE / "packages" / org / bare / ref_hex / "task.toml"
                if org != "*" and candidate.exists():
                    match_path = candidate
                    resolution = "exact-ref"
            if match_path is None and ref.startswith("sha256:"):
                # Bare (un-namespaced) package name, so org is "*": the
                # content hash alone still identifies a real, unambiguous
                # directory regardless of which org it lives under. Outranks
                # the generation-mixing fallback glob below because the hash
                # match is exact.
                ref_hex = ref.split(":", 1)[1]
                glob_matches = sorted(
                    HARBOR_TASK_CACHE.glob(f"packages/*/{bare}/{ref_hex}/task.toml")
                )
                if glob_matches:
                    (
                        match_path, resolution, ambiguous, candidates_considered,
                        candidate_task_tomls, candidate_timeouts_sec,
                    ) = _resolve_pkg_candidates(glob_matches, "exact-ref-glob")
            if match_path is None:
                # Ref missing/mismatched (e.g. cache evicted and re-fetched
                # under a different content hash) -- fall back to a glob
                # restricted to the package layout ONLY. Never fall through
                # to the legacy glob here; that would silently read a
                # different dataset generation's metadata.
                pkg_matches = sorted(HARBOR_TASK_CACHE.glob(f"packages/{org}/{bare}/*/task.toml"))
                if not pkg_matches:
                    return _fallback_timeout_info()
                # Package-cache mtimes may all be epoch, and (when org=="*")
                # this glob also spans every org sharing the bare name -- so
                # mtime is never trusted here; _resolve_pkg_candidates reads
                # each candidate's real timeout_sec instead.
                (
                    match_path, resolution, ambiguous, candidates_considered,
                    candidate_task_tomls, candidate_timeouts_sec,
                ) = _resolve_pkg_candidates(pkg_matches, "fallback-glob")
            layout = "package"
        else:
            # Legacy shape.
            bare = task["path"].rsplit("/", 1)[-1]
            legacy_matches = [
                p for p in HARBOR_TASK_CACHE.glob(f"*/{bare}/task.toml")
                if "packages" not in p.relative_to(HARBOR_TASK_CACHE).parts
            ]
            if not legacy_matches:
                return _fallback_timeout_info()
            legacy_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            match_path = legacy_matches[0]
            resolution = "glob"
            layout = "legacy"
            candidates_considered = 1
            ambiguous = False
            candidate_task_tomls = None
            candidate_timeouts_sec = None

        toml_data = tomllib.loads(match_path.read_text())
        base_timeout_sec = float(toml_data["agent"]["timeout_sec"])
        effective_timeout_sec = base_timeout_sec * multiplier * DEADLINE_SAFETY_MARGIN
        info = {
            "cache_layout": layout,
            "resolution": resolution,
            "selected_task_toml": str(match_path),
            "base_timeout_sec": base_timeout_sec,
            "multiplier": multiplier,
            "effective_timeout_sec": effective_timeout_sec,
            "candidates_considered": candidates_considered,
            "ambiguous": ambiguous,
        }
        if ambiguous:
            info["candidate_task_tomls"] = candidate_task_tomls
            info["candidate_timeouts_sec"] = candidate_timeouts_sec
        return info
    except Exception:
        return _fallback_timeout_info()


def _resolve_trial_timeout_sec(logs_dir: Path | None) -> float:
    """Thin wrapper around _resolve_trial_timeout_info for callers that only
    need the effective timeout float."""
    return _resolve_trial_timeout_info(logs_dir)["effective_timeout_sec"]


def _derive_benchmark_label(logs_dir: Path | None) -> str:
    """Best-effort derivation of this trial's own dataset identity, for
    context.metadata["benchmark"] -- a hardcoded literal like
    "terminal_bench_2.0" would go stale as harbor_pilot.sh's default dataset
    changes (e.g. to TB2.1).

    Reads the same trial config.json _resolve_trial_timeout_info reads
    (confirmed on disk: both the package shape -- task.name/ref/source --
    and the legacy shape -- task.path/git_url/git_commit_id/source -- carry
    a "source" key giving the dataset identity directly, e.g.
    "terminal-bench/terminal-bench-2-1"). Legacy datasets can be re-pinned to
    different commits under the same name@version string, so a short
    git_commit_id is appended when present to disambiguate; package datasets
    are already uniquely identified by "source" alone (the content-hash
    "ref" is a task-cache lookup key, not part of the dataset's own
    identity).

    Never raises -- a label derivation failure must not fail a trial; falls
    back to "terminal_bench_unknown". This only changes what's reported in
    Harbor's own result metadata; the LITTLE_CODER_BENCHMARK env var
    (consumed by benchmark-profiles for model-profile selection) is a
    separate, unaffected mechanism and always stays "terminal_bench".
    """
    try:
        if logs_dir is None:
            return "terminal_bench_unknown"
        trial_dir = logs_dir.parent
        config = json.loads((trial_dir / "config.json").read_text())
        task = config["task"]
        source = task.get("source")
        if not source:
            return "terminal_bench_unknown"
        if task.get("name") is not None:
            # Package shape -- source alone is the dataset identity.
            return source
        # Legacy shape -- append a short commit id when present.
        commit = task.get("git_commit_id")
        if commit:
            return f"{source}@{commit[:12]}"
        return source
    except Exception:
        return "terminal_bench_unknown"


def _build_environment_snapshot(
    model: str,
    *,
    max_turns: int,
    ambient_max_turns_env: str | None,
    timeout_info: dict,
    toolchain: list[str] | None = None,
    toolchain_probe_status: str | None = None,
) -> dict:
    """Assemble the per-trial environment_snapshot.json payload:
    rpc_client.capture_environment_snapshot()'s existing pi-config
    introspection, plus the config values this adapter itself resolved --
    the active turn cap, the ambient env var seen at process entry, this
    process's own code identity, and the FULL timeout-provenance dict from
    _resolve_trial_timeout_info() (not just the effective float) so a
    reader can tell exactly which task.toml/cache-layout produced it.

    Split out from run() so it's unit-testable without spinning up a real
    PiRpc/environment. Never raises on its own logic; capture_environment_
    snapshot() already guarantees no-raise for its half.
    """
    snapshot = capture_environment_snapshot(model)
    snapshot["max_turns"] = max_turns
    snapshot["ambient_max_turns_env"] = ambient_max_turns_env
    snapshot["little_coder_version"] = _AGENT_VERSION
    snapshot["code_sha"] = _CODE_SHA
    snapshot["adapter_file"] = str(Path(__file__).resolve())
    snapshot["adapter_mtime"] = _ADAPTER_MTIME
    snapshot["timeout_provenance"] = timeout_info
    # toolchain_probe alone can't tell "found nothing" from "never
    # completed" -- both serialize as null. toolchain_probe_status is the
    # raw record (the exception message, or the probe's actual stdout) that
    # makes that distinction from the JSON file alone, without having to go
    # find the matching logger.info line in the trial log.
    snapshot["toolchain_probe"] = toolchain
    snapshot["toolchain_probe_status"] = toolchain_probe_status
    return snapshot

# Same line-dedup + ANSI-strip + truncation used by the TB 1.0 adapter so
# output-format consistency is preserved across benchmarks.
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MAX_LINES = 200
# Tiny relative to the byte caps below, which still catch anything past this
# that's actually large.
SMALL_OUTPUT_FLOOR_BYTES = 4 * 1024

# Byte caps, which the line cap alone cannot enforce: one 1MB line is one
# "line", so a `grep` hit on a single-line JSON file used to reach the model
# whole and blow the context window (this is the path that actually crashed a
# gpt2-codegolf trial). Head/tail split mirrors the 2:1 line ratio below.
MAX_BODY_HEAD_BYTES = 32 * 1024
MAX_BODY_TAIL_BYTES = 16 * 1024
# Pre-dedup gate: bounds the cost of split/dedup, which otherwise walk the
# whole output before any truncation runs.
MAX_RAW_HEAD_BYTES = 256 * 1024
MAX_RAW_TAIL_BYTES = 128 * 1024


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s or "")


def _format_size(n: int) -> str:
    """Human-readable byte count, matching pi's own truncation markers."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _cap_bytes_head_tail(s: str, head_bytes: int, tail_bytes: int) -> tuple[str, int]:
    """Keep the first `head_bytes` and last `tail_bytes` of `s`, joined by a
    marker. Returns (text, dropped_bytes).

    Cuts prefer a line boundary but must never require one -- the case this
    exists for is output with no newline in it at all. Byte-identical to the
    TypeScript capBytesHeadTail in .pi/extensions/shell-session/helpers.ts;
    test_format_output.py pins a shared multi-byte vector across the two.
    """
    buf = s.encode("utf-8")
    if len(buf) <= head_bytes + tail_bytes:
        return s, 0

    # Last newline in the head window, so the kept head is as long as the
    # budget allows; the first newline would legally cut at byte 10 of 32K.
    head_end = buf.rfind(b"\n", 0, head_bytes)
    if head_end < 0:
        head_end = head_bytes
        # Back off continuation bytes, keeping the shorter valid prefix.
        while head_end > 0 and buf[head_end] & 0xC0 == 0x80:
            head_end -= 1

    tail_start = buf.find(b"\n", len(buf) - tail_bytes)
    if tail_start >= 0:
        tail_start += 1
    else:
        tail_start = len(buf) - tail_bytes
        # Mirror image of the head side: skip *forward* off a continuation
        # byte. decode(errors="ignore") is not equivalent here -- it drops a
        # phantom partial character instead of skipping to the next real one.
        while tail_start < len(buf) and buf[tail_start] & 0xC0 == 0x80:
            tail_start += 1

    dropped = tail_start - head_end
    head = buf[:head_end].decode("utf-8")
    tail = buf[tail_start:].decode("utf-8")
    return f"{head}\n  [... {_format_size(dropped)} truncated ...]\n{tail}", dropped


def _compose_raw(stdout: str, stderr: str) -> str:
    """The stdout/stderr composition _format_output cleans, split out from
    _cleaned_output so the composition itself stays directly testable."""
    return (stdout or "") + (("\n[stderr]\n" + stderr) if stderr else "")


def _cleaned_output(stdout: str, stderr: str) -> str:
    """The exact ANSI-stripped, CR-normalized text _format_output's byte-cap
    pipeline operates on. Shared with _HarborShellProxy's overflow-capture
    path, which uploads it verbatim: two separate compositions could drift."""
    return _strip_ansi(_compose_raw(stdout, stderr)).replace("\r", "")


def _format_output(stdout: str, stderr: str, code: int, cwd: str, timed_out: bool) -> str:
    cleaned = _cleaned_output(stdout, stderr)
    raw_bytes = len(cleaned.encode("utf-8"))
    pre_capped, pre_dropped = _cap_bytes_head_tail(cleaned, MAX_RAW_HEAD_BYTES, MAX_RAW_TAIL_BYTES)
    lines = pre_capped.split("\n")
    # dedup
    deduped, last, dup = [], None, 0
    for ln in lines:
        if ln == last:
            dup += 1; continue
        if dup > 0:
            deduped.append(f"  [... {dup} duplicate line(s) collapsed ...]")
        dup = 0
        deduped.append(ln)
        last = ln
    if dup > 0:
        deduped.append(f"  [... {dup} duplicate line(s) collapsed ...]")
    # truncate
    truncated = False
    if len(deduped) > MAX_LINES and len("\n".join(deduped).encode("utf-8")) > SMALL_OUTPUT_FLOOR_BYTES:
        head, tail = MAX_LINES // 2, MAX_LINES // 4
        skipped = len(deduped) - head - tail
        deduped = deduped[:head] + [f"  [... {skipped} lines truncated ...]"] + deduped[-tail:]
        truncated = True
    body, post_dropped = _cap_bytes_head_tail(
        "\n".join(deduped), MAX_BODY_HEAD_BYTES, MAX_BODY_TAIL_BYTES
    )
    byte_capped = pre_dropped > 0 or post_dropped > 0
    # No "Full output:" line here, unlike the local subprocess backend: this
    # function is pure and synchronous (and shared with tb_adapter), while
    # producing a container-readable path needs an async upload. The harbor
    # proxy splices that line in afterwards -- see
    # _HarborShellProxy._capture_overflow.
    bits = [f"exit={code}", f"cwd={cwd}", f"timed_out={'true' if timed_out else 'false'}"]
    if truncated or byte_capped:
        bits.append("output_truncated=true")
        # Only alongside output_truncated: untruncated output is its own raw
        # size, and the existing footer shape stays byte-identical for normal
        # results.
        bits.append(f"raw_bytes={raw_bytes}")
    # Distinct from output_truncated=true, which also fires for the routine
    # 200-line cap. Overflow capture gates on this one so a long-but-small
    # pip/pytest log doesn't cost a docker-cp round-trip.
    if byte_capped:
        bits.append("byte_capped=true")
    bits.append("backend=harbor-env")
    footer = "[" + " ".join(bits) + "]"
    return f"{body}\n{footer}" if body else footer


def _wrap_command(command: str, cwd: str | None, sentinel: str) -> str:
    """Pure, module-level composition of the wrapper _exec_async prepends and
    appends around a caller's command -- split out specifically so a test
    can shell-parse the exact string that gets sent to env.exec() (see
    test_harbor_snapshot.py's `sh -n` parse tests). Without this seam, the
    2.1 bug (an unterminated `#` comment on _SNAPSHOT_COMMAND's last line
    silently swallowing everything _exec_async appended after it) was
    invisible to any test that only substring-matched _SNAPSHOT_COMMAND in
    isolation -- it never actually composed and parsed the real string.

    cwd=None omits the leading `cd` and the trailing `pwd` entirely (used
    when track_cwd=False): sound only for a command that never itself needs
    a starting cwd and never `cd`s in a way the caller needs reported back --
    true of every cwd=None caller today (both snapshot commands and the
    start-marker touch, which use absolute paths (/app, /tmp) throughout;
    the toolchain probe, whose only path resolution is `command -v`'s
    PATH search).
    """
    body = f"{{ {command} ; }} ; __rc=$? ; printf '\\n{sentinel}:%d:' $__rc"
    if cwd is None:
        return body
    return f"cd {cwd} 2>/dev/null; {body} ; pwd"


# Overflow capture: what _format_output's byte cap discards is pushed into the
# container as a file the model can cat back. env.upload_file() is an
# @abstractmethod on BaseEnvironment, so every harbor backend implements it --
# unlike exec(), whose base signature has no stdin parameter, this is a
# portable host->container transfer that needs no _wrap_command changes.
# Deliberately an order of magnitude under the TS local backend's 512MB, which
# this was first copied from: that one stages on the *host's* /tmp, while these
# files land in the container's /tmp and are never deleted for the life of the
# trial -- the same filesystem _SNAPSHOT_COMMAND refuses to snapshot into
# unless 500MB stays free. At 64MB, full utilization still cannot push /tmp
# under that reserve, so capture can't cost the trial its work-recovery net.
_OVERFLOW_BUDGET_BYTES = 64 * 1024 * 1024
# Prefix only: the directory gets a random suffix per proxy (see __init__),
# because a path known before the trial starts can be pre-planted as a symlink
# by the task principal, and both of docker's upload paths write as root.
_CONTAINER_OVERFLOW_DIR_PREFIX = "/tmp/.lc_shell-"
# Must fit inside run()'s fixed timeout+30 slack: a slow docker-cp has to
# degrade to Phase 1's disclosure, not eat the caller's whole margin.
_OVERFLOW_CAPTURE_TIMEOUT_SEC = 15


def _stage_overflow_file(fd: int, cleaned: str) -> None:
    """Write the capture's host-side staging file. Runs in a worker thread
    (see _capture_overflow_inner) so a multi-megabyte payload never blocks the
    event loop -- which asyncio.wait_for could not have preempted anyway.

    fdopen takes ownership of fd before the fchmod so that a raising fchmod
    (EPERM/EINVAL on a restrictive filesystem) still closes it on the way out.

    mkstemp's 0600 is widened because both the docker-cp fast path and the tar
    fallback (which force-extracts as uid=0/gid=0) land the file under an owner
    the agent's own container process may not be -- without this, a non-root
    task's shell gets Permission denied on the very file this feature exists to
    make readable. What that costs on the host is contained by the 0700
    staging directory the file lives in.
    """
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        os.fchmod(f.fileno(), 0o644)
        f.write(cleaned)


# Pinned to harbor 0.22.0's exact RuntimeError text (docker.py:682/741) --
# re-check this match on any harbor upgrade. timeout_sec is always int here,
# so no decimal-seconds alternate is needed.
_HARBOR_TIMEOUT_MSG_RE = re.compile(r"^Command timed out after \d+ seconds$")
_TIMEOUT_KILL_WARNING = (
    "WARNING: this command hit its {N}s timeout and its connection was killed. "
    "Any file it was mid-way through writing may now be HALF-WRITTEN, and any "
    "cleanup/restore logic at the end of a script may not have completed — re-verify "
    "(cat/wc/diff) any file it touched before trusting it. A compute-bound process may "
    "even still be running in the container (check with ps). If the command simply "
    "needs more time, re-run it with a larger `timeout` parameter (up to 600 seconds)."
)
# Distinct from _TIMEOUT_KILL_WARNING: run()'s bridge timeout (below) means only
# that _exec_async has not returned to the reader thread within timeout+30s --
# it says nothing about env.exec()'s own state. _exec_async may still be
# running, may be about to return normally, or may already have hit the real
# docker timeout above; we simply don't know from here. So this warning must
# not claim the connection was killed, and must not tell the model to re-run
# (a still-running command re-run now would duplicate its side effects).
_BRIDGE_TIMEOUT_WARNING = (
    "WARNING: this command hit its {N}s timeout and was cut off from this side -- "
    "its connection was NOT confirmed killed, and the command may still be running in "
    "the container. Any file it was mid-way through writing may now be HALF-WRITTEN, "
    "and any cleanup/restore logic at the end of a script may not have run — re-verify "
    "(cat/wc/diff) any file it touched before trusting it. Check whether it is still "
    "running (ps) before doing anything else; re-running it now risks starting a "
    "duplicate copy of a command that hasn't actually stopped."
)


class _HarborShellProxy:
    """Stateful shell proxy over harbor's BaseEnvironment.exec().

    harbor's env.exec() is stateless — each call is a fresh shell. To give
    little-coder's ShellSession tool the persistent-cwd / persistent-env
    semantics it expects, we track cwd in the proxy and prepend `cd <cwd>`
    to each command. `pwd` is echoed after the user command so we can
    capture the possibly-updated cwd for the next call.
    """

    def __init__(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop, logger: logging.Logger):
        self.env = environment
        self.loop = loop
        self.logger = logger
        self.cwd = "/app"  # TB 2.0 convention — overridden by first `pwd`
        # Serializes the actual env.exec() call across BOTH model-issued
        # commands (via run(), the sync thread-bridge entry point) and
        # harness-issued ones (via run_harness(), awaited directly on this
        # loop -- see its docstring).
        # Before this lock existed, only the reader thread ever called run(),
        # so there was nothing to serialize against; run_harness is the first
        # caller that can execute concurrently with it.
        #
        # The lock does NOT cover overflow capture's own upload_file call --
        # see _capture_overflow.
        self._exec_lock = asyncio.Lock()
        # Per-proxy-instance, i.e. per trial: a fresh proxy (fresh container)
        # starts with a fresh budget and a fresh overflow dir.
        self._overflow_bytes_written = 0
        # Guards the budget check and its reservation together; never held
        # across the upload itself.
        self._overflow_budget_lock = asyncio.Lock()
        # Unpredictable until this proxy exists, so nothing can be waiting at
        # the path -- see _CONTAINER_OVERFLOW_DIR_PREFIX.
        self._overflow_dir = f"{_CONTAINER_OVERFLOW_DIR_PREFIX}{uuid.uuid4().hex[:12]}"
        # Host-side staging dir, created on first capture: mkdtemp is 0700, so
        # the 0644 files inside it are still unreachable to other users of a
        # shared harness machine.
        self._host_stage_dir: str | None = None

    async def _capture_overflow(self, cleaned: str) -> str | None:
        """Upload the full untruncated output into the container so a later
        ShellSession call can cat it back, mirroring the TS local backend's
        temp-file behavior.

        Never raises and never blocks past _OVERFLOW_CAPTURE_TIMEOUT_SEC: on
        any failure or timeout it degrades to returning None, leaving
        output_truncated=true/raw_bytes= as the only disclosure -- exactly
        Phase 1's behavior. Capture is a bonus; it must never make a result
        worse than it was before this existed.

        Runs outside _exec_lock deliberately (see run_harness's docstring):
        the upload neither reads nor writes self.cwd, so there is nothing
        here for the lock to protect, and holding it would stall unrelated
        execs behind a docker-cp.
        """
        nbytes = len(cleaned.encode("utf-8"))
        async with self._overflow_budget_lock:
            if self._overflow_bytes_written + nbytes > _OVERFLOW_BUDGET_BYTES:
                # Deliberately not shaped like the real "Full output: <path>"
                # line it replaces: a naive split on that prefix would hand the
                # model this sentence as a path and waste a turn catting it.
                return "(full output not saved: per-session overflow-file budget exhausted)"
            # Reserved before the upload rather than added after it: a
            # model-issued and a harness-issued capture can overlap, and both
            # would otherwise clear a check neither had yet paid for.
            self._overflow_bytes_written += nbytes
        try:
            return await asyncio.wait_for(
                self._capture_overflow_inner(cleaned),
                timeout=_OVERFLOW_CAPTURE_TIMEOUT_SEC,
            )
        except Exception as e:
            # Catches the wait_for timeout too (asyncio.TimeoutError is an
            # Exception subclass): a slow docker-cp must degrade the same way
            # a failed one does.
            async with self._overflow_budget_lock:
                # Bytes that never landed must not stay reserved, or one
                # flaky upload permanently shrinks the trial's budget.
                self._overflow_bytes_written -= nbytes
            self.logger.warning(f"LittleCoderAgent: overflow capture failed: {e}")
            return None

    async def _capture_overflow_inner(self, cleaned: str) -> str:
        if self._host_stage_dir is None:
            self._host_stage_dir = tempfile.mkdtemp(prefix="lc-shell-overflow-")
        container_path = f"{self._overflow_dir}/{uuid.uuid4().hex[:12]}.out"
        fd, host_tmp = tempfile.mkstemp(dir=self._host_stage_dir, suffix=".log")
        try:
            await asyncio.to_thread(_stage_overflow_file, fd, cleaned)
            # The container-side directory is never mkdir'd here: harbor's
            # docker backend falls back to a tar stream that mkdir -p's the
            # target as root, so at worst the first capture of a trial costs
            # one extra round-trip.
            await self.env.upload_file(host_tmp, container_path)
        finally:
            os.unlink(host_tmp)
        return f"Full output: {container_path}"

    async def _exec_async(self, command: str, timeout: int, track_cwd: bool = True) -> str:
        """track_cwd=False (used by run_harness -- see its docstring) skips
        reading/writing self.cwd entirely: no leading `cd`, no trailing
        `pwd`, no assignment back to self.cwd.

        track_cwd=True (the model-command path, via run()) holds
        self._exec_lock across the full read(self.cwd)-exec-write(self.cwd)
        triple: the wrap is built from self.cwd, the exec happens, and any
        new cwd parsed from the trailing `pwd` is written back to self.cwd,
        all before the lock is released. This closes a race where a model
        command that `cd`s could interleave with a concurrent exec (e.g. the
        snapshot's run_harness call) and leave self.cwd pointing somewhere
        wrong for every subsequent command -- previously the wrap was built,
        and self.cwd assigned, both outside the lock.

        The lock is released (via `async with` exit) before formatting the
        result; the except branches below read self.cwd purely for display
        after that release, which is harmless since a failed exec never
        wrote a new cwd.
        """
        sentinel = f"__LC_END_{uuid.uuid4().hex[:8]}__"
        try:
            async with self._exec_lock:
                wrapped = _wrap_command(command, self.cwd if track_cwd else None, sentinel)
                result = await self.env.exec(command=wrapped, timeout_sec=timeout)
                out = result.stdout or ""
                err = result.stderr or ""
                # Peel sentinel to recover exit code + (if tracked) new cwd
                marker = out.rfind(sentinel + ":")
                code = result.return_code if result.return_code is not None else 0
                if marker >= 0:
                    tail = out[marker + len(sentinel) + 1:]
                    parts = tail.split(":", 1)
                    try: code = int(parts[0])
                    except (ValueError, IndexError): pass
                    if track_cwd and len(parts) > 1:
                        cwd_line = parts[1].lstrip("\r\n").split("\n")
                        if cwd_line and cwd_line[0].strip():
                            self.cwd = cwd_line[0].strip()
                    out = out[:marker].rstrip()
        except asyncio.TimeoutError:
            # Dead for the docker backend today (harbor swallows this and
            # raises RuntimeError instead -- see below), kept for other
            # backends/future harbor versions where it may still fire.
            warning = _TIMEOUT_KILL_WARNING.format(N=timeout)
            return _format_output("", f"command timed out\n{warning}", -1, self.cwd, True)
        except Exception as e:
            if _HARBOR_TIMEOUT_MSG_RE.fullmatch(str(e)):
                # The real timeout path for docker -- see asyncio.TimeoutError branch above.
                warning = _TIMEOUT_KILL_WARNING.format(N=timeout)
                return _format_output("", f"env.exec error: {e}\n{warning}", -1, self.cwd, True)
            return _format_output("", f"env.exec error: {e}", -1, self.cwd, False)
        result_str = _format_output(out, err, code, self.cwd, False)
        # byte_capped=true, not output_truncated=true: the latter also fires
        # for the plain 200-line cap, which is the common case for a coding
        # agent (pip install, pytest -v, git log) and has no discarded bytes
        # worth a docker-cp.
        #
        # Footer line only, never the whole string: a command's own output can
        # legitimately contain the literal flag text (the model catting back a
        # file it wrote, or one of this feature's own overflow files), and a
        # body-wide scan would upload needlessly and splice a lying "Full
        # output:" note into an untruncated result. Same hazard
        # _extract_exit_code's last-match rule already guards against.
        if "byte_capped=true" in result_str.rsplit("\n", 1)[-1]:
            note = await self._capture_overflow(_cleaned_output(out, err))
            if note:
                body, _, footer = result_str.rpartition("\n")
                result_str = f"{body}\n{note}\n{footer}" if body else f"{note}\n{footer}"
        return result_str

    def run(self, command: str, timeout: int) -> str:
        """Sync entry point called by PiRpc's reader thread."""
        fut = asyncio.run_coroutine_threadsafe(self._exec_async(command, timeout), self.loop)
        try:
            return fut.result(timeout=timeout + 30)
        except concurrent.futures.TimeoutError as e:
            warning = _BRIDGE_TIMEOUT_WARNING.format(N=timeout)
            return _format_output("", f"shell proxy error: {e}\n{warning}", -1, self.cwd, True)
        except Exception as e:
            return _format_output("", f"shell proxy error: {e}", -1, self.cwd, False)

    async def run_harness(self, command: str, timeout: int) -> str:
        """Harness-issued exec, awaited directly on the caller's event loop.

        NOT proxy.run(): run() is the sync bridge for PiRpc's reader THREAD --
        it calls asyncio.run_coroutine_threadsafe(...).result(), which blocks
        the calling thread until the coroutine finishes on `self.loop`. Calling
        run() from an asyncio task running ON that same loop would deadlock:
        fut.result() blocks the very loop that must run _exec_async to
        complete it. run_harness instead awaits _exec_async directly -- same
        loop, no thread bridge, no fut.result().

        This is also why every harness-issued path -- the deadline-snapshot
        task, the start-of-trial snapshot, the toolchain probe, the
        start-marker touch -- must use ONLY run_harness, never run().

        _exec_async's own _exec_lock still serializes this against
        model-issued commands (via run()), so a harness command and a model
        command never execute concurrently inside the container -- with one
        deliberate, disclosed exception: overflow capture's upload_file (see
        _capture_overflow) runs outside the lock and so can overlap either.
        That is safe because it never touches self.cwd -- the only state the
        lock exists to protect -- and its target path is absolute.

        Passes track_cwd=False: self.cwd is genuinely never read or written
        by a harness call (by construction, not by accident -- contrast the
        old docstring here, which claimed the same result but only held
        because every harness command happened to never `cd`). Sound because
        every harness command today either uses absolute paths (/app, /tmp)
        throughout (both snapshot instantiations, the start-marker touch) or
        resolves paths only through `command -v`'s PATH search (the
        toolchain probe), so none needs a starting cwd.
        """
        return await self._exec_async(command, timeout, track_cwd=False)

    def reset(self) -> str:
        self.cwd = "/app"
        return f"shell reset (cwd → /app)"

    def cleanup_overflow_staging(self) -> None:
        """Remove the private host-side staging directory (see
        _capture_overflow_inner), if this proxy ever created one.

        Split out so LittleCoderAgent.run()'s finally block is a single call
        whose own correctness needs no test of its own -- what's worth
        testing (None-safe, ignore_errors, actually removes a populated
        directory) lives here and is exercised directly, the same way
        _build_environment_snapshot is split out from run() for the same
        reason. Best-effort: cleanup must never fail the trial.
        """
        if self._host_stage_dir is not None:
            shutil.rmtree(self._host_stage_dir, ignore_errors=True)


def _resolve_token_usage(result_usage: dict, turn_count: int, stats: dict | None) -> dict:
    """Pick a token-usage source and compute the Harbor AgentContext mapping.

    Plan 7 (token-usage-tracking): pi computes full per-turn `Usage` and a
    session-cumulative `get_session_stats`, but Harbor's `result.json` fields
    were left null since nothing read either. `stats` (PiRpc.session_stats()'s
    return value) is preferred when available -- it's session-complete,
    including tool-result usage and compaction/branch-summary generation
    tokens (docs/rpc.md), which summing `turn_end` events alone misses.
    `result_usage` (PromptResult.usage, itself summed from `turn_end` events)
    is the fallback for when pi already died (process_exit/deadline paths) or
    is running an older build without `get_session_stats` -- exactly the
    trials where token counts matter most, per the plan's rejected
    alternatives.

    Pulled out as a standalone function (rather than inlined in run()) so it
    can be unit-tested without needing a live PiRpc/AgentContext/harbor
    environment -- pure function of already-fetched data.

    Returns a dict with "n_input_tokens", "n_cache_tokens", "n_output_tokens",
    "cost_usd" (Harbor AgentContext field values) plus "raw" (the flat
    input/output/cache_read/cache_write/cost breakdown) and "token_source"
    ("session_stats" | "event_sum" | "unavailable") for context.metadata.
    """
    def _num(v):
        # bool is an int subclass -- exclude it so a stray True doesn't
        # silently count as 1. Anything else non-numeric (a string, a dict,
        # None, ...) coerces to 0 rather than propagating a TypeError into
        # `input_tok + cache_read + cache_write` or `cost > 0` below, which
        # would otherwise fail an OTHERWISE-SUCCESSFUL Harbor trial purely
        # over token telemetry -- untrusted wire data (stats is pi's
        # get_session_stats response; result_usage is itself summed from
        # equally-untrusted turn_end events) must not do that.
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

    if isinstance(stats, dict) and isinstance(stats.get("tokens"), dict):
        tok = stats["tokens"]
        input_tok = _num(tok.get("input", 0))
        output_tok = _num(tok.get("output", 0))
        cache_read = _num(tok.get("cacheRead", 0))
        cache_write = _num(tok.get("cacheWrite", 0))
        cost = _num(stats.get("cost", 0))
        # token_source reflects that a session_stats payload WAS available
        # and had a "tokens" dict -- even if every field inside it turned out
        # to be non-numeric junk that _num() coerced to 0. Don't silently
        # reclassify that as event_sum/unavailable; the source was real.
        token_source = "session_stats"
    else:
        input_tok = _num(result_usage.get("input", 0))
        output_tok = _num(result_usage.get("output", 0))
        cache_read = _num(result_usage.get("cache_read", 0))
        cache_write = _num(result_usage.get("cache_write", 0))
        cost = _num(result_usage.get("cost", 0))
        # No turn_end ever fired (e.g. pi crashed before completing a single
        # turn) -- there is nothing meaningful to have summed, so this is
        # genuinely "unavailable" rather than a real (zero) event sum.
        # Distinct from a provider that legitimately reports zero usage for
        # turns that DID complete -- that case is still "event_sum", just
        # with zero-valued fields.
        token_source = "event_sum" if turn_count > 0 else "unavailable"

    return {
        "n_input_tokens": input_tok + cache_read + cache_write,
        "n_cache_tokens": cache_read,
        "n_output_tokens": output_tok,
        "cost_usd": cost if cost > 0 else None,
        "token_source": token_source,
        "raw": {
            "input": input_tok,
            "output": output_tok,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "cost": cost,
        },
    }


class LittleCoderAgent(BaseAgent):
    """Harbor (TB 2.0) adapter for little-coder (v0.1.0+ pi port)."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "little-coder"

    def version(self) -> str | None:
        return _AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        # little-coder runs pi on the host; no in-container setup needed.
        # The environment is used only for command proxying during run().
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or DEFAULT_MODEL
        session_id = f"hb-{uuid.uuid4().hex[:10]}"

        loop = asyncio.get_running_loop()
        proxy = _HarborShellProxy(environment, loop, self.logger)

        def tb_shell_handler(payload: dict) -> str:
            op = payload.get("op")
            if op == "run":
                return proxy.run(payload.get("command", ""), int(payload.get("timeout", 30)))
            if op == "reset":
                return proxy.reset()
            return f"Error: unknown ShellSession op '{op}'"

        # Two halves with a seam between them (see _compose_prompt): the
        # paragraphs that belong there are produced by container probes that
        # only run further down, after the environment is reachable.
        prompt_prefix = (
            "You are solving a Terminal-Bench 2.0 task inside a Linux container.\n"
            "The ONLY way to interact with the container is the ShellSession tool; "
            "its cwd persists between calls (tracked by the adapter). Any shell "
            "command is available — no command whitelist applies in this benchmark "
            "(Docker is the isolation boundary, not a shell filter), so use `cd`, "
            "compilers, daemonizing tools (setsid/nohup), package managers, etc. "
            "directly as needed, the same as any other terminal session.\n"
            "Default working directory is /app. `cd <path>` (standalone or as the "
            "first part of a `&&` chain) persists across every subsequent "
            "ShellSession call.\n"
            "File tools like Read/Write/Edit are NOT available — use shell commands "
            "(cat, sed -i, heredoc 'cat > file <<EOF') through ShellSession instead.\n\n"
            "Approach: briefly investigate the task first (inspect the relevant files, "
            "commands, or error output to understand what's actually being asked), "
            "form a short plan, then implement a quick first-pass solution rather "
            "than exhaustively enumerating options before writing anything. Get a "
            "working attempt in place early, verify it, and refine from there if "
            "time remains.\n\n"
            "Verification: when the instructions describe a specific client-side "
            "interface (e.g. `ssh`/`git clone user@host:...`, a particular port, "
            "protocol, or auth method), test through that exact path rather than a "
            "simplified local substitute — a substitute that skips part of the real "
            "flow (e.g. a local clone instead of an SSH one) can falsely appear to "
            "pass while the actual grading exercises a code path you never tested. "
            "If the instructions say you don't need to worry about some part of the "
            "setup (like login credentials), that usually still means the server "
            "side of it must be functional and reachable, just that you're not "
            "responsible for the client's half.\n\n"
        )
        prompt_task_block = (
            f"TASK:\n{instruction}\n\n"
            "When the task is complete, stop calling tools and say 'done'."
        )

        log_path = self.logs_dir / "little_coder.log"
        log_fh = log_path.open("w") if self.logs_dir else None

        # little_coder.log (above) is only written once prompt_and_collect
        # finally returns, so `tail -f` on it shows nothing for the whole
        # trial. This second file is flushed turn-by-turn via PiRpc's
        # on_event callback (see rpc_client.py's prompt_and_collect) so a
        # trial can actually be watched live, not just post-mortemed.
        live_log_path = self.logs_dir / "little_coder.live.log"
        live_log_fh = live_log_path.open("w") if self.logs_dir else None
        pending_text: list[str] = []
        # Turn boundary counter for the markers below. One prompt_and_collect
        # call can legitimately span several agent_end events (an
        # abort-then-recover continuation, an auto-retry, ...), so an
        # unqualified "=== agent_end ===" marker would stop a reader tailing
        # the live log at the FIRST agent_end and hide every turn after it.
        # Track turn boundaries explicitly instead -- agent_settled is the
        # only line a reader should treat as "the trial's turn is actually
        # done".
        turn_counter = 0

        def on_event(ev: dict) -> None:
            nonlocal turn_counter
            if live_log_fh is None:
                return
            t = ev.get("type")
            if t == "message_update":
                delta = ev.get("assistantMessageEvent", {})
                if delta.get("type") == "text_delta":
                    pending_text.append(delta.get("delta", ""))
                return
            if t == "tool_execution_start":
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.write(
                    f">> {ev.get('toolName', '')}({ev.get('args', {})})\n"
                )
                live_log_fh.flush()
            elif t == "tool_execution_end":
                res = ev.get("result", {})
                content = res.get("content", [])
                text = "\n".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
                live_log_fh.write(f"<< {preview_tool_result(text)}\n")
                live_log_fh.flush()
            elif t == "turn_end":
                # A turn that ended in a provider error, logged inline so the
                # live trajectory shows WHY a session stopped rather than
                # just stopping. Nothing else about turn_end is logged here;
                # PromptResult already aggregates the rest.
                msg = ev.get("message")
                if isinstance(msg, dict) and (
                    msg.get("errorMessage") or msg.get("stopReason") == "error"
                ):
                    # Drain the streamed text first, as every other marker
                    # branch does: text still buffered here belongs to the
                    # turn that just failed, and would otherwise surface
                    # after the line explaining why it stopped.
                    if pending_text:
                        live_log_fh.write("".join(pending_text) + "\n")
                        pending_text.clear()
                    live_log_fh.write(
                        f"=== turn error (stopReason="
                        f"{msg.get('stopReason')}): "
                        f"{msg.get('errorMessage') or '(no errorMessage)'} ===\n"
                    )
                    live_log_fh.flush()
            elif t == "auto_retry_start":
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.write(
                    f"=== pi auto-retry {ev.get('attempt')}/"
                    f"{ev.get('maxAttempts')} in {ev.get('delayMs')}ms: "
                    f"{ev.get('errorMessage', '')} ===\n"
                )
                live_log_fh.flush()
            elif t == "auto_retry_end":
                live_log_fh.write(
                    f"=== pi auto-retry {ev.get('attempt')} finished "
                    f"success={ev.get('success')} "
                    f"{ev.get('finalError', '')} ===\n"
                )
                live_log_fh.flush()
            elif t == "agent_start":
                turn_counter += 1
                live_log_fh.write(f"=== turn {turn_counter} start ===\n")
                live_log_fh.flush()
            elif t == "agent_end":
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.write(f"=== turn {turn_counter} agent_end (may continue) ===\n")
                live_log_fh.flush()
            elif t == "agent_settled":
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.write("=== agent_settled — trial turn complete ===\n")
                live_log_fh.flush()

        timeout_info = _resolve_trial_timeout_info(self.logs_dir)
        effective_timeout_sec = timeout_info["effective_timeout_sec"]
        if timeout_info["resolution"] == "fallback-default":
            self.logger.info(
                f"LittleCoderAgent: timeout resolved from fallback-default: "
                f"-> {effective_timeout_sec:.0f}s"
            )
        else:
            self.logger.info(
                "LittleCoderAgent: timeout resolved from "
                f"{timeout_info['cache_layout']}/{timeout_info['resolution']}: "
                f"{timeout_info['selected_task_toml']} "
                f"base={timeout_info['base_timeout_sec']:.0f}s "
                f"x{timeout_info['multiplier']} -> {effective_timeout_sec:.0f}s"
            )
        # deadline_epoch_ms/prompt_deadline are anchored below, after the
        # start-of-trial snapshot rather than here -- see that anchoring for
        # why. Nothing between here and there needs either value:
        # _build_environment_snapshot takes the resolved timeout_info dict,
        # not these, and _compute_snapshot_delay_sec below only needs the
        # plain effective_timeout_sec duration.

        # No turn cap: 40 was too tight (train-fasttext hit 41/40, one call
        # from its correct final fix), so it was raised to 80 -- which
        # mteb-leaderboard then hit at 80/80, again one investigation away
        # from a real answer (it had already built a correct top-30
        # leaderboard table and was fetching the required historical
        # snapshot when the cap fired). Any fixed N is a guess at how many
        # turns a task needs, and task difficulty varies enormously -- this
        # is whack-a-mole, not a fix. Wall-clock (finalize-warn's deadline
        # trigger, _resolve_trial_timeout_sec above) is the more principled
        # boundary: it's what Harbor itself actually enforces as the grading
        # limit, and matches the vendor's own published Terminal-Bench
        # methodology (flat 3h timeout, no turn limit at all). Local
        # inference has no per-token cost, so the downside of no cap -- a
        # genuinely stuck trial burning its full wall-clock budget instead
        # of aborting early and cheaply -- is an acceptable trade against
        # truncating trials that are still making real progress. Explicit 0
        # rather than leaving it unset, so the "no cap" choice reads as
        # deliberate, not an oversight -- and since rpc_client.py writes the
        # env var whenever max_turns is not None (not based on truthiness),
        # this 0 is guaranteed to actually reach the subprocess rather than
        # being silently skipped.
        #
        # Hoisted to a named local (rather than inlined as the kwarg below)
        # so this log line and the PiRpc kwarg read the exact same value and
        # can never diverge.
        max_turns = 0
        ambient_max_turns_env = os.environ.get("LITTLE_CODER_MAX_TURNS")
        self.logger.info(
            "LittleCoderAgent: config provenance "
            f"max_turns={max_turns} "
            f"ambient_LITTLE_CODER_MAX_TURNS={ambient_max_turns_env!r} "
            f"code_sha={_CODE_SHA} "
            f"adapter_file={__file__} adapter_mtime={_ADAPTER_MTIME}"
        )

        # One cheap probe of what the image actually ships, before the
        # deadline is anchored so its cost is not charged to the model. Run
        # ahead of the start-of-trial snapshot below for the same reason the
        # env-snapshot write right below is ordered ahead of it too -- see
        # that comment.
        toolchain = await _probe_toolchain(proxy, self.logger)

        # Per-trial environment_snapshot.json: best-effort, must never fail
        # a trial. Sits after the toolchain probe so it can record what that
        # found, but deliberately AHEAD of _snapshot_initial_state below --
        # unlike the probe, nothing this snapshot writes depends on that
        # snapshot's outcome, and _snapshot_initial_state is bounded at 75s
        # (including a docker-cp of up to 200MB) versus the probe's 10s.
        # environment_snapshot.json must land early enough to survive even a
        # mid-start termination, rather than depend on the initial-state
        # stage+download finishing first.
        # Still ahead of the PiRpc-construction try/except further below so
        # it (and the config-provenance log line) land even if PiRpc itself
        # fails to construct (e.g. PI_BIN missing) -- exactly the
        # diagnostics that failure needs most.
        if self.logs_dir:
            try:
                snapshot = _build_environment_snapshot(
                    model,
                    max_turns=max_turns,
                    ambient_max_turns_env=ambient_max_turns_env,
                    timeout_info=timeout_info,
                    toolchain=toolchain.tools,
                    toolchain_probe_status=toolchain.status,
                )
                (self.logs_dir / "environment_snapshot.json").write_text(
                    json.dumps(snapshot, indent=2, default=str)
                )
            except Exception as e:
                self.logger.warning(f"LittleCoderAgent: environment snapshot failed: {e}")

        # Start-of-trial snapshot of the task's pre-existing /app files.
        # Unconditional, and deliberately ahead of the deadline-snapshot gate
        # below rather than inside it: that gate's rationale is "this trial is
        # too short for a mid-run snapshot of the model's own work to be worth
        # scheduling", which does not transfer to a snapshot of originals. A
        # short trial destroys a task-provided input just as irrecoverably as
        # a long one, with less time left to notice, so nesting this inside
        # the `if` would exempt exactly the trials least able to recover.
        #
        # Also why the model's own deadline is anchored below, after this
        # await, rather than back where effective_timeout_sec was resolved:
        # anchoring it there would let a slow container-side copy here
        # silently eat into the model's nominal budget before its first
        # prompt is even sent. Anchoring after means the model always gets
        # the full effective_timeout_sec, regardless of how long staging
        # and downloading this snapshot took.
        #
        # Both this and the toolchain probe above go through run_harness,
        # which shares _exec_async's _exec_lock with every model-issued
        # command -- so gathering the two concurrently would not actually
        # run them in parallel inside the container, only add scheduling
        # complexity for no wall-clock benefit. Staying sequential keeps
        # this simple.
        initial_snapshot = await _snapshot_initial_state(
            proxy, environment, self.logs_dir, self.logger
        )

        # Composed here rather than right after the probe above:
        # _initial_snapshot_advertisement derives from the initial-state
        # snapshot's outcome, so composition has to wait for it -- and
        # `prompt` isn't read until prompt_with_error_retry far below, so
        # waiting costs nothing.
        prompt = _compose_prompt(
            prompt_prefix,
            prompt_task_block,
            [
                _toolchain_probe_note(toolchain.tools),
                _initial_snapshot_advertisement(initial_snapshot),
            ],
        )

        budget_start_epoch_ms = int(time.time() * 1000)
        deadline_epoch_ms = budget_start_epoch_ms + int(effective_timeout_sec * 1000)
        # The same instant as deadline_epoch_ms, on the monotonic clock the
        # error-retry loop measures against. Derived from one shared
        # effective_timeout_sec rather than re-read later, so a retry can
        # never outlive the deadline pi itself was handed above.
        prompt_deadline = time.monotonic() + effective_timeout_sec

        # Schedule the best-effort deadline snapshot. Skipped entirely below
        # SNAPSHOT_MIN_BUDGET_SEC (see that constant's comment); the
        # start-marker touch is itself failure-tolerated (a missing marker
        # just means find -newer fails later and the snapshot command
        # degrades to "no snapshot", same as any other failure mode here).
        snapshot_task: asyncio.Task | None = None
        snapshot_delay_sec = _compute_snapshot_delay_sec(effective_timeout_sec)
        if snapshot_delay_sec is not None:
            try:
                await proxy.run_harness(f"touch {SNAPSHOT_START_MARKER}", 10)
            except Exception as e:
                self.logger.info(
                    f"LittleCoderAgent: snapshot start-marker touch failed (non-fatal): {e}"
                )
            snapshot_task = asyncio.create_task(
                _snapshot_at_deadline(proxy, snapshot_delay_sec, self.logger)
            )
        else:
            self.logger.info(
                "LittleCoderAgent: skipping deadline snapshot -- trial budget "
                f"({effective_timeout_sec:.0f}s) is below the "
                f"{SNAPSHOT_MIN_BUDGET_SEC:.0f}s floor"
            )

        try:
            # PiRpc spawns pi --mode rpc and wires the shell proxy. The reader
            # thread invokes tb_shell_handler synchronously; the handler
            # bridges to this event loop via run_coroutine_threadsafe.
            rpc = await asyncio.to_thread(
                PiRpc,
                model=model,
                cwd=str(Path.cwd()),
                benchmark="terminal_bench",
                allowed_tools=DEFAULT_ALLOWED_TOOLS,
                session_id=session_id,
                tb_mode=True,
                max_turns=max_turns,
                tb_shell_handler=tb_shell_handler,
                env=_pi_env(
                    budget_start_epoch_ms=budget_start_epoch_ms,
                    deadline_epoch_ms=deadline_epoch_ms,
                    initial_snapshot=initial_snapshot,
                ),
            )
            try:
                # Retried in place on a provider-error completion rather than
                # called bare: a single errored completion used to end the
                # whole trial with most of the wall clock unspent (measured
                # at 62-81% unused across three of five failed TB2.1 trials).
                # Same rpc, same session -- see prompt_with_error_retry.
                retry_outcome = await asyncio.to_thread(
                    prompt_with_error_retry,
                    rpc,
                    prompt,
                    effective_timeout_sec,
                    on_event,
                    deadline=prompt_deadline,
                    log=self.logger.warning,
                )
                result = retry_outcome.result
                stop_reason = getattr(result, "stop_reason", "unknown")
                if log_fh:
                    # Distinguishes a crashed pi from a model that ran long;
                    # both used to look identical here.
                    log_fh.write(f"=== stop_reason: {stop_reason} ===\n")
                    if retry_outcome.n_error_retries or retry_outcome.error_message:
                        log_fh.write(
                            f"=== error retries: {retry_outcome.n_error_retries} "
                            f"(last error: {retry_outcome.error_message}) ===\n"
                        )
                    if retry_outcome.retry_exception:
                        log_fh.write(
                            f"=== retry raised (not propagated): "
                            f"{retry_outcome.retry_exception} ===\n"
                        )
                    log_fh.write(f"=== assistant text ===\n{result.assistant_text}\n\n")
                    for tc in result.tool_calls:
                        log_fh.write(f">> {tc['name']}({tc.get('args', {})})\n")
                        log_fh.write(
                            f"<< {preview_tool_result(tc.get('result_text', '') or '')}\n"
                        )
                    notes = rpc.notifications() if hasattr(rpc, "notifications") else []
                    if notes:
                        log_fh.write(f"\n=== pi notifications ({len(notes)}) ===\n")
                        for n in notes:
                            log_fh.write(f"[{n.get('notifyType','info')}] {n.get('message','')}\n")
                    stderr = rpc.stderr()
                    if stderr:
                        log_fh.write(f"\n=== pi stderr ===\n{stderr}\n")

                # Token usage: prefer pi's own cumulative get_session_stats
                # (includes tool-result usage and compaction/branch-summary
                # generation -- see PiRpc.session_stats' docstring) and fall
                # back to the per-turn event sum on PromptResult.usage when
                # stats are unavailable (pi already dead on the
                # process_exit/deadline paths, or an older pi build without
                # the get_session_stats command). Queried here, before
                # rpc.close(), because a closed/dead process can't answer it.
                stats = await asyncio.to_thread(rpc.session_stats)
                tokens = _resolve_token_usage(result.usage, result.turn_count, stats)

                # Harbor's AgentContext (harbor/models/agent/context.py) field
                # semantics are underspecified beyond their doc comments, so
                # this mapping is spelled out explicitly:
                #   n_input_tokens: field doc says "including cache" -> add
                #     both cache directions in, not just reads.
                #   n_cache_tokens: report cache READ hits (the reused-token
                #     savings) here; cache_write is mirrored into metadata
                #     separately since Harbor has no dedicated field for it.
                #   cost_usd: local providers (llama.cpp, omlx) report 0 --
                #     None reads as "not priced", matching how other Harbor
                #     agents report an unpriced run, rather than "free".
                context.n_input_tokens = tokens["n_input_tokens"]
                context.n_cache_tokens = tokens["n_cache_tokens"]
                context.n_output_tokens = tokens["n_output_tokens"]
                context.cost_usd = tokens["cost_usd"]

                context.metadata = {
                    "stop_reason": stop_reason,
                    # Both always present, not conditionally, so the field set
                    # in result.json stays constant across trials: a run that
                    # never errored reads as 0/"" rather than as a missing key
                    # indistinguishable from an older adapter build.
                    "n_error_retries": retry_outcome.n_error_retries,
                    "error_message": retry_outcome.error_message,
                    # A retry that raised is turned into a normal return by
                    # prompt_with_error_retry, so this field is the only place
                    # a harness fault reaches result.json at all.
                    "retry_exception": retry_outcome.retry_exception,
                    "n_tool_calls": len(result.tool_calls),
                    "n_turns": result.turn_count,
                    "n_compactions": result.compaction_events,
                    "n_notifications": len(rpc.notifications()) if hasattr(rpc, "notifications") else 0,
                    "little_coder_version": self.version(),
                    # Read from the trial's own config.json -- the pilot's
                    # dataset varies by run (TB2.0 vs TB2.1), so a hardcoded
                    # literal would silently go stale.
                    "benchmark": _derive_benchmark_label(self.logs_dir),
                    "token_usage": tokens["raw"],
                    "token_source": tokens["token_source"],
                }
            finally:
                await asyncio.to_thread(rpc.close, 3)
        except Exception as e:
            self.logger.error(f"LittleCoderAgent run failed: {e}")
            if log_fh:
                log_fh.write(f"\nAGENT ERROR: {e}\n")
            raise
        finally:
            # The snapshot task must never outlive run() -- cancel it here
            # regardless of how run() is exiting (normal completion, timeout,
            # or exception above).
            if snapshot_task is not None:
                snapshot_task.cancel()
                try:
                    await snapshot_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            # Per-capture files are unlinked as they're uploaded, but the
            # private staging directory itself otherwise outlives the trial
            # -- one empty 0700 dir leaked per trial that ever byte-capped,
            # forever, on the shared harness host.
            proxy.cleanup_overflow_staging()
            if log_fh:
                log_fh.flush()
                log_fh.close()
            if live_log_fh:
                # pending_text is only flushed inline on tool_execution_start
                # / agent_end. A trial that ends on "deadline" or
                # "process_exit" (no agent_end event -- routine: finalize-warn
                # itself notes trials regularly end this way) would otherwise
                # lose whatever trailing assistant text was still buffered.
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.flush()
                live_log_fh.close()
