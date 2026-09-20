"""Tests for the two container-snapshot mechanisms in _HarborShellProxy /
LittleCoderAgent.run() in little_coder_agent.py -- the deadline snapshot of
what the model changed, and the start-of-trial snapshot of the task's
pre-existing files:

  - the bounded, atomically-staged snapshot shell command itself
  - _build_snapshot_command's two instantiations: the deadline one byte-for-byte
    unchanged from the literal it replaced (the whole safety argument for
    templating a command already running in real trials), the start-of-trial one
    differing in exactly its find predicate, publish path and stage prefix
  - the start-of-trial snapshot's stage-then-download wiring: the target
    directory it must create itself (harbor's docker download_dir does not),
    the container-side copy it must leave behind, and its three-way
    succeeded/partial/refused outcome decision
  - what that outcome is then allowed to tell the model
    (_initial_snapshot_advertisement), including the restore-restraint
    sentence that keeps a deadline-pressed model from overwriting its own
    finished solution with the pristine originals
  - _wrap_command() actually composing a parseable shell script -- both for
    a harness call's cwd=None form and a model call's cd/pwd-tracking form
    -- verified by actually feeding the composed string to `sh -n`/`bash -n`
    rather than merely substring-matching pieces of it. This is the
    regression test for an unterminated trailing `#` comment on
    _SNAPSHOT_COMMAND's last line silently swallowing everything _exec_async
    appended after it -- a bug that every existing substring-matching test
    missed, because none of them ever actually composed and parsed the real
    string.
  - the start-of-trial toolchain probe that runs alongside them, and the
    prompt assembly both it and the static hard-limits paragraph feed
  - _compute_snapshot_delay_sec()'s scheduling arithmetic (incl. the
    short-task skip edge case)
  - _HarborShellProxy.run_harness() staying on the caller's event loop
    (never routing through asyncio.run_coroutine_threadsafe(), which is
    exactly the sync thread-bridge path that would deadlock a same-loop
    asyncio task)
  - _exec_lock actually serializing a model-issued command (via the sync
    run() thread-bridge, exercised from a real background thread since
    calling it from the loop's own thread would itself deadlock) against a
    harness-issued run_harness() call, and run_harness's track_cwd=False
    never mutating proxy.cwd even when the underlying exec's stdout happens
    to look like it echoed back a pwd line

Requires the `harbor` package (only installed in harbor's own uv-tool venv,
not the plain system Python these other benchmark tests run under) purely
because little_coder_agent.py imports harbor.agents.base at module load
time; none of this file's actual assertions touch Harbor itself. Same
scaffolding convention as test_harbor_adapter_timeout.py.
"""
import asyncio
import inspect
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "harbor_adapter"))
import little_coder_agent as lca  # noqa: E402


def _logger() -> logging.Logger:
    return logging.getLogger("test-harbor-snapshot")


class _RecordingEnv:
    """Fake harbor BaseEnvironment: records every command it was asked to
    exec and returns a canned success result."""

    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_sec: int | None = None, **kwargs) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout="", stderr="", return_code=0)


class _OverlapTrackingEnv:
    """Fake environment that records the max number of concurrently-active
    exec() calls, to detect whether two callers' commands ever overlapped
    inside the container.

    fake_pwd_line, if set, fabricates a `{sentinel}:0:{fake_pwd_line}` tail
    on the response to any command that does NOT itself end in ` ; pwd`
    (i.e. a track_cwd=False / harness-issued composition, which never asks
    for one) -- simulating an environment that echoes back something that
    *looks* like a pwd line even though none was requested, so a test can
    confirm run_harness's track_cwd=False guard (not merely the absence of a
    real trailing `pwd` in the command) is what keeps self.cwd untouched.
    """

    def __init__(self, hold_sec: float = 0.1, fake_pwd_line: str | None = None):
        self.hold_sec = hold_sec
        self.fake_pwd_line = fake_pwd_line
        self.current = 0
        self.max_concurrent = 0
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_sec: int | None = None, **kwargs) -> SimpleNamespace:
        self.commands.append(command)
        self.current += 1
        self.max_concurrent = max(self.max_concurrent, self.current)
        try:
            await asyncio.sleep(self.hold_sec)
        finally:
            self.current -= 1
        stdout = ""
        if self.fake_pwd_line is not None and not command.rstrip().endswith("pwd"):
            m = re.search(r"(__LC_END_\w+__):", command)
            if m:
                stdout = f"\n{m.group(1)}:0:{self.fake_pwd_line}\n"
        return SimpleNamespace(stdout=stdout, stderr="", return_code=0)


# ── 1. Snapshot command construction ────────────────────────────────────────

def test_snapshot_command_has_bounded_caps_and_staged_publish():
    """Every safety cap and the atomic-publish mechanism must actually be
    present in the composite command -- this is the command that literally
    runs in the container, so a typo here silently defeats one of the
    bounds."""
    cmd = lca._SNAPSHOT_COMMAND
    # per-file size cap
    assert "-size -10M" in cmd
    # freshness filter keyed to the start marker
    assert "-newer /tmp/.lc-start" in cmd
    assert lca.SNAPSHOT_START_MARKER in cmd
    # aggregate file-count cap
    assert "head -z -n 500" in cmd
    # aggregate byte cap: 200MB
    assert "209715200" in cmd
    # free-space reserve: 500MB
    assert "524288000" in cmd
    # staged-then-atomically-published, not written in place
    assert "STAGE=/tmp/.lc-snapshot.stage.$$" in cmd
    assert 'mv "$STAGE" /tmp/.lc-snapshot' in cmd
    assert lca.SNAPSHOT_PUBLISH_PATH in cmd
    # internal timeout + outer stage-glob cleanup for the kill-mid-copy case
    assert "timeout 20 sh -c" in cmd
    assert "rm -rf /tmp/.lc-snapshot.stage.*" in cmd
    # refuses rather than partially copies when oversize
    assert "set -e" in cmd
    # 2.3: publish is gated on a non-empty candidate list, so a zero-match
    # find (or a deleted start marker) refuses rather than publishing an
    # empty snapshot directory
    assert '[ -s "$STAGE/.list" ]' in cmd


def test_snapshot_command_last_line_has_no_trailing_inline_comment():
    """Regression pin for 2.1: a trailing `#` comment (with no newline after
    it) on _SNAPSHOT_COMMAND's last line is exactly what silently swallowed
    everything _exec_async/_wrap_command appended after it, producing a
    command that never parsed. Guard against re-introducing one."""
    last_line = lca._SNAPSHOT_COMMAND.rsplit("\n", 1)[-1]
    assert "#" not in last_line, (
        f"_SNAPSHOT_COMMAND's last line contains a trailing inline comment "
        f"with no newline after it -- this swallows everything appended "
        f"after the constant when it's wrapped (2.1): {last_line!r}"
    )


# ── 1b. The composed command must actually shell-parse ─────────────────────
# (not just substring-match -- substring matching is exactly what let 2.1's
# unterminated-comment bug ship undetected)

_SHELLS_TO_TRY = ("sh", "bash")


@pytest.mark.parametrize("shell", _SHELLS_TO_TRY)
def test_composed_snapshot_command_parses_under_sh_n(shell):
    """The model-command (track_cwd=True) composition of _SNAPSHOT_COMMAND,
    fed to a real shell's syntax checker -- this is the actual regression
    test for 2.1: `sh -n`/`bash -n` reported "unexpected end of file", rc=2,
    every single time before the fix."""
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not found on PATH")
    composed = lca._wrap_command(lca._SNAPSHOT_COMMAND, "/app", "__LC_END_test__")
    result = subprocess.run([shell, "-n"], input=composed, text=True, capture_output=True)
    assert result.returncode == 0, (
        f"{shell} -n rejected the composed snapshot command "
        f"(rc={result.returncode}): {result.stderr!r}"
    )
    assert result.stderr == ""


@pytest.mark.parametrize("shell", _SHELLS_TO_TRY)
def test_composed_harness_command_parses_under_sh_n(shell):
    """Same, for the track_cwd=False composition (2.2) -- run_harness's
    actual call shape: no leading `cd`, no trailing `pwd`."""
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not found on PATH")
    composed = lca._wrap_command(lca._SNAPSHOT_COMMAND, None, "__LC_END_test__")
    result = subprocess.run([shell, "-n"], input=composed, text=True, capture_output=True)
    assert result.returncode == 0, (
        f"{shell} -n rejected the composed harness (track_cwd=False) command "
        f"(rc={result.returncode}): {result.stderr!r}"
    )
    assert result.stderr == ""


# ── 1c. Command template: the deadline instance must not have moved, and
#      the start-of-trial instance must differ only where intended ─────────

# _SNAPSHOT_COMMAND's literal exactly as it shipped, before it became an
# instantiation of _build_snapshot_command. Frozen here rather than re-derived
# from the template, which is the only way this pins anything.
_SNAPSHOT_COMMAND_AS_SHIPPED = (
    "timeout 20 sh -c '\n"
    "  set -e\n"
    "  STAGE=/tmp/.lc-snapshot.stage.$$\n"
    "  rm -rf \"$STAGE\" && mkdir -p \"$STAGE\"\n"
    "  # candidate list: files under /app changed since trial start, per-file <10M\n"
    "  find /app -xdev -maxdepth 3 -type f -size -10M -newer /tmp/.lc-start -print0 2>/dev/null \\\n"
    "    | head -z -n 500 > \"$STAGE/.list\"           # aggregate file-count cap\n"
    "  TOTAL=$(du -cb --files0-from=\"$STAGE/.list\" 2>/dev/null | tail -1 | cut -f1)\n"
    "  FREE=$(df -B1 --output=avail /tmp | tail -1)\n"
    "  # non-empty candidate list AND aggregate byte cap 200MB AND leave >=500MB free space reserve\n"
    "  if [ -s \"$STAGE/.list\" ] && [ \"${TOTAL:-0}\" -le 209715200 ] && [ \"${FREE:-0}\" -ge $((TOTAL + 524288000)) ]; then\n"
    "    xargs -0 -a \"$STAGE/.list\" cp --parents -t \"$STAGE\" 2>/dev/null || true\n"
    "    rm -f \"$STAGE/.list\"\n"
    "    rm -rf /tmp/.lc-snapshot && mv \"$STAGE\" /tmp/.lc-snapshot   # atomic publish\n"
    "  else\n"
    "    rm -rf \"$STAGE\"                                             # refuse oversize or empty\n"
    "  fi\n"
    "' ; rm -rf /tmp/.lc-snapshot.stage.* 2>/dev/null"
)


def test_deadline_instantiation_is_byte_identical_to_the_shipped_command():
    """The whole safety argument for templating a command that already runs in
    real trials: the deadline snapshot's emitted string did not change."""
    assert lca._SNAPSHOT_COMMAND == _SNAPSHOT_COMMAND_AS_SHIPPED


def test_initial_instantiation_drops_the_freshness_filter():
    """At trial start "every file under /app" is exactly "every pre-existing
    file" -- the scope this snapshot exists to preserve. A -newer clause would
    also match nothing here: this runs before run() touches the start
    marker, so find would error on a path that doesn't exist yet."""
    cmd = lca._INITIAL_SNAPSHOT_COMMAND
    assert "-newer" not in cmd
    assert lca.SNAPSHOT_START_MARKER not in cmd
    assert "find /app -xdev -maxdepth 3 -type f -size -10M -print0" in cmd


def test_initial_instantiation_publishes_to_its_own_path():
    cmd = lca._INITIAL_SNAPSHOT_COMMAND
    assert lca.INITIAL_SNAPSHOT_PUBLISH_PATH == "/tmp/.lc-initial"
    assert f'mv "$STAGE" {lca.INITIAL_SNAPSHOT_PUBLISH_PATH}' in cmd
    # Never the deadline snapshot's published copy: the two have different
    # scopes and different lifetimes, and either overwriting the other loses
    # the one the model or a post-mortem actually wanted.
    assert lca.SNAPSHOT_PUBLISH_PATH not in cmd


def test_initial_instantiation_clears_its_publish_path_before_staging():
    """A refused or crashed run must never let the probe count stale content
    left at the publish path by an earlier invocation -- that would make a
    refusal look identical to a real success. The clear has to be its own
    statement outside the shared template, since that template only clears
    the publish path on its own success branch."""
    cmd = lca._INITIAL_SNAPSHOT_COMMAND
    clear_idx = cmd.index(f"rm -rf {lca.INITIAL_SNAPSHOT_PUBLISH_PATH} ;")
    stage_idx = cmd.index(f"STAGE={lca.INITIAL_SNAPSHOT_STAGE_PREFIX}")
    assert clear_idx < stage_idx


def test_initial_instantiation_stages_and_cleans_up_under_its_own_prefix():
    """Parameterizing the stage prefix without its cleanup glob leaks one
    stage dir per timeout kill under a name nothing reaps; reusing the
    deadline prefix instead has each command's cleanup line reaping the
    other's in-progress stage dir."""
    cmd = lca._INITIAL_SNAPSHOT_COMMAND
    assert f"STAGE={lca.INITIAL_SNAPSHOT_STAGE_PREFIX}.$$" in cmd
    assert f"rm -rf {lca.INITIAL_SNAPSHOT_STAGE_PREFIX}.* 2>/dev/null" in cmd
    assert lca.SNAPSHOT_STAGE_PREFIX not in cmd


def test_initial_instantiation_keeps_every_shared_cap():
    """The caps are the reason the start snapshot reuses this command instead
    of inventing a second copy discipline."""
    cmd = lca._INITIAL_SNAPSHOT_COMMAND
    for fragment in (
        "timeout 20 sh -c",
        "set -e",
        "-size -10M",
        f"head -z -n {lca.SNAPSHOT_MAX_FILES}",
        "209715200",
        "524288000",
        '[ -s "$STAGE/.list" ]',
    ):
        assert fragment in cmd, fragment


def test_initial_snapshot_command_last_line_has_no_trailing_inline_comment():
    """Same 2.1 regression pin as the deadline command: the last line gets
    _exec_async's epilogue appended directly onto it."""
    last_line = lca._INITIAL_SNAPSHOT_COMMAND.rsplit("\n", 1)[-1]
    assert "#" not in last_line, last_line


@pytest.mark.parametrize("shell", _SHELLS_TO_TRY)
@pytest.mark.parametrize("cwd", [None, "/app"], ids=["harness-form", "model-form"])
def test_composed_initial_snapshot_command_parses_under_sh_n(shell, cwd):
    """The start-of-trial command composes a file-count probe onto the shared
    template, so it needs its own parse check -- substring assertions are
    exactly what let the 2.1 syntax error ship."""
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not found on PATH")
    composed = lca._wrap_command(lca._INITIAL_SNAPSHOT_COMMAND, cwd, "__LC_END_test__")
    result = subprocess.run([shell, "-n"], input=composed, text=True, capture_output=True)
    assert result.returncode == 0, f"{shell} -n rejected it: {result.stderr!r}"
    assert result.stderr == ""


# ── 1d. Three-way outcome decision (pure helpers) ──────────────────────────

@pytest.mark.parametrize(
    "rc, file_count, outcome, download",
    [
        (0, 1, "succeeded", True),
        (0, 12, "succeeded", True),
        (0, 499, "succeeded", True),
        # At/over the file-count cap: the copy really is truncated, and rc is
        # 0 either way -- reporting this as "succeeded" is the specific lie
        # this split exists to prevent.
        (0, 500, "partial", True),
        (0, 700, "partial", True),
        # Published nothing: aggregate over 200MB, free-space reserve unmet,
        # or no candidate files. Still rc=0, and nothing to download.
        (0, 0, "refused", False),
        # No probe line at all -- not the same as zero files.
        (0, None, "failed", False),
        (None, None, "failed", False),
        # rc is deliberately NOT branched on (see the function's own
        # docstring): the wrapped command's last statement is always its
        # trailing cleanup `rm`, which exits 0 regardless of what the
        # staging step itself did, so a nonzero rc here carries no signal.
        # A real file count -- meaning the probe's own printf did run --
        # takes precedence over whatever rc claims.
        (2, 3, "succeeded", True),
        (1, 0, "refused", False),
    ],
)
def test_classify_initial_snapshot(rc, file_count, outcome, download):
    result = lca._classify_initial_snapshot(rc, file_count)
    assert result.outcome == outcome
    assert result.download is download
    assert result.message.startswith(outcome)


def test_classify_initial_snapshot_never_reports_a_bare_attempt():
    """An unconditional "attempted" is what let the 2.1 syntax error (rc=2,
    every trial) run undetected; against a whole-tree scope a bare
    "succeeded" is the same failure one level up."""
    for rc, count in ((0, 3), (0, 500), (0, 0), (0, None), (2, 3)):
        assert "attempted" not in lca._classify_initial_snapshot(rc, count).message


def test_parse_file_count_tolerates_padded_wc_output():
    """BSD `wc -l` pads its count with leading spaces; GNU's doesn't."""
    assert lca._parse_initial_snapshot_file_count(
        "lc-initial-files=       7\n[exit=0 cwd=/app backend=harbor-env]"
    ) == 7


def test_parse_file_count_missing_probe_line_is_none_not_zero():
    """None ("the probe never ran") and 0 ("it ran and published nothing")
    take different branches in _classify_initial_snapshot, so they must not
    collapse here."""
    assert lca._parse_initial_snapshot_file_count(
        "[exit=0 cwd=/app backend=harbor-env]"
    ) is None
    assert lca._parse_initial_snapshot_file_count("lc-initial-files=0") == 0


def test_parse_file_count_takes_the_last_match():
    """A command's own output can contain a line shaped like the probe's --
    the real one is always last."""
    out = "lc-initial-files=99\nlc-initial-files=4\n[exit=0 cwd=/app]"
    assert lca._parse_initial_snapshot_file_count(out) == 4


# ── 1e. Stage + download wiring ────────────────────────────────────────────

class _InitialSnapshotEnv:
    """Fake environment for _snapshot_initial_state.

    exec() answers the stage command with the file-count line the real
    container's probe would print. download_dir mimics `docker compose cp
    service:SRC/. DEST` by refusing when DEST does not already exist -- that
    is the real Docker behaviour (harbor's docker download_dir never creates
    the target; only download_dir_with_exclusions' base implementation does),
    and it's what turns a missing mkdir into a test failure here instead of a
    silently swallowed no-op on every real trial.
    """

    def __init__(self, file_count: int | None = 3, return_code: int = 0,
                 download_error: Exception | None = None, download_delay_sec: float = 0.0):
        self.file_count = file_count
        self.return_code = return_code
        self.download_error = download_error
        self.download_delay_sec = download_delay_sec
        self.commands: list[str] = []
        self.downloads: list[tuple[str, str]] = []

    async def exec(self, command: str, timeout_sec: int | None = None, **kwargs) -> SimpleNamespace:
        self.commands.append(command)
        stdout = ""
        if lca._INITIAL_SNAPSHOT_COUNT_PREFIX in command and self.file_count is not None:
            stdout = f"{lca._INITIAL_SNAPSHOT_COUNT_PREFIX}{self.file_count}\n"
        return SimpleNamespace(stdout=stdout, stderr="", return_code=self.return_code)

    async def download_dir(self, source_dir: str, target_dir) -> None:
        self.downloads.append((source_dir, str(target_dir)))
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(
                f"docker cp: destination {target_dir} does not exist"
            )
        if self.download_delay_sec:
            await asyncio.sleep(self.download_delay_sec)
        if self.download_error is not None:
            raise self.download_error
        Path(target_dir, "app").mkdir(exist_ok=True)
        Path(target_dir, "app", "input.tex").write_text("original bytes")


def _run_initial_snapshot(env, logs_dir, logger=None):
    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        return await lca._snapshot_initial_state(
            proxy, env, logs_dir, logger or _logger()
        )

    return asyncio.run(scenario())


def test_initial_snapshot_stages_then_downloads_into_the_trial_dir(tmp_path):
    """The headline path: one stage command, then a download into
    logs_dir/initial_state/ -- whose creation is this adapter's job, not
    download_dir's."""
    env = _InitialSnapshotEnv(file_count=3)
    _run_initial_snapshot(env, tmp_path)

    assert len(env.commands) == 1
    assert "STAGE=/tmp/.lc-initial.stage.$$" in env.commands[0]
    assert env.downloads == [
        (lca.INITIAL_SNAPSHOT_PUBLISH_PATH, str(tmp_path / lca.INITIAL_SNAPSHOT_DIR_NAME))
    ]
    assert (tmp_path / "initial_state" / "app" / "input.tex").read_text() == "original bytes"


def test_initial_snapshot_creates_the_target_dir_before_downloading(tmp_path):
    """Pins the mkdir specifically: the fake download_dir raises exactly as
    `docker cp SRC/. DEST` does when DEST is missing, and the caller swallows
    every exception -- so without the mkdir this feature would produce
    nothing on every trial while looking fine."""
    env = _InitialSnapshotEnv(file_count=3)
    logs_dir = tmp_path / "trial" / "nested"  # not created by anyone else
    _run_initial_snapshot(env, logs_dir)

    assert (logs_dir / "initial_state").is_dir()
    assert env.downloads, "download never ran"


def test_initial_snapshot_keeps_the_container_side_copy(tmp_path):
    """Deliberate: the published copy is the model's in-container restore
    source for the rest of the trial, so nothing may delete it once the
    host-side download has succeeded."""
    env = _InitialSnapshotEnv(file_count=3)
    _run_initial_snapshot(env, tmp_path)

    # A post-download cleanup would need a second exec.
    assert len(env.commands) == 1
    publish_rms = [
        line for line in env.commands[0].splitlines()
        if f"rm -rf {lca.INITIAL_SNAPSHOT_PUBLISH_PATH} " in line
    ]
    # Two: the unconditional pre-stage clear (this run's own leftover-content
    # guard), and the atomic publish's own pre-clean on the same line as the
    # mv that immediately replaces it. Neither is a post-download cleanup.
    assert len(publish_rms) == 2
    assert any(f'mv "$STAGE" {lca.INITIAL_SNAPSHOT_PUBLISH_PATH}' in line for line in publish_rms)


def test_initial_snapshot_refused_skips_the_download(tmp_path, caplog):
    """A refusal publishes nothing, so downloading would just raise. It is
    logged as "refused", not as a failure and not as a success."""
    env = _InitialSnapshotEnv(file_count=0)
    with caplog.at_level(logging.INFO):
        _run_initial_snapshot(env, tmp_path)

    assert env.downloads == []
    assert not (tmp_path / "initial_state").exists()
    assert any("refused" in r.message for r in caplog.records)


def test_initial_snapshot_partial_still_downloads_and_says_so(tmp_path, caplog):
    env = _InitialSnapshotEnv(file_count=lca.SNAPSHOT_MAX_FILES)
    with caplog.at_level(logging.INFO):
        _run_initial_snapshot(env, tmp_path)

    assert len(env.downloads) == 1
    assert any("partial" in r.message for r in caplog.records)


def test_initial_snapshot_ignores_rc_when_a_real_file_count_is_present(tmp_path, caplog):
    """rc is never trustworthy here (see _classify_initial_snapshot's own
    docstring: the wrapped command's last statement is always its harmless
    trailing cleanup, so rc is ~0 regardless of what staging actually did).
    A real, parsed file count -- meaning the probe's own printf genuinely
    ran -- must win over whatever rc claims, not be second-guessed by it."""
    env = _InitialSnapshotEnv(file_count=3, return_code=2)
    with caplog.at_level(logging.INFO):
        _run_initial_snapshot(env, tmp_path)

    assert len(env.downloads) == 1
    assert any("succeeded" in r.message for r in caplog.records)


def test_initial_snapshot_no_probe_line_fails_regardless_of_rc(tmp_path, caplog):
    """The one genuine failure signal: the probe's own printf never ran at
    all, so there is no file count to trust -- rc is logged for visibility
    only, never branched on."""
    env = _InitialSnapshotEnv(file_count=None, return_code=0)
    with caplog.at_level(logging.INFO):
        _run_initial_snapshot(env, tmp_path)

    assert env.downloads == []
    assert any("failed" in r.message and "rc=0" in r.message for r in caplog.records)


def test_initial_snapshot_download_failure_is_non_fatal(tmp_path):
    """Insurance must never take the trial down with it."""
    env = _InitialSnapshotEnv(file_count=3, download_error=RuntimeError("docker cp exploded"))
    _run_initial_snapshot(env, tmp_path)  # must not raise


def test_initial_snapshot_hang_degrades_within_its_timeout(tmp_path, monkeypatch):
    """A wedged docker-cp must not wedge the trial: the download's own
    timeout turns the hang into a bounded wait. The outcome-preserving half
    of this same scenario is test_snapshot_outcome_survives_a_hung_download."""
    monkeypatch.setattr(lca, "_INITIAL_SNAPSHOT_DOWNLOAD_TIMEOUT_SEC", 0.05)
    env = _InitialSnapshotEnv(file_count=3, download_delay_sec=60)

    started = time.monotonic()
    _run_initial_snapshot(env, tmp_path)
    assert time.monotonic() - started < 5


def test_initial_snapshot_without_a_logs_dir_never_touches_the_container():
    """No per-trial dir means nowhere to put the download; don't pay for the
    stage either."""
    env = _InitialSnapshotEnv(file_count=3)
    _run_initial_snapshot(env, None)

    assert env.commands == []
    assert env.downloads == []


def test_initial_snapshot_runs_before_and_outside_the_deadline_snapshot_gate():
    """Placement pin. The deadline snapshot is skipped below
    SNAPSHOT_MIN_BUDGET_SEC because a short trial has little of the model's
    own work worth recovering -- a rationale that does not transfer to the
    task's pre-existing files. Nesting the initial snapshot inside that gate
    would exempt exactly the trials with the least time to recover, and no
    unit test of the helper itself would notice."""
    src = textwrap.dedent(inspect.getsource(lca.LittleCoderAgent.run))
    call_line = next(
        line for line in src.splitlines() if "_snapshot_initial_state(" in line
    )
    gate_line = next(
        line for line in src.splitlines() if "if snapshot_delay_sec is not None:" in line
    )
    assert src.index(call_line) < src.index(gate_line), (
        "the initial snapshot must run before the deadline-snapshot gate"
    )
    # Same indentation as the gate itself: one level deeper would mean it sits
    # inside some conditional.
    assert len(call_line) - len(call_line.lstrip()) == len(gate_line) - len(gate_line.lstrip())
    assert call_line.lstrip().startswith(
        "initial_snapshot = await _snapshot_initial_state("
    )


def test_environment_snapshot_write_lands_before_the_initial_state_snapshot():
    """Placement pin, restoring a guarantee an earlier refactor narrowed:
    environment_snapshot.json must land before _snapshot_initial_state (bounded
    at 75s, including a docker-cp of up to 200MB) rather than after it, so it
    survives even a mid-start termination during that stage+download. Nothing
    the env-snapshot write needs -- max_turns, timeout_info, the toolchain
    probe result -- depends on _snapshot_initial_state's outcome."""
    src = textwrap.dedent(inspect.getsource(lca.LittleCoderAgent.run))
    write_line = next(
        line for line in src.splitlines() if '"environment_snapshot.json"' in line
    )
    snapshot_call_line = next(
        line for line in src.splitlines() if "await _snapshot_initial_state(" in line
    )
    assert src.index(write_line) < src.index(snapshot_call_line), (
        "environment_snapshot.json must be written before _snapshot_initial_state runs"
    )


# ── 1f. Start-of-trial toolchain probe + prompt assembly ───────────────────

@pytest.mark.parametrize("shell", _SHELLS_TO_TRY)
@pytest.mark.parametrize("cwd", [None, "/app"], ids=["harness-form", "model-form"])
def test_composed_toolchain_probe_parses_under_sh_n(shell, cwd):
    """Same parse check the snapshot commands get: this file has a history of
    a shell syntax error shipping past substring-only assertions."""
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not found on PATH")
    composed = lca._wrap_command(lca._TOOLCHAIN_PROBE_COMMAND, cwd, "__LC_END_test__")
    result = subprocess.run([shell, "-n"], input=composed, text=True, capture_output=True)
    assert result.returncode == 0, f"{shell} -n rejected it: {result.stderr!r}"
    assert result.stderr == ""


def test_toolchain_probe_command_asks_for_every_candidate():
    """The command and the accept-list are one tuple, so the probe can never
    ask about a tool _parse_toolchain_probe would then discard."""
    listed = lca._TOOLCHAIN_PROBE_COMMAND.split("for c in ", 1)[1].split(";", 1)[0]
    assert listed.split() == list(lca._TOOLCHAIN_CANDIDATES)


def test_toolchain_probe_command_last_line_has_no_trailing_inline_comment():
    """Same 2.1 regression pin as the snapshot commands: _exec_async's
    epilogue is appended directly onto the last line."""
    assert "#" not in lca._TOOLCHAIN_PROBE_COMMAND.rsplit("\n", 1)[-1]


def test_parse_toolchain_probe_normal_output():
    out = "python3 gcc make \n[exit=0 cwd=/app timed_out=false backend=harbor-env]"
    assert lca._parse_toolchain_probe(out) == ["python3", "gcc", "make"]


def test_parse_toolchain_probe_orders_by_candidate_list_and_dedupes():
    """Stable ordering: the line goes into a prompt, so it must not vary with
    whatever order the container's loop happened to print."""
    assert lca._parse_toolchain_probe("make gcc python3 gcc") == ["python3", "gcc", "make"]


def test_parse_toolchain_probe_filters_non_candidate_tokens():
    """The harness wrapper's own footer (and anything else a broken image
    prints) shares this string. Nothing outside the candidate list may reach
    the model's prompt as a "detected" tool."""
    out = "python3 rustc definitely-not-a-tool perl\n[exit=0 cwd=/app]"
    assert lca._parse_toolchain_probe(out) == ["python3", "perl"]


@pytest.mark.parametrize(
    "out",
    ["", "\n", "[exit=0 cwd=/app]", "sh: 1: Syntax error: bad for loop variable"],
    ids=["empty", "blank", "footer-only", "garbage"],
)
def test_parse_toolchain_probe_returns_none_not_empty_list(out):
    """None, never [] -- the caller omits the prompt line entirely rather
    than advertising an empty toolchain it cannot actually vouch for."""
    assert lca._parse_toolchain_probe(out) is None


def test_toolchain_note_names_the_tools_and_keeps_the_hedge():
    note = lca._toolchain_probe_note(["python3", "gcc"])
    assert "python3 gcc" in note
    # The probe asks about nine tools; the container has more. Dropping this
    # would turn a closed candidate list into a false exhaustive inventory.
    assert "others may exist" in note


@pytest.mark.parametrize("tools", [None, []], ids=["none", "empty"])
def test_toolchain_note_is_none_when_nothing_was_detected(tools):
    assert lca._toolchain_probe_note(tools) is None


def test_prompt_always_carries_the_hard_limits_paragraph():
    for notes in ([], [None], ["something"]):
        prompt = lca._compose_prompt("PREFIX\n\n", "TASK:\nx", notes)
        assert lca._HARD_LIMITS_PARAGRAPH in prompt


def _shell_session_timeout_constants() -> tuple[int, int]:
    """Read the ShellSession tool's own default/max timeout straight out of
    .pi/extensions/shell-session -- the only place that actually enforces
    them -- so the paragraph test below can't drift silently from the code
    it claims to pin. Regex, not an import: this is TypeScript, and there is
    no existing pattern in this repo for a Python test to load a TS module,
    so parsing the two literals out of source is the cheap alternative to
    either a shared JSON constants file or leaving the claim unverified.
    """
    ts_dir = lca._REPO_ROOT / ".pi" / "extensions" / "shell-session"
    helpers_src = (ts_dir / "helpers.ts").read_text()
    default_timeout = int(
        re.search(r"DEFAULT_TIMEOUT\s*=\s*(\d+)", helpers_src).group(1)
    )
    index_src = (ts_dir / "index.ts").read_text()
    max_timeout = int(
        re.search(r"Math\.min\(rawTimeout,\s*(\d+)\)", index_src).group(1)
    )
    return default_timeout, max_timeout


def test_hard_limits_paragraph_states_the_caps_the_harness_enforces():
    """Pins the numbers to the code that enforces them -- a stale prompt here
    is worse than none, since the model would trust it.

    All four numbers are now actually derived from the enforcing side: the
    200-line/48KB caps from this module's own MAX_LINES/MAX_BODY_*_BYTES
    (which _exec_async's formatting path applies), and the 30s/600s timeout
    bounds from .pi/extensions/shell-session's own source (the ShellSession
    tool description these numbers used to only echo, hand-typed, with no
    check that they still matched)."""
    para = lca._HARD_LIMITS_PARAGRAPH
    default_timeout, max_timeout = _shell_session_timeout_constants()
    assert f"default {default_timeout}s" in para
    assert f"up to {max_timeout}" in para
    assert f"{lca.MAX_LINES} lines" in para
    capped_kb = (lca.MAX_BODY_HEAD_BYTES + lca.MAX_BODY_TAIL_BYTES) // 1024
    assert f"{capped_kb}KB" in para


def test_prompt_splices_the_toolchain_line_before_the_task_block():
    """Position, not just presence: after the final "say 'done'" sentence it
    would displace the model's last instruction."""
    note = lca._toolchain_probe_note(["python3", "gcc"])
    prompt = lca._compose_prompt("PREFIX\n\n", "TASK:\ndo it", [note])

    assert note in prompt
    assert prompt.index(lca._HARD_LIMITS_PARAGRAPH) < prompt.index(note) < prompt.index("TASK:")
    assert prompt.startswith("PREFIX")
    assert prompt.endswith("TASK:\ndo it")


def test_prompt_omits_the_toolchain_line_when_the_probe_found_nothing():
    prompt = lca._compose_prompt(
        "PREFIX\n\n", "TASK:\ndo it", [lca._toolchain_probe_note(None)]
    )
    assert "Toolchain probe" not in prompt
    assert lca._HARD_LIMITS_PARAGRAPH in prompt
    assert prompt.endswith("TASK:\ndo it")


def test_probe_toolchain_returns_none_when_the_container_call_raises():
    """Best-effort: a probe failure costs the prompt line, never the trial --
    but the raw exception must still be recoverable from `status`, since
    `tools=None` alone is indistinguishable from a probe that ran clean and
    found nothing."""

    class _ExplodingProxy:
        async def run_harness(self, command, timeout):
            raise RuntimeError("exec exploded")

    result = asyncio.run(lca._probe_toolchain(_ExplodingProxy(), _logger()))
    assert result.tools is None
    assert "probe failed" in result.status
    assert "exec exploded" in result.status


def test_probe_toolchain_reads_stdout_and_ignores_a_nonzero_rc():
    """The loop's own rc is non-zero whenever the LAST candidate is absent,
    so rc must carry no weight here."""
    env = _RecordingEnv()

    async def scenario():
        proxy = lca._HarborShellProxy(env, asyncio.get_running_loop(), _logger())

        async def fake_exec(command, timeout, track_cwd=True):
            return "perl awk\n[exit=127 cwd=/app timed_out=false backend=harbor-env]"

        proxy._exec_async = fake_exec
        return await lca._probe_toolchain(proxy, _logger())

    result = asyncio.run(scenario())
    assert result.tools == ["perl", "awk"]
    assert "probe ran, raw output" in result.status


def test_environment_snapshot_records_the_probe_result_even_when_empty():
    """Post-mortem needs "probe found nothing" to be distinguishable from an
    adapter build that never probed at all -- toolchain_probe alone can't do
    that (both are JSON null), so toolchain_probe_status must carry the raw
    record of what actually happened."""
    info = lca._fallback_timeout_info()
    for toolchain in (["python3"], None):
        snapshot = lca._build_environment_snapshot(
            "llamacpp/x", max_turns=0, ambient_max_turns_env=None,
            timeout_info=info, toolchain=toolchain,
            toolchain_probe_status="probe ran, raw output: 'python3 '",
        )
        assert "toolchain_probe" in snapshot
        assert snapshot["toolchain_probe"] == toolchain
        assert snapshot["toolchain_probe_status"] == "probe ran, raw output: 'python3 '"

    # And the None/None case (probe never even attempted a status) still
    # serializes cleanly rather than raising.
    snapshot = lca._build_environment_snapshot(
        "llamacpp/x", max_turns=0, ambient_max_turns_env=None,
        timeout_info=info, toolchain=None, toolchain_probe_status=None,
    )
    assert snapshot["toolchain_probe"] is None
    assert snapshot["toolchain_probe_status"] is None


# ── 1g. What the outcome is allowed to tell the model ──────────────────────

# Asserted verbatim, deliberately not imported from the module under test: the
# failure this guards is the wording being softened, which a test sharing the
# module's own constant could never notice.
_RESTRAINT_SENTENCE = (
    "Restore from there only a file you believe you corrupted — never over "
    "your own completed solution."
)


@pytest.mark.parametrize("outcome", [None, "failed", "refused"])
def test_advertisement_is_silent_when_the_copy_may_not_be_there(outcome):
    """Pointing the model at a path that does not exist costs it turns for
    nothing -- strictly worse than never mentioning the copy."""
    arg = None if outcome is None else lca._InitialSnapshotOutcome(outcome, False, "msg")
    assert lca._initial_snapshot_advertisement(arg) is None


@pytest.mark.parametrize("outcome", ["succeeded", "partial"])
def test_advertisement_points_at_the_real_published_path(outcome):
    """Path comes from the constant the snapshot command itself publishes to,
    never a second hardcoded copy that could drift."""
    text = lca._initial_snapshot_advertisement(
        lca._InitialSnapshotOutcome(outcome, True, "msg")
    )
    assert f"{lca.INITIAL_SNAPSHOT_PUBLISH_PATH}/app/" in text
    # The stage command's `cp --parents` keeps the /app prefix, so the
    # example's app/ layout is the published copy's real one.
    assert f"{lca.INITIAL_SNAPSHOT_PUBLISH_PATH}/app/somefile" in text


@pytest.mark.parametrize("outcome", ["succeeded", "partial"])
def test_advertisement_keeps_the_restore_restraint_sentence(outcome):
    text = lca._initial_snapshot_advertisement(
        lca._InitialSnapshotOutcome(outcome, True, "msg")
    )
    assert _RESTRAINT_SENTENCE in text
    assert "read-only" in text


def test_advertisement_does_not_ask_for_routine_diffing():
    """Reactive only: it earns its turns when a specific file is suspected of
    being clobbered, not as a standing habit on every trial."""
    text = lca._initial_snapshot_advertisement(
        lca._InitialSnapshotOutcome("succeeded", True, "msg")
    )
    assert "If you ever suspect" in text
    for nudge in ("before you start", "first thing", "always diff", "routinely"):
        assert nudge not in text


def test_partial_advertisement_adds_the_absence_caveat():
    """A truncated copy must not read as evidence a missing file never
    existed."""
    text = lca._initial_snapshot_advertisement(
        lca._InitialSnapshotOutcome("partial", True, "msg")
    )
    assert f"{lca.SNAPSHOT_MAX_FILES}-file cap" in text
    assert "absence there does not mean the file didn't exist" in text
    assert _RESTRAINT_SENTENCE in text


def test_succeeded_advertisement_omits_the_absence_caveat():
    text = lca._initial_snapshot_advertisement(
        lca._InitialSnapshotOutcome("succeeded", True, "msg")
    )
    assert "-file cap" not in text


def test_snapshot_outcome_survives_a_failing_host_side_download(tmp_path):
    """The model is pointed at the CONTAINER-side copy, which a broken
    docker-cp neither removes nor invalidates. Discarding the stage outcome
    here would silence the recovery note over a problem the model can't
    see."""
    env = _InitialSnapshotEnv(file_count=3, download_error=RuntimeError("docker cp exploded"))
    outcome = _run_initial_snapshot(env, tmp_path)

    assert outcome is not None and outcome.outcome == "succeeded"
    assert lca._initial_snapshot_advertisement(outcome) is not None


@pytest.mark.parametrize(
    "file_count, expected",
    [(3, "succeeded"), (500, "partial"), (0, "refused"), (None, "failed")],
)
def test_snapshot_returns_the_classified_outcome(tmp_path, file_count, expected):
    env = _InitialSnapshotEnv(file_count=file_count)
    outcome = _run_initial_snapshot(env, tmp_path)
    assert outcome is not None and outcome.outcome == expected


def test_snapshot_returns_none_without_a_logs_dir():
    assert _run_initial_snapshot(_InitialSnapshotEnv(file_count=3), None) is None


def test_snapshot_outcome_survives_a_hung_download(tmp_path, monkeypatch):
    """A timed-out download must degrade the same way a raised one does
    (test_snapshot_outcome_survives_a_failing_host_side_download): the
    container-side stage already succeeded and the model can still reach it,
    so a slow docker-cp must not erase that outcome."""
    monkeypatch.setattr(lca, "_INITIAL_SNAPSHOT_DOWNLOAD_TIMEOUT_SEC", 0.05)
    env = _InitialSnapshotEnv(file_count=3, download_delay_sec=60)
    outcome = _run_initial_snapshot(env, tmp_path)
    assert outcome is not None and outcome.outcome == "succeeded"


def test_snapshot_returns_none_when_the_outer_backstop_fires(tmp_path, monkeypatch):
    """The download-specific timeout above is what normally degrades a hang
    without losing the outcome; this covers the other timeout -- the outer
    wait_for around the whole stage+download call, which exists only as a
    backstop for a bound that somehow doesn't fire on its own. Only
    _INITIAL_SNAPSHOT_TIMEOUT_SEC is shrunk here -- the download's own
    timeout stays at its real value, so it's the outer bound that fires
    first and cancels mid-download."""
    monkeypatch.setattr(lca, "_INITIAL_SNAPSHOT_TIMEOUT_SEC", 0.05)
    env = _InitialSnapshotEnv(file_count=3, download_delay_sec=60)
    assert _run_initial_snapshot(env, tmp_path) is None


def test_prompt_splices_the_advertisement_between_limits_and_task():
    """Same placement rule as the toolchain line: never after the closing
    "say 'done'" sentence."""
    ad = lca._initial_snapshot_advertisement(
        lca._InitialSnapshotOutcome("succeeded", True, "msg")
    )
    prompt = lca._compose_prompt("PREFIX\n\n", "TASK:\ndo it", [None, ad])

    assert ad in prompt
    assert prompt.index(lca._HARD_LIMITS_PARAGRAPH) < prompt.index(ad) < prompt.index("TASK:")
    assert prompt.endswith("TASK:\ndo it")


def test_prompt_omits_the_advertisement_when_nothing_was_staged():
    prompt = lca._compose_prompt(
        "PREFIX\n\n", "TASK:\ndo it", [None, lca._initial_snapshot_advertisement(None)]
    )
    assert "Recovery note" not in prompt
    assert lca._HARD_LIMITS_PARAGRAPH in prompt


# ── 1b. What the snapshot outcome tells pi's extensions (_pi_env) ──────────

def _outcome(name: str) -> lca._InitialSnapshotOutcome:
    return lca._InitialSnapshotOutcome(name, name in ("succeeded", "partial"), "msg")


@pytest.mark.parametrize("outcome", ["succeeded", "partial"])
def test_pi_env_publishes_the_outcome_when_a_copy_was_staged(outcome):
    """The extensions' only evidence that /tmp/.lc-initial is really there.
    The outcome string itself, not a flag: a reader has to be able to repeat
    the cap-truncation caveat that "partial" carries."""
    env = lca._pi_env(deadline_epoch_ms=1, initial_snapshot=_outcome(outcome))
    assert env["LITTLE_CODER_INITIAL_SNAPSHOT"] == outcome


@pytest.mark.parametrize("outcome", [None, "failed", "refused"])
def test_pi_env_stays_silent_when_no_copy_was_staged(outcome):
    """Unset, not a falsy value: an extension reading an empty string as
    "there is a copy" would point the model at a path that does not exist --
    the same cost _initial_snapshot_advertisement refuses to pay."""
    arg = None if outcome is None else _outcome(outcome)
    assert "LITTLE_CODER_INITIAL_SNAPSHOT" not in lca._pi_env(
        deadline_epoch_ms=1, initial_snapshot=arg
    )


@pytest.mark.parametrize("outcome", [None, "failed", "refused", "succeeded", "partial"])
def test_pi_env_gate_agrees_with_the_prompt_advertisement(outcome):
    """One gate, two consumers: the model must never be told about the copy
    in the prompt and not in the nudges, or the reverse."""
    arg = None if outcome is None else _outcome(outcome)
    advertised = lca._initial_snapshot_advertisement(arg) is not None
    in_env = "LITTLE_CODER_INITIAL_SNAPSHOT" in lca._pi_env(
        deadline_epoch_ms=1, initial_snapshot=arg
    )
    assert advertised == in_env


def test_pi_env_always_carries_the_deadline_and_permission_mode():
    env = lca._pi_env(deadline_epoch_ms=1700000000000, initial_snapshot=None)
    assert env["LITTLE_CODER_DEADLINE_EPOCH_MS"] == "1700000000000"
    assert env["LITTLE_CODER_PERMISSION_MODE"] == "accept-all"


def test_run_hands_pi_env_to_the_rpc_client():
    """Wiring pin. Every assertion above is on a pure helper, so a run()
    that built its own literal dict instead would pass all of them."""
    src = textwrap.dedent(inspect.getsource(lca.LittleCoderAgent.run))
    assert "env=_pi_env(" in src
    assert "initial_snapshot=initial_snapshot," in src


# ── 2. Scheduling arithmetic (_compute_snapshot_delay_sec) ─────────────────

def test_snapshot_delay_normal_budget_fires_lead_seconds_before_deadline():
    eff = 3600.0
    delay = lca._compute_snapshot_delay_sec(eff)
    assert delay == pytest.approx(eff - lca.SNAPSHOT_LEAD_SEC)


def test_snapshot_delay_clamped_to_zero_when_lead_exceeds_budget():
    """Budget is between the min-budget floor and the lead time -- the naive
    delay would be negative, so it's clamped to 0 (snapshot fires almost
    immediately) rather than skipped."""
    eff = 400.0
    assert eff >= lca.SNAPSHOT_MIN_BUDGET_SEC
    assert eff - lca.SNAPSHOT_LEAD_SEC < 0
    assert lca._compute_snapshot_delay_sec(eff) == 0.0


def test_snapshot_delay_skips_entirely_below_min_budget():
    """Short-task edge case: below SNAPSHOT_MIN_BUDGET_SEC the snapshot is
    skipped outright (None), not merely delay-clamped."""
    eff = lca.SNAPSHOT_MIN_BUDGET_SEC - 1.0
    assert lca._compute_snapshot_delay_sec(eff) is None


def test_snapshot_delay_at_exact_min_budget_boundary_is_not_skipped():
    eff = lca.SNAPSHOT_MIN_BUDGET_SEC
    assert lca._compute_snapshot_delay_sec(eff) is not None


# ── 3. _snapshot_at_deadline: fires via run_harness after its delay, and is
#      cleanly cancellable before it fires ──────────────────────────────────

def test_snapshot_at_deadline_fires_run_harness_after_delay():
    env = _RecordingEnv()

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        task = asyncio.create_task(lca._snapshot_at_deadline(proxy, 0.01, _logger()))
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())
    assert len(env.commands) == 1
    assert "STAGE=/tmp/.lc-snapshot.stage.$$" in env.commands[0]


def test_snapshot_at_deadline_cancelled_before_firing_never_execs():
    """Mirrors run()'s own finally: task.cancel() + suppressed
    CancelledError. A snapshot scheduled far in the future and cancelled
    immediately (as happens on early trial completion) must never touch the
    container."""
    env = _RecordingEnv()

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        task = asyncio.create_task(lca._snapshot_at_deadline(proxy, 10.0, _logger()))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    assert env.commands == []


# ── 4. run_harness stays on the caller's loop -- never the thread bridge ───

def test_run_harness_never_routes_through_run_coroutine_threadsafe(monkeypatch):
    """Pins the deadlock fix: run_harness must await the async exec path
    directly. Under the original (rejected) design, the snapshot task called
    proxy.run() -- which submits via asyncio.run_coroutine_threadsafe(...)
    and then blocks on fut.result(). Called from an asyncio task on the same
    loop, fut.result() would block that very loop, deadlocking it. Patching
    run_coroutine_threadsafe to blow up turns any regression back to that
    call path into a hard test failure instead of a hang."""

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "run_harness routed through run_coroutine_threadsafe -- this is "
            "the thread-bridge path that deadlocks when called from an "
            "asyncio task on the same loop"
        )

    monkeypatch.setattr(lca.asyncio, "run_coroutine_threadsafe", _must_not_be_called)

    env = _RecordingEnv()

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        return await proxy.run_harness("echo harness-only", 5)

    asyncio.run(scenario())
    assert len(env.commands) == 1


# ── 5. _exec_lock serializes model-issued (run()) and harness-issued
#      (run_harness()) commands so they never overlap inside the container ─

def test_exec_lock_serializes_run_and_run_harness():
    """Simulates PiRpc's reader thread (which calls the sync proxy.run()
    thread-bridge) issuing a model command concurrently with the snapshot
    task's run_harness() call on the loop itself. Before this lock existed
    there was nothing to serialize the two paths against each other; without
    it, two coroutines scheduled on the same loop that each `await
    asyncio.sleep(...)` mid-exec would interleave, and OverlapTrackingEnv's
    max_concurrent would show 2.

    Also (2.2) pins that run_harness's track_cwd=False never mutates
    proxy.cwd -- the fake env is set up to fabricate a bogus pwd-looking
    tail on any non-pwd-ending command (i.e. exactly the harness-issued
    composition), so a regression that let run_harness read/write self.cwd
    would show up here as proxy.cwd changing to that bogus value.
    """
    env = _OverlapTrackingEnv(hold_sec=0.15, fake_pwd_line="/bogus/should-not-be-used")
    thread_result: dict[str, str] = {}

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())

        def thread_target():
            thread_result["out"] = proxy.run("echo model-issued", 5)

        t = threading.Thread(target=thread_target)
        t.start()
        # Let the reader-thread's command actually start executing on the
        # loop before we submit the harness-issued one, so the two windows
        # genuinely overlap in time rather than trivially running back-to-back.
        await asyncio.sleep(0.05)
        harness_out = await proxy.run_harness("echo harness-issued", 5)
        t.join(timeout=5)
        assert not t.is_alive()
        return harness_out, proxy.cwd

    harness_out, cwd_after = asyncio.run(scenario())

    assert env.max_concurrent == 1, (
        f"exec calls overlapped (max_concurrent={env.max_concurrent}) -- "
        "_exec_lock failed to serialize model- and harness-issued commands"
    )
    assert len(env.commands) == 2
    assert any("echo model-issued" in c for c in env.commands)
    assert any("echo harness-issued" in c for c in env.commands)
    assert thread_result.get("out")  # sync run() returned formatted output
    assert harness_out
    assert cwd_after == "/app", (
        f"proxy.cwd changed to {cwd_after!r} -- run_harness's track_cwd=False "
        "must never read or write self.cwd, even when the underlying exec's "
        "stdout looks like it echoed back a pwd line"
    )


# ── 6. Overflow capture: pushing the untruncated output into the container ─
#
# Phase 1 capped ShellSession output at ~48KB and told the model nothing about
# where the discarded bytes went, because a host tmp path is unreachable from
# inside the container. Phase 2 uploads the full cleaned output through
# env.upload_file() (BaseEnvironment's portable host->container primitive) and
# points the model at the container-side path.

_OVERFLOW_STDOUT = "x" * 200_000  # one line, so only the byte cap can catch it


class _OverflowEnv:
    """Fake harbor BaseEnvironment for the overflow-capture tests.

    exec() returns `stdout` only for model-issued commands (the ones carrying
    an _exec_async sentinel) -- there is no longer any other exec the code
    issues (the mkdir bootstrap was removed; harbor's own tar-fallback mkdir
    -p's the target as root when the fast `cp` path needs it, so this fake
    never needs to simulate that side).

    upload_file() reads the host file's content and mode *during* the call:
    _capture_overflow_inner unlinks the tmp file in a finally before control
    returns, so a stub that only recorded the path would find nothing left to
    read by assertion time.
    """

    def __init__(self, stdout: str = "", stderr: str = "",
                 upload_error: Exception | None = None, upload_delay_sec: float = 0.0):
        self.stdout = stdout
        self.stderr = stderr
        self.upload_error = upload_error
        self.upload_delay_sec = upload_delay_sec
        self.commands: list[str] = []
        # (source_path, target_path, content_at_call_time, mode_at_call_time)
        self.uploads: list[tuple[str, str, str | None, int | None]] = []

    async def exec(self, command: str, timeout_sec: int | None = None, **kwargs) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout=self.stdout, stderr=self.stderr, return_code=0)

    async def upload_file(self, source_path, target_path: str) -> None:
        content, mode = None, None
        try:
            mode = os.stat(source_path).st_mode & 0o777
            with open(source_path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            pass
        self.uploads.append((str(source_path), target_path, content, mode))
        if self.upload_delay_sec:
            await asyncio.sleep(self.upload_delay_sec)
        if self.upload_error is not None:
            raise self.upload_error


def _run_exec(env, command: str = "echo hi", timeout: int = 5) -> str:
    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        return await proxy._exec_async(command, timeout)

    return asyncio.run(scenario())


def _host_tmp_paths(env) -> list[str]:
    return [src for src, _target, _content, _mode in env.uploads]


def test_byte_capped_output_uploads_the_full_content_and_points_at_it():
    """The headline case. Asserts all of: upload happens exactly once, the
    target is under this proxy's randomized container overflow dir, the
    uploaded bytes are the full untruncated cleaned output (not the capped
    body), the host file was widened to 0644 (mkstemp's 0600 would be
    unreadable to a non-root task's shell after docker cp lands it as root),
    and the note is spliced between the body and the footer rather than
    after it."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT)
    out = _run_exec(env)

    assert len(env.uploads) == 1
    _src, target, content, mode = env.uploads[0]
    assert target.startswith(lca._CONTAINER_OVERFLOW_DIR_PREFIX)
    assert target.endswith(".out")
    assert content == lca._cleaned_output(_OVERFLOW_STDOUT, "")
    assert len(content) == len(_OVERFLOW_STDOUT)  # the full output, not the capped body
    assert mode == 0o644, f"host tmp file was mode {oct(mode or 0)}, not 0644"

    lines = out.split("\n")
    assert lines[-1].startswith("[exit=0 ")
    assert "output_truncated=true" in lines[-1]
    assert "byte_capped=true" in lines[-1]
    assert lines[-2] == f"Full output: {target}"


def test_normal_output_never_uploads_anything():
    """Regression guard: capture must fire on truncation only, not on every
    single ShellSession call."""
    env = _OverflowEnv(stdout="hello\nworld\n")
    out = _run_exec(env)

    assert env.uploads == []
    assert "Full output:" not in out
    assert "output_truncated=true" not in out


def test_body_containing_the_truncation_flag_does_not_trigger_capture():
    """The gate reads the footer line only. A command's own output can
    legitimately contain the literal `byte_capped=true` -- e.g. the model
    catting back an earlier transcript, or one of this feature's own overflow
    files -- and a whole-string scan would upload needlessly and splice a
    lying "Full output:" note onto an untruncated result."""
    env = _OverflowEnv(stdout="[exit=0 cwd=/app timed_out=false byte_capped=true]\nreal output")
    out = _run_exec(env)

    assert env.uploads == []
    assert "Full output:" not in out
    assert "byte_capped=true" not in out.rsplit("\n", 1)[-1]


def test_line_only_truncation_does_not_trigger_capture():
    """Routine output that trips only the 200-line cap (pip install, pytest
    -v, git log) sets output_truncated=true but not byte_capped=true, and
    must not cost a docker-cp round trip -- that would make the common case
    pay for a feature built for the rare, pathological one."""
    many_short_lines = "\n".join(f"line {i}" for i in range(400))  # far under any byte cap
    env = _OverflowEnv(stdout=many_short_lines)
    out = _run_exec(env)

    assert env.uploads == []
    footer = out.rsplit("\n", 1)[-1]
    assert "output_truncated=true" in footer
    assert "byte_capped=true" not in footer
    assert "Full output:" not in out


def test_no_mkdir_bootstrap_and_dir_stays_stable_across_calls():
    """The mkdir bootstrap was removed entirely (harbor's own tar-fallback
    mkdir -p's the target as root when the fast `cp` path needs it), so no
    exec beyond the model's own wrapped command is ever issued, and every
    capture on one proxy instance targets the same per-proxy directory."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT)

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        first = await proxy._exec_async("echo one", 5)
        second = await proxy._exec_async("echo two", 5)
        return proxy, first, second

    proxy, _first, _second = asyncio.run(scenario())
    assert not any(c.strip().startswith("mkdir") for c in env.commands), (
        f"unexpected mkdir exec: {env.commands}"
    )
    assert len(env.uploads) == 2
    dirs = {target.rsplit("/", 1)[0] for _src, target, _content, _mode in env.uploads}
    assert dirs == {proxy._overflow_dir}


def test_cleanup_overflow_staging_removes_the_dir_run_relies_on_it_for():
    """Per-file cleanup (see test_host_tmp_file_is_always_unlinked below)
    unlinks each staged file as it's uploaded, but the private 0700
    directory those files lived in otherwise outlives the trial -- one
    leaked empty directory per trial that ever byte-capped, forever, on the
    shared harness host.

    This exercises LittleCoderAgent.run()'s actual cleanup call, not a
    re-implementation of it: run()'s finally block is (deliberately) just
    `proxy.cleanup_overflow_staging()`, so calling that same method here --
    rather than reaching in and calling shutil.rmtree directly on
    proxy._host_stage_dir -- is what would actually catch a regression if
    run()'s wiring or this method's own body ever drifted apart. Full
    end-to-end coverage of run() itself would need a real PiRpc/fake_pi
    harness, which is a heavier lift than this method's own logic warrants.
    """
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT)

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        await proxy._exec_async("echo one", 5)
        return proxy

    proxy = asyncio.run(scenario())
    assert proxy._host_stage_dir is not None
    assert os.path.isdir(proxy._host_stage_dir)
    assert os.listdir(proxy._host_stage_dir) == [], "a staged file leaked past its upload"

    proxy.cleanup_overflow_staging()
    assert not os.path.exists(proxy._host_stage_dir)


def test_cleanup_overflow_staging_is_a_safe_no_op_when_nothing_ever_overflowed():
    """A trial with no byte-capped output never creates a staging dir at
    all; run()'s unconditional finally call must not raise on that proxy."""
    env = _OverflowEnv(stdout="hello\nworld\n")

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        await proxy._exec_async("echo hi", 5)
        return proxy

    proxy = asyncio.run(scenario())
    assert proxy._host_stage_dir is None
    proxy.cleanup_overflow_staging()  # must not raise


def test_budget_exhaustion_stops_uploading_and_says_so(monkeypatch):
    """The per-trial ceiling exists to stop a pathological loop from filling
    the container's /tmp. Once it's hit the model is told the file wasn't
    saved rather than being pointed at a path that doesn't exist -- and the
    message must not itself look like a "Full output: <path>" line, or a
    naive prefix-split would hand a model this sentence as a path."""
    monkeypatch.setattr(lca, "_OVERFLOW_BUDGET_BYTES", 300_000)
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT)  # 200_000 bytes cleaned

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        first = await proxy._exec_async("echo one", 5)
        second = await proxy._exec_async("echo two", 5)
        return first, second

    first, second = asyncio.run(scenario())

    assert f"Full output: {lca._CONTAINER_OVERFLOW_DIR_PREFIX}" in first
    assert len(env.uploads) == 1, "second capture uploaded despite an exhausted budget"
    assert "Full output:" not in second
    assert "not saved" in second and "budget exhausted" in second
    assert "output_truncated=true" in second.rsplit("\n", 1)[-1]


def test_upload_failure_degrades_to_phase_1_disclosure():
    """Capture is a bonus on top of the truncation footer. If it fails the
    result must be exactly what it was before this feature existed -- never
    an exception, and never a "Full output:" line pointing at a file the
    upload didn't actually produce."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT, upload_error=RuntimeError("docker cp exploded"))
    out = _run_exec(env)

    assert "Full output:" not in out
    footer = out.rsplit("\n", 1)[-1]
    assert "output_truncated=true" in footer
    assert f"raw_bytes={len(_OVERFLOW_STDOUT)}" in footer
    (host_tmp,) = _host_tmp_paths(env)
    assert not os.path.exists(host_tmp)


def test_capture_timeout_degrades_the_same_way_and_returns_promptly(monkeypatch):
    """A docker cp that hangs must not eat run()'s fixed timeout+30 slack and
    turn a truncated-but-present result into a hard shell-proxy error. The
    real constant is pinned by the first assertion; the test itself then
    swaps in a short one, so a would-be 60s hang shows up as a fast,
    clean degrade instead of a 15s one."""
    assert lca._OVERFLOW_CAPTURE_TIMEOUT_SEC < 30, (
        "the capture timeout must fit inside run()'s timeout+30 slack"
    )
    monkeypatch.setattr(lca, "_OVERFLOW_CAPTURE_TIMEOUT_SEC", 0.05)
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT, upload_delay_sec=60)

    started = time.monotonic()
    out = _run_exec(env)
    elapsed = time.monotonic() - started

    assert elapsed < 5, f"a hung upload blocked the result for {elapsed:.1f}s"
    assert "Full output:" not in out
    footer = out.rsplit("\n", 1)[-1]
    assert "output_truncated=true" in footer
    assert f"raw_bytes={len(_OVERFLOW_STDOUT)}" in footer


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"upload_error": RuntimeError("boom")}],
    ids=["upload-succeeds", "upload-raises"],
)
def test_host_tmp_file_is_always_unlinked(kwargs):
    """The staging file lives on the *host* (the harness machine, shared
    across concurrent trials), so leaking one per capped command is a real
    disk leak. The unlink is in a finally; pin that it covers the failure
    path too."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT, **kwargs)
    _run_exec(env)

    paths = _host_tmp_paths(env)
    assert len(paths) == 1
    assert not os.path.exists(paths[0]), f"host tmp file {paths[0]} survived the call"
