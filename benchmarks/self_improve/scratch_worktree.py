"""Disposable git worktree for live candidate evaluation.

benchmarks/rpc_client.py binds REPO_ROOT = Path(__file__).parent.parent at
IMPORT time (rpc_client.py:29), and .pi/extensions/skill-inject/index.ts
resolves skills/tools from its own import.meta.url. Both mean the ONLY way
to make a candidate's proposed text reach a live agent is to run the harness
as a SUBPROCESS from a tree whose files are the candidate's -- pointing
cwd= at a scratch dir from an in-process PiRpc does nothing.

Design choices, all deliberate:
- `git worktree add --detach`: detached HEAD means it is structurally
  impossible to collide with any of this repo's real branches (several are
  checked out in sibling worktrees at any given time), and no stray commit
  from an agent with bash access can land anywhere real.
- The scratch path lives OUTSIDE the repo tree by default (a nested worktree
  confuses `git status`/`.gitignore` handling in the SOURCE repo).
- A marker file makes automated orphan cleanup (gepa_scratch_gc.py) safe to
  run against a repo that has several other real worktrees checked out --
  the GC utility must never touch anything without this marker present.
- The pi binary is resolved to an ABSOLUTE path once and handed to the
  worktree via LITTLE_CODER_PI_BIN_OVERRIDE (rpc_client.py:41-50 already
  resolves this override to absolute at import time specifically so it
  survives being handed to a subprocess with an arbitrary cwd) -- symlinking
  node_modules into the scratch tree was considered and rejected: it's
  gitignored, so a bare `git clean -fdx` during reset() would delete it,
  and the symlink itself would show up as an untracked entry complicating
  `worktree remove`.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

#: Written into every scratch worktree's root immediately after creation.
#: gepa_scratch_gc.py refuses to remove anything lacking this marker.
SCRATCH_MARKER_NAME = ".self-improve-scratch.json"


class ScratchWorktreeError(RuntimeError):
    """Base error for scratch worktree lifecycle failures."""


class ScratchWorktreeCorrupted(ScratchWorktreeError):
    """The scratch worktree's git state is unexpectedly dirty -- real
    corruption (or a materialization bug), not a transient failure. The
    caller should abort the whole run rather than continue on unverified
    state: every score after an undetected corruption is untrustworthy."""


def _run_git(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise ScratchWorktreeError(f"git {' '.join(args)!r} failed (cwd={cwd}): {result.stderr.strip()}")
    return result


def resolve_pi_bin(source_repo_root: Path, explicit: Path | None = None) -> Path:
    """Resolve an absolute path to a `pi` binary (real or a test double) for
    a scratch worktree to use.

    Precedence: explicit argument > an already-exported
    LITTLE_CODER_PI_BIN_OVERRIDE (this is what lets a test route through
    fake_pi.py) > <source_repo_root>/node_modules/.bin/pi. A NEW git
    worktree never has its own node_modules/ (gitignored) -- without one of
    the first two, PiRpc.__init__ would raise FileNotFoundError the moment
    it's constructed inside the scratch tree.
    """
    if explicit:
        return Path(explicit).resolve()
    env_override = os.environ.get("LITTLE_CODER_PI_BIN_OVERRIDE")
    if env_override:
        return Path(env_override).resolve()
    candidate = Path(source_repo_root) / "node_modules" / ".bin" / "pi"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"pi CLI not found for a scratch worktree of {source_repo_root}. Tried "
        f"LITTLE_CODER_PI_BIN_OVERRIDE (unset) and {candidate} (missing). Run "
        f"`npm install` in {source_repo_root}, or set LITTLE_CODER_PI_BIN_OVERRIDE "
        f"explicitly (e.g. to benchmarks/fake_pi.py for testing)."
    )


def prune_stale(source_repo_root: Path) -> None:
    """`git worktree prune`: deregisters worktrees whose directory is gone
    (e.g. a prior run was SIGKILLed after `worktree add` but before
    `worktree remove`) so a later `worktree add` at the same path doesn't
    fail with a stale registration."""
    _run_git(["worktree", "prune"], cwd=source_repo_root, check=False)


@dataclass(frozen=True)
class ScratchWorktree:
    path: Path
    source_repo_root: Path
    base_commit: str
    pi_bin: Path

    def env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Environment for a subprocess run inside this worktree: the base
        environment (defaults to the real os.environ) plus the resolved pi
        binary override."""
        merged = dict(base if base is not None else os.environ)
        merged["LITTLE_CODER_PI_BIN_OVERRIDE"] = str(self.pi_bin)
        return merged

    def set_active_pid(self, pid: int | None) -> None:
        """Record (or clear) the pid of the exercise subprocess currently
        running against this worktree, in the marker file.

        Without this, gepa_scratch_gc.py can only see the ORCHESTRATOR's own
        pid (the "pid" field, set once at creation) -- if the orchestrator is
        SIGKILLed mid-exercise, that pid dies, but the exercise subprocess
        (started with start_new_session=True, its OWN process group/pid --
        see live_eval.py) does not die with it and can still be actively
        writing into this worktree. A naive orphan check based only on the
        orchestrator's pid would then call the worktree safe to remove while
        real work is still in flight."""
        marker_path = self.path / SCRATCH_MARKER_NAME
        try:
            marker = json.loads(marker_path.read_text())
        except (json.JSONDecodeError, OSError):
            marker = {}
        marker["active_pid"] = pid
        marker_path.write_text(json.dumps(marker, indent=2))

    def reset(self) -> None:
        """Hard-reset to base_commit and remove every untracked file except
        the node_modules symlink and the scratch marker. Idempotent -- safe
        to call before every single candidate evaluation, including the
        first (a plain `git reset --hard` on an already-clean detached-HEAD
        checkout is a no-op)."""
        _run_git(["checkout", "--detach", "--force", self.base_commit], cwd=self.path)
        _run_git(["reset", "--hard", self.base_commit], cwd=self.path)
        _run_git(["clean", "-fdx", "-e", "node_modules", "-e", SCRATCH_MARKER_NAME], cwd=self.path)

    def assert_only_expected_dirty(self, expected: list[Path]) -> None:
        """After writing a candidate's files, verify `git status` shows
        exactly the expected paths changed and nothing else. Any surprise
        here means real corruption (or a materialization bug) -- raise
        rather than continue scoring on unverified state.

        SCRATCH_MARKER_NAME is always implicitly expected: it's never
        committed to the source repo (it's created fresh per scratch
        worktree), so `git status` always reports it as untracked (`??`) --
        that's the marker doing its job, not corruption.
        """
        status = _run_git(["status", "--porcelain"], cwd=self.path).stdout
        dirty: set[Path] = set()
        for line in status.splitlines():
            if not line.strip():
                continue
            # Porcelain format: "XY <path>" (or "XY <path> -> <newpath>" for
            # renames) -- path starts at column 3.
            rel = line[3:].strip()
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1]
            dirty.add((self.path / rel).resolve())
        expected_set = {p.resolve() for p in expected} | {(self.path / SCRATCH_MARKER_NAME).resolve()}
        unexpected = dirty - expected_set
        if unexpected:
            raise ScratchWorktreeCorrupted(
                f"scratch worktree {self.path} has unexpected changes after "
                f"materializing a candidate: {sorted(str(p) for p in unexpected)}\n"
                f"full `git status --porcelain`:\n{status}"
            )


@contextlib.contextmanager
def scratch_worktree(
    source_repo_root: Path,
    *,
    commit: str = "HEAD",
    parent_dir: Path | None = None,
    pi_bin: Path | None = None,
    keep: bool = False,
) -> Iterator[ScratchWorktree]:
    """Create a disposable `git worktree add --detach` checkout of
    source_repo_root, pinned at `commit` (resolved to a concrete sha once,
    so every candidate evaluated within the `with` block is scored against
    one immutable base). Always cleaned up on exit (`git worktree remove
    --force`, falling back to `shutil.rmtree` + `git worktree prune` if that
    fails) unless `keep=True`, in which case the path is left on disk and
    logged for post-mortem -- `__exit__`-equivalent cleanup here never masks
    an exception raised inside the `with` block.
    """
    source_repo_root = Path(source_repo_root).resolve()
    prune_stale(source_repo_root)

    base_commit = _run_git(["rev-parse", commit], cwd=source_repo_root).stdout.strip()
    resolved_pi_bin = resolve_pi_bin(source_repo_root, pi_bin)

    root = Path(parent_dir).resolve() if parent_dir else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    scratch_path = root / f"gepa-scratch-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if scratch_path.exists():
        raise ScratchWorktreeError(f"scratch path already exists, refusing to reuse: {scratch_path}")

    _run_git(["worktree", "add", "--detach", str(scratch_path), base_commit], cwd=source_repo_root)

    # If anything between here and the marker write raises (disk full, an
    # injected KeyboardInterrupt, ...), the worktree registration must not be
    # left behind UNMARKED -- gepa_scratch_gc.py refuses to touch anything
    # without a marker present, so an unmarked orphan needs a human to run a
    # manual `git worktree remove` forever, defeating the whole point of the
    # marker-gated GC. Undo the `worktree add` on any failure in this window.
    try:
        marker = {
            "little_coder_self_improve_scratch": True,
            "pid": os.getpid(),
            "created_at": time.time(),
            "base_commit": base_commit,
            "repo_root": str(source_repo_root),
        }
        (scratch_path / SCRATCH_MARKER_NAME).write_text(json.dumps(marker, indent=2))
    except BaseException:
        try:
            _run_git(["worktree", "remove", "--force", str(scratch_path)], cwd=source_repo_root)
        except ScratchWorktreeError:
            shutil.rmtree(scratch_path, ignore_errors=True)
            prune_stale(source_repo_root)
        raise

    worktree = ScratchWorktree(
        path=scratch_path, source_repo_root=source_repo_root,
        base_commit=base_commit, pi_bin=resolved_pi_bin,
    )

    try:
        yield worktree
    finally:
        if keep:
            print(
                f"SCRATCH WORKTREE PRESERVED FOR POST-MORTEM: {scratch_path}\n"
                f"Remove with: python -m benchmarks.self_improve.gepa_scratch_gc "
                f"--clean --repo-root {source_repo_root}"
            )
        else:
            try:
                _run_git(["worktree", "remove", "--force", str(scratch_path)], cwd=source_repo_root)
            except ScratchWorktreeError:
                shutil.rmtree(scratch_path, ignore_errors=True)
                prune_stale(source_repo_root)
