"""Disposable private git repo ("scratch tree") for live candidate evaluation.

benchmarks/rpc_client.py binds REPO_ROOT = Path(__file__).parent.parent at
IMPORT time, and .pi/extensions/skill-inject/index.ts
resolves skills/tools from its own import.meta.url. Both mean the ONLY way
to make a candidate's proposed text reach a live agent is to run the harness
as a SUBPROCESS from a tree whose files are the candidate's -- pointing
cwd= at a scratch dir from an in-process PiRpc does nothing.

Design choices, all deliberate:
- A PRIVATE repo, not a `git worktree` of the source. A linked worktree
  shares the source's hooks, config, refs and object store, so anything the
  agent-under-test wrote through it (a hook, a config key, a ref) reached
  the developer's real checkout. Here the scratch repo is created with
  `git init --template=` and a depth-1 fetch of just the base commit from
  the source, so it has its own object store, no hooks directory, no remote,
  and the source's .git is only ever read (rev-parse, upload-pack).
- The git dir G (`<tree>.git`) sits BESIDE the tree, never inside it. A
  `.git` inside the tree would be agent-writable: a planted
  `core.fsmonitor`, hook or `gitdir:` pointer there would be honoured by the
  orchestrator's next reset(). Every orchestrator git call passes
  `--git-dir`/`--work-tree` explicitly, so a `.git` planted in the tree is
  ignored, and reset() deletes every entry named `.git` (any case, any depth).
- G is still reachable by a same-user agent (its path is the tree's path
  plus `.git`, and the tree path is in pi's argv), so every git call on the
  scratch repo is hardened rather than trusting G or the tree:
    * config is pinned: GIT_CONFIG_GLOBAL=/dev/null, GIT_CONFIG_NOSYSTEM=1,
      and the git location/config env vars (`git rev-parse
      --local-env-vars`, GIT_CONFIG_KEY_*/VALUE_*, GIT_ATTR_SOURCE) removed;
      GIT_COMMON_DIR is then pinned to G, so a planted G/commondir cannot
      redirect config, refs or objects elsewhere;
    * before every call (ScratchWorktree._verify_git_dir): the tree and G
      must still be the directories this run created (lstat compared with
      a directory fd held since creation); G's top level must hold only
      the entries in _GIT_DIR_REQUIRED/_GIT_DIR_OPTIONAL, each of its
      expected type (so a logs/ fails; see the core.logAllRefUpdates
      pin); everything under G/refs must be a real directory or regular
      file, and a directory there that cannot be listed fails too (a
      symlinked directory would send a ref update into another
      directory); and G/config must have the bytes captured when creation
      finished -- otherwise ScratchWorktreeCorrupted (fail closed:
      config-defined hooks have no global off switch in git 2.54).
      These are point-in-time checks: a swap made after one and undone
      before the next is not seen, and git's exec takes paths;
    * `-c core.logAllRefUpdates=false`, so git never creates G/logs/.
      The pin does not stop git appending to a reflog that already exists,
      through a symlink or hard link, so a logs/ in G fails the check
      above instead of being trusted;
    * `-c core.hooksPath=/dev/null -c core.fsmonitor=false
      -c core.attributesFile=/dev/null`, and `--attr-source=<base_commit>`
      so an agent-written .gitattributes cannot select a filter driver.
      `--attr-source` needs git >= 2.40; on older git reset() runs `clean`
      before `checkout` instead, so untracked attribute files are gone
      before checkout reads them;
    * `status` runs with --no-optional-locks (no index refresh, so no
      post-index-change hook path) and --ignore-submodules=all, and
      core.excludesFile=/dev/null keeps the default user excludes file
      from hiding untracked files;
    * the orchestrator-only secrets are removed from the environment of
      every git call, creation included.
- The orchestrator's own file operations on the tree and G (the G checks,
  the index and ORIG_HEAD unlinks, the `.git` removal, teardown) go
  through those held directory fds, never by path, so a path swapped
  after a check cannot redirect them.
- reset() deletes G/index before checking out (a crafted index is rebuilt
  from the base commit) and G/ORIG_HEAD (`reset --hard` would follow a
  symbolic one and write the base sha into the ref it names; HEAD is
  rewritten by `checkout --detach` without being followed), and finishes
  by comparing every tracked path with
  `git ls-tree -r <base>` as recorded at creation, before the agent ran:
  type, exec bit and blob hash. Tampering with G/objects (a rewritten
  object, a planted pack or alternates) that changes the checkout fails
  closed instead of persisting into every later candidate.
- The marker, the lock and G are all siblings of the tree. Nothing the
  orchestrator reads or executes lives inside the tree. The marker records
  the orchestrator pid for gepa_scratch_gc.py, which never touches anything
  without it.
- The pi binary is resolved to an ABSOLUTE path once and handed to the
  scratch tree via LITTLE_CODER_PI_BIN_OVERRIDE (rpc_client.py already
  resolves that override to absolute at import time, so it survives being
  handed to a subprocess with an arbitrary cwd). The tree has no
  node_modules of its own (gitignored, never fetched); a symlink placed
  there by hand survives reset() because `clean` excludes node_modules.
"""
from __future__ import annotations

import contextlib
import ctypes
import fcntl
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

#: The reflection LM's key. Defined here rather than in run_gepa (which
#: re-exports it) so ALWAYS_ORCHESTRATOR_ONLY_ENV below is built from the
#: same name the orchestrator reads.
REFLECTION_LM_API_KEY_ENV = "REFLECTION_LM_API_KEY"

#: Removed from every environment ScratchWorktree.env() builds, whatever the
#: caller passes as orchestrator_only_env. SELF_IMPROVE_DOTENV is included
#: because it tells the agent where the secrets file is.
ALWAYS_ORCHESTRATOR_ONLY_ENV = frozenset({REFLECTION_LM_API_KEY_ENV, "SELF_IMPROVE_DOTENV"})

#: The IN-TREE marker name of the old `git worktree` layout. Only
#: gepa_scratch_gc.py's legacy pass reads it; nothing writes it any more.
LEGACY_SCRATCH_MARKER_NAME = ".self-improve-scratch.json"

#: Recorded in every external marker so the GC can tell layouts apart.
MARKER_LAYOUT = "private-repo-v1"

#: First git release with the global `--attr-source` option.
MIN_GIT_FOR_ATTR_SOURCE = (2, 40)

#: Every scratch-repo git call gets these. gc/maintenance are off so git
#: never rewrites G behind the G/config check's back.
_HARDEN = [
    "-c", "core.hooksPath=" + os.devnull,
    "-c", "core.fsmonitor=false",
    "-c", "core.attributesFile=" + os.devnull,
    # A refs/replace/<base> planted in the sibling git dir would otherwise
    # swap in another commit on every checkout/reset.
    "-c", "core.useReplaceRefs=false",
    "-c", "gc.auto=0",
    "-c", "maintenance.auto=false",
    # Defense in depth, not a reproduced attack: a commit-graph planted in
    # G/objects did not change what checkout produced on git 2.54.
    "-c", "core.commitGraph=false",
    "-c", "core.multiPackIndex=false",
    # Defense in depth against an index changed after reset() (which deletes
    # G/index first); G/config itself is byte-checked.
    "-c", "core.splitIndex=false",
    "-c", "core.untrackedCache=false",
    "-c", "core.sparseCheckout=false",
    "-c", "index.sparse=false",
    # GIT_CONFIG_GLOBAL=/dev/null does not stop git reading the default
    # $XDG_CONFIG_HOME/git/ignore, which could hide a file from status.
    "-c", "core.excludesFile=" + os.devnull,
    # git then never creates G/logs/, which lets the G check reject any
    # logs/ entry. The pin alone is not enough: git still appends to a
    # reflog that already exists, through a symlink or hard link.
    "-c", "core.logAllRefUpdates=false",
]

#: Every top-level entry of G that git 2.54 creates during `init --template=`,
#: the depth-1 fetch (FETCH_HEAD is unlinked before creation finishes),
#: checkout, `reset --hard`, `status` and `clean`, with its required type.
#: Anything else -- commondir, gitdir, config.worktree, logs/, worktrees/,
#: hooks/, info/, modules/, sharedindex.*, index.lock, packed-refs, ... --
#: fails closed. packed-refs is left out on purpose: nothing here creates it.
#: logs/ is left out because git appends to a reflog that already exists,
#: through a symlink or hard link; core.logAllRefUpdates=false in _HARDEN
#: keeps git from creating it.
_GIT_DIR_REQUIRED = {"HEAD": stat.S_ISREG, "config": stat.S_ISREG, "objects": stat.S_ISDIR, "refs": stat.S_ISDIR}
_GIT_DIR_OPTIONAL = {"index": stat.S_ISREG, "shallow": stat.S_ISREG, "ORIG_HEAD": stat.S_ISREG}

#: `git rev-parse --local-env-vars` on git 2.54, used when that call fails.
#: GIT_COMMON_DIR is stripped here like the rest, then re-pinned to G by
#: _scratch_repo_env() for every call on the scratch repo (never for the
#: rev-parse on the source checkout).
_FALLBACK_LOCAL_ENV_VARS = frozenset({
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY", "GIT_DIR", "GIT_WORK_TREE", "GIT_IMPLICIT_WORK_TREE", "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE", "GIT_NO_REPLACE_OBJECTS", "GIT_REPLACE_REF_BASE", "GIT_PREFIX",
    "GIT_SHALLOW_FILE", "GIT_COMMON_DIR",
})


def scratch_lock_path(scratch_path: Path) -> Path:
    """Sibling lock file, OUTSIDE the tree, that scratch_worktree() holds an
    exclusive flock on for the tree's lifetime. gepa_scratch_gc.py derives it
    from the tree path and treats a held lock as a live owner: the marker can
    be rewritten by anyone running as this user to look dead, but a lock held
    by the orchestrator cannot be released by editing a file."""
    scratch_path = Path(scratch_path)
    return scratch_path.with_name(scratch_path.name + ".lock")


def scratch_marker_path(scratch_path: Path) -> Path:
    """Sibling marker file recording ownership and liveness for the GC."""
    scratch_path = Path(scratch_path)
    return scratch_path.with_name(scratch_path.name + ".marker.json")


def scratch_git_dir(scratch_path: Path) -> Path:
    """Sibling git dir of the scratch tree's private repo."""
    scratch_path = Path(scratch_path)
    return scratch_path.with_name(scratch_path.name + ".git")


def _write_marker_atomic(marker_path: Path, marker: dict) -> None:
    """Update the marker via a randomly-named temp file + os.replace, never
    by writing through marker_path.

    The marker sits in the scratch root, by default the shared system temp
    dir, under a predictable name, so a symlink can be planted at it. A
    plain `marker_path.write_text(...)` would follow that symlink and let the
    orchestrator's bookkeeping write overwrite an arbitrary file it can
    write to. `os.replace()` acts on the directory entry itself (POSIX
    rename), so it swaps a planted symlink out instead of writing through it,
    and `tempfile.mkstemp` keeps the temp file's own creation from being
    pre-empted by a planted symlink either."""
    fd, tmp_name = tempfile.mkstemp(dir=str(marker_path.parent), prefix=f".{marker_path.name}.tmp-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(marker, indent=2))
        os.replace(tmp_name, marker_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _create_marker_exclusive(marker_path: Path, marker: dict) -> None:
    """First write of the marker: O_EXCL|O_NOFOLLOW, so anything planted at
    the path after the existence check makes creation fail instead of being
    overwritten (and later destroyed as if it were ours)."""
    fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(marker, indent=2))


def _close_fd(fd: int) -> None:
    """os.close() that never raises, so one failed close cannot skip the
    cleanup that follows it."""
    try:
        os.close(fd)
    except OSError:
        pass


def _read_nofollow(path: Path | str | bytes, dir_fd: int | None = None) -> bytes:
    """Read a file without following a symlink at its last component;
    with dir_fd, `path` is relative to that directory fd. O_NONBLOCK so a
    FIFO planted at the name cannot hang the read."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        _close_fd(fd)


def _read_marker(marker_path: Path) -> dict:
    try:
        marker = json.loads(_read_nofollow(marker_path))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}
    return marker if isinstance(marker, dict) else {}


class ScratchWorktreeError(RuntimeError):
    """Base error for scratch tree lifecycle failures."""


class ScratchWorktreeCorrupted(ScratchWorktreeError):
    """The scratch tree or its git dir is in a state the orchestrator did not
    put it in -- unexpected dirty files or flagged index entries, a tree or
    git dir that is no longer the directory this run created, an entry in
    G's top level outside the allowlist, a symlink or special file under
    G/refs, a tampered G/config, or a checkout
    that differs from the base commit's recorded tree. The caller should
    abort the whole run rather than continue on unverified state: every
    score after an undetected corruption is untrustworthy, and a tampered
    git dir could run code on the next call."""


def _probe_env() -> dict[str, str]:
    """Minimal environment for git's static self-description probes, which
    read no repo: nothing from os.environ but PATH and HOME."""
    env = {"PATH": os.environ.get("PATH", os.defpath), "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    return env


@functools.lru_cache(maxsize=None)
def _git_version() -> tuple[int, int]:
    out = subprocess.run(["git", "version"], capture_output=True, text=True, env=_probe_env()).stdout
    m = re.search(r"(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _git_supports_attr_source() -> bool:
    return _git_version() >= MIN_GIT_FOR_ATTR_SOURCE


@functools.lru_cache(maxsize=None)
def _git_local_env_vars() -> frozenset[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True,
                             env=_probe_env(), check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        out = ""
    return _FALLBACK_LOCAL_ENV_VARS | frozenset(out.split()) | {"GIT_ATTR_SOURCE"}


def _git_env(withheld: Iterable[str] = ()) -> dict[str, str]:
    """Environment for the orchestrator's own git calls: os.environ minus
    the orchestrator-only secrets and every git location/config override,
    with user and system config pinned off."""
    drop = ALWAYS_ORCHESTRATOR_ONLY_ENV | frozenset(withheld) | _git_local_env_vars()
    env = {
        k: v for k, v in os.environ.items()
        if k not in drop and not k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }
    env.pop("GIT_CONFIG_SYSTEM", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    git_dir: Path | None = None,
    work_tree: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    global_opts: Sequence[str] = (),
    text: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [
        "git",
        *([f"--git-dir={git_dir}"] if git_dir else []),
        *([f"--work-tree={work_tree}"] if work_tree else []),
        *global_opts,
        *_HARDEN,
        *args,
    ]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=text,
                            env=dict(env) if env is not None else _git_env())
    if check and result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise ScratchWorktreeError(
            f"git {' '.join(args)!r} failed (cwd={cwd}, git_dir={git_dir}): {stderr.strip()}"
        )
    return result


#: Opens a directory itself, never a symlink planted at its name.
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _lstat_at(dir_fd: int, name: str | bytes) -> os.stat_result:
    return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)


def _reraise(e: OSError) -> None:
    """os.fwalk onerror: without it fwalk silently skips a directory it
    cannot open or list."""
    raise e


@functools.lru_cache(maxsize=None)
def _libc_fchflags():
    """libc's fchflags(), or None where BSD file flags do not exist. Needed
    because os.chflags takes neither an fd nor dir_fd."""
    if not hasattr(os, "lchflags"):
        return None
    try:
        fn = ctypes.CDLL(None, use_errno=True).fchflags
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_int, ctypes.c_uint]
    fn.restype = ctypes.c_int
    return fn


def _clear_flags_fd(fd: int) -> None:
    fchflags = _libc_fchflags()
    try:
        if fchflags is not None and os.fstat(fd).st_flags:
            fchflags(fd, 0)
    except OSError:
        pass


def _clear_flags_at(dir_fd: int, name: str | bytes, st: os.stat_result) -> None:
    """Drop BSD file flags (uchg etc.) from the entry `name` of dir_fd,
    through an fd opened on the entry itself (O_SYMLINK for a symlink,
    O_NOFOLLOW otherwise), so nothing is followed. Best effort: an entry
    that cannot be opened (mode 000 and uchg) keeps its flags."""
    if not getattr(st, "st_flags", 0) or _libc_fchflags() is None:
        return
    if stat.S_ISLNK(st.st_mode):
        if not hasattr(os, "O_SYMLINK"):
            return
        flags = os.O_RDONLY | os.O_SYMLINK | os.O_NONBLOCK
    else:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError:
        return
    try:
        _clear_flags_fd(fd)
    finally:
        _close_fd(fd)


def _chmod_dir_at(dir_fd: int, name: str | bytes) -> None:
    try:
        os.chmod(name, 0o700, dir_fd=dir_fd, follow_symlinks=False)
    except NotImplementedError:
        # No lchmod (Linux): the entry was lstat'ed as a directory just before.
        try:
            os.chmod(name, 0o700, dir_fd=dir_fd)
        except OSError:
            pass
    except OSError:
        pass


def _make_removable_at(dir_fd: int, name: str | bytes) -> None:
    """Clear flags on every entry under `name` (relative to dir_fd) and give
    every real directory u+rwx, top down so a chmod 000 directory is opened
    only after it is fixed. Never follows symlinks; every step is relative
    to a directory fd."""
    try:
        st = _lstat_at(dir_fd, name)
    except OSError:
        return
    if not stat.S_ISDIR(st.st_mode):
        _clear_flags_at(dir_fd, name, st)
        return
    _chmod_dir_at(dir_fd, name)  # so a uchg'd 000 directory can be opened
    _clear_flags_at(dir_fd, name, st)
    _chmod_dir_at(dir_fd, name)  # and in case uchg blocked the first chmod
    try:
        fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
    except OSError:
        return
    try:
        for child in os.listdir(fd):
            _make_removable_at(fd, child)
    except OSError:
        pass
    finally:
        _close_fd(fd)


def _force_rmtree_at(dir_fd: int, name: str | bytes) -> None:
    """Remove the entry `name` of directory fd `dir_fd` (a directory tree,
    file or symlink) without following symlinks. If plain removal fails --
    an agent can chmod 000 a directory or chflags uchg a file -- reset
    permissions and flags and retry once. Raises OSError if it still cannot
    be removed."""
    try:
        st = _lstat_at(dir_fd, name)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(st.st_mode):
        _clear_flags_at(dir_fd, name, st)
        os.unlink(name, dir_fd=dir_fd)
        return
    try:
        shutil.rmtree(name, dir_fd=dir_fd)
        return
    except OSError:
        pass
    _make_removable_at(dir_fd, name)
    shutil.rmtree(name, dir_fd=dir_fd)


class _NotOurDirectory(OSError):
    """What is at a scratch path is no longer the directory this run created."""


@dataclass(frozen=True)
class _HeldDir:
    """A directory this run created plus the fd it has held on it since its
    mkdir. _force_rmtree() of one removes only that directory: it empties it
    through the fd, then rmdir()s the path, which cannot remove a non-empty
    directory or a symlink swapped in meanwhile."""
    path: Path
    fd: int

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


def _same_dir(path: Path, fd: int) -> bool:
    try:
        st, ours = os.lstat(path), os.fstat(fd)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and (st.st_dev, st.st_ino) == (ours.st_dev, ours.st_ino)


def _remove_held_dir(held: _HeldDir) -> None:
    if not _same_dir(held.path, held.fd):
        if os.path.lexists(held.path):
            raise _NotOurDirectory(f"{held.path} is not the directory this run created")
        return
    _clear_flags_fd(held.fd)
    try:
        os.chmod(held.fd, 0o700)
    except OSError:
        pass
    for name in os.listdir(held.fd):
        _force_rmtree_at(held.fd, name)
    try:
        os.rmdir(held.path)
    except FileNotFoundError:
        pass


def _force_rmtree(path: Path | _HeldDir) -> None:
    """Remove `path` (a directory tree, file or symlink) without following
    symlinks, relative to an fd on its parent directory (see
    _force_rmtree_at). A _HeldDir is removed through its own held fd
    instead. Raises OSError if it cannot be removed."""
    if isinstance(path, _HeldDir):
        _remove_held_dir(path)
        return
    p = os.path.abspath(path)
    try:
        parent_fd = os.open(os.path.dirname(p), os.O_RDONLY | os.O_DIRECTORY)
    except FileNotFoundError:
        return
    try:
        _force_rmtree_at(parent_fd, os.path.basename(p))
    finally:
        _close_fd(parent_fd)


def _destroy(
    created: Sequence[Path], marker_path: Path | None, held: Mapping[Path, int] | None = None,
) -> bool:
    """Remove the paths this process created, then the marker -- but only
    once every one of them is really gone. A tree left behind with no marker
    would be invisible to gepa_scratch_gc.py forever. A path in `held` is
    removed through its held directory fd, and left alone (with a warning,
    and not kept for the GC) if what is there is no longer the directory
    this run created. Returns whether everything was removed; never raises
    (it runs in `finally`)."""
    held = held or {}
    remaining: list[Path] = []
    for p in created:
        try:
            _force_rmtree(_HeldDir(p, held[p]) if p in held else p)
        except _NotOurDirectory as e:
            print(f"WARNING: not removing {e}; the directory this run created was moved "
                  f"elsewhere and is not tracked any more.", file=sys.stderr)
            continue
        except OSError as e:
            print(f"WARNING: could not remove {p}: {e}", file=sys.stderr)
        if os.path.lexists(p):
            remaining.append(p)
    if remaining:
        print(
            f"WARNING: could not remove {', '.join(map(str, remaining))}; keeping "
            f"{marker_path} so `python -m benchmarks.self_improve.gepa_scratch_gc --clean "
            f"--scratch-root {Path(remaining[0]).parent}` can remove it later.",
            file=sys.stderr,
        )
        return False
    if marker_path is not None:
        try:
            os.unlink(marker_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"WARNING: could not remove {marker_path}: {e}", file=sys.stderr)
    return True


def resolve_pi_bin(source_repo_root: Path, explicit: Path | None = None) -> Path:
    """Resolve an absolute path to a `pi` binary (real or a test double) for
    a scratch tree to use.

    Precedence: explicit argument > an already-exported
    LITTLE_CODER_PI_BIN_OVERRIDE (this is what lets a test route through
    fake_pi.py) > <source_repo_root>/node_modules/.bin/pi. The scratch tree
    never has its own node_modules/ (gitignored) -- without one of the first
    two, PiRpc.__init__ would raise FileNotFoundError the moment it's
    constructed inside the scratch tree.
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
        f"pi CLI not found for a scratch tree of {source_repo_root}. Tried "
        f"LITTLE_CODER_PI_BIN_OVERRIDE (unset) and {candidate} (missing). Run "
        f"`npm install` in {source_repo_root}, or set LITTLE_CODER_PI_BIN_OVERRIDE "
        f"explicitly (e.g. to benchmarks/fake_pi.py for testing)."
    )


def prune_stale(source_repo_root: Path) -> None:
    """`git worktree prune` on the source. Legacy only: scratch_worktree()
    no longer creates worktrees, so only gepa_scratch_gc.py's legacy pass
    (for worktrees left by older runs) calls this."""
    _run_git(["worktree", "prune"], cwd=source_repo_root, check=False)


@dataclass(frozen=True)
class ScratchWorktree:
    path: Path
    source_repo_root: Path
    base_commit: str
    pi_bin: Path
    git_dir: Path
    marker_path: Path
    #: Names withheld from env() on top of ALWAYS_ORCHESTRATOR_ONLY_ENV.
    orchestrator_only_env: frozenset[str] = frozenset()
    #: G/config as it was when creation finished; verified before every git call.
    config_bytes: bytes = field(default=b"", repr=False, compare=False)
    #: O_DIRECTORY|O_NOFOLLOW fds on the tree and on G, opened right after
    #: our own mkdir and held until teardown. Every Python-side file
    #: operation on the tree and G goes through them. Holding them also
    #: keeps both inodes allocated, so (st_dev, st_ino) cannot be reused by
    #: a look-alike.
    tree_fd: int = field(default=-1, repr=False, compare=False)
    git_dir_fd: int = field(default=-1, repr=False, compare=False)
    #: `git ls-tree -r -z <base_commit>` as (mode, object id, path) bytes/str
    #: tuples, recorded at creation before the agent ran; reset() checks
    #: the checkout against it.
    tracked: tuple[tuple[bytes, str, bytes], ...] = field(default=(), repr=False, compare=False)

    def env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Environment for a subprocess run inside this scratch tree: a copy
        of the base environment (defaults to the real os.environ) minus every
        orchestrator-only name, plus the resolved pi binary override.

        Everything the agent-under-test runs (pi, its bash tool, the
        exercise's tests) inherits this environment, so the reflection LM's
        key and the other orchestrator-only names are removed here. Neither
        `base` nor os.environ is modified. The override is set after the
        removal, so naming it orchestrator-only cannot drop it."""
        withheld = ALWAYS_ORCHESTRATOR_ONLY_ENV | self.orchestrator_only_env
        merged = {k: v for k, v in (base if base is not None else os.environ).items() if k not in withheld}
        merged["LITTLE_CODER_PI_BIN_OVERRIDE"] = str(self.pi_bin)
        return merged

    def _git_env(self) -> dict[str, str]:
        return _scratch_repo_env(_git_env(self.orchestrator_only_env), self.git_dir)

    def _verify_identity(self) -> None:
        """Fail closed unless the paths of the tree and G still name the
        directories this process created (lstat compared with the held fds,
        so a symlink planted at either path fails rather than being
        followed). git's exec takes these paths; a swap made after this
        check and undone before the next one is not seen."""
        for what, p, fd in (("tree", self.path, self.tree_fd), ("git dir", self.git_dir, self.git_dir_fd)):
            if not _same_dir(p, fd):
                raise ScratchWorktreeCorrupted(f"scratch {what} {p} is not the directory this run created")

    def _verify_git_dir(self) -> None:
        """Fail closed if G was tampered with: the tree and G must still be
        the directories this run created (_verify_identity); G's top level
        must hold every _GIT_DIR_REQUIRED entry and nothing outside
        _GIT_DIR_REQUIRED/_GIT_DIR_OPTIONAL, each of its expected type (so
        no symlinks); everything under G/refs must be a directory or a
        regular file, every directory there listable; and G/config must
        have exactly the bytes it had when creation finished. G is read
        through the held fd, never by path."""
        self._verify_identity()
        g, gfd = self.git_dir, self.git_dir_fd
        try:
            names = set(os.listdir(gfd))
            missing = set(_GIT_DIR_REQUIRED) - names
            if missing:
                raise ScratchWorktreeCorrupted(f"{g} is missing {sorted(missing)}")
            allowed = {**_GIT_DIR_REQUIRED, **_GIT_DIR_OPTIONAL}
            unexpected = names - set(allowed)
            if unexpected:
                raise ScratchWorktreeCorrupted(f"{g} has unexpected entries {sorted(unexpected)}")
            for name in names:
                if not allowed[name](_lstat_at(gfd, name).st_mode):
                    raise ScratchWorktreeCorrupted(f"{g / name} has the wrong file type")
            # git writes loose refs and their lockfiles inside refs/; a
            # symlinked directory there sends a ref update that follows a
            # symbolic ref into another directory. git only needs search
            # permission to traverse, so a directory this walk cannot list
            # fails closed (onerror) instead of being skipped.
            for _, dirnames, filenames, dirfd in os.fwalk("refs", follow_symlinks=False, dir_fd=gfd,
                                                          onerror=_reraise):
                for name in [*dirnames, *filenames]:
                    mode = _lstat_at(dirfd, name).st_mode
                    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                        raise ScratchWorktreeCorrupted(f"{g / 'refs'} holds a symlink or special file")
            if _read_nofollow("config", dir_fd=gfd) != self.config_bytes:
                raise ScratchWorktreeCorrupted(f"{g / 'config'} changed since the scratch repo was created")
        except OSError as e:
            raise ScratchWorktreeCorrupted(f"cannot verify scratch git dir {g}: {e}") from e

    def _verify_checkout(self) -> None:
        """Fail closed unless every path of the base commit's tree (as
        recorded at creation) is in the tree with the recorded type, exec
        bit and blob hash. Read through the held tree fd, one O_NOFOLLOW
        directory fd per component. This catches anything that makes
        checkout produce other content -- a rewritten object, a planted pack
        or alternates, a crafted index -- whatever the mechanism."""
        dir_fds: dict[bytes, int | None] = {b"": self.tree_fd}

        def _dir(rel: bytes) -> int | None:
            if rel not in dir_fds:
                parent, _, name = rel.rpartition(b"/")
                pfd = _dir(parent)
                try:
                    dir_fds[rel] = None if pfd is None else os.open(name, _DIR_OPEN_FLAGS, dir_fd=pfd)
                except OSError:
                    dir_fds[rel] = None
            return dir_fds[rel]

        mismatched: list[str] = []
        try:
            for mode, oid, path in self.tracked:
                parent, _, name = path.rpartition(b"/")
                dfd = _dir(parent)
                if dfd is None or not _entry_matches(dfd, name, mode, oid):
                    mismatched.append(os.fsdecode(path))
        finally:
            for rel, fd in dir_fds.items():
                if rel and fd is not None:
                    _close_fd(fd)
        if mismatched:
            more = f" (and {len(mismatched) - 20} more)" if len(mismatched) > 20 else ""
            raise ScratchWorktreeCorrupted(
                f"scratch tree {self.path} does not match base commit {self.base_commit} after checkout "
                f"(type, exec bit or content): {mismatched[:20]}{more}"
            )

    def git(
        self, args: Sequence[str], check: bool = True, global_opts: Sequence[str] = (),
    ) -> subprocess.CompletedProcess:
        """Run git on this scratch repo with the hardened, scrubbed setup
        (see the module docstring), after verifying G."""
        self._verify_git_dir()
        attr = [f"--attr-source={self.base_commit}"] if _git_supports_attr_source() else []
        return _run_git(
            args, git_dir=self.git_dir, work_tree=self.path, env=self._git_env(),
            check=check, global_opts=[*attr, *global_opts],
        )

    def mark_spawn_pending(self) -> None:
        """Record, BEFORE subprocess.Popen() is even called, that a new
        exercise subprocess is about to start against this scratch tree.

        Closes a real TOCTOU gap: set_active_pid(proc.pid) can only run
        AFTER Popen() returns a pid, so if the orchestrator is SIGKILLed in
        the (normally tiny, but non-zero) window between Popen() returning
        and set_active_pid() actually writing the marker, gepa_scratch_gc.py
        would see no active_pid at all and could wrongly call the tree
        orphaned while the just-spawned subprocess is still alive and
        writing into it. gepa_scratch_gc.py treats a spawn_pending_at as
        reason enough to withhold removal, regardless of what active_pid
        looks like."""
        marker = _read_marker(self.marker_path)
        marker["spawn_pending_at"] = time.time()
        _write_marker_atomic(self.marker_path, marker)

    def set_active_pid(self, pid: int | None) -> None:
        """Record (or clear) the pid of the exercise subprocess currently
        running against this scratch tree, in the marker file. Also clears
        spawn_pending_at, which gepa_scratch_gc.py treats as permanent
        proof of a crash mid-spawn: without the clear, every tree that had
        ever run one exercise would carry it forever. live_eval.py calls
        this from a `finally` on every path after mark_spawn_pending(), so a
        still-set spawn_pending_at can only mean the orchestrator died
        before reaching here.

        Without active_pid tracking at all, gepa_scratch_gc.py can only see
        the ORCHESTRATOR's own pid (the "pid" field, set once at creation)
        -- if the orchestrator is SIGKILLed mid-exercise, that pid dies, but
        the exercise subprocess (started with start_new_session=True, its
        OWN process group/pid -- see live_eval.py) does not die with it and
        can still be actively writing into this tree. A naive orphan check
        based only on the orchestrator's pid would then call the tree safe
        to remove while real work is still in flight."""
        marker = _read_marker(self.marker_path)
        marker["active_pid"] = pid
        marker.pop("spawn_pending_at", None)
        _write_marker_atomic(self.marker_path, marker)

    def _remove_dot_git_entries(self, top_level_only: bool) -> None:
        """Delete every entry named `.git` (case-insensitively: APFS treats
        `.GIT` as the same name) in the tree. Git skips such names at every
        depth, so `clean` never removes them and `status` never reports
        them -- left alone they are a hidden store that survives reset().
        Walks from the held tree fd, so it never leaves the tree even if
        the tree's path is swapped meanwhile."""
        if top_level_only:
            for name in os.listdir(self.tree_fd):
                if name.lower() == ".git":
                    _force_rmtree_at(self.tree_fd, name)
            return
        for _, dirnames, filenames, dirfd in os.fwalk(".", topdown=True, follow_symlinks=False,
                                                      dir_fd=self.tree_fd):
            # fwalk lists a symlink to a directory in dirnames (and never
            # descends it); _force_rmtree_at unlinks it rather than following.
            for name in [*dirnames, *filenames]:
                if name.lower() == ".git":
                    _force_rmtree_at(dirfd, name)
            dirnames[:] = [d for d in dirnames if d.lower() != ".git"]

    def reset(self) -> None:
        """Hard-reset to base_commit and remove every untracked file except
        node_modules, plus every `.git` entry anywhere in the tree, then
        verify the result against the base commit's recorded tree.
        Idempotent -- safe to call before every single candidate evaluation,
        including the first. Raises ScratchWorktreeCorrupted if the tree or
        git dir was tampered with or the checkout does not match."""
        self._verify_git_dir()
        # A crafted index (skip-worktree or assume-unchanged bits, split or
        # sparse index, untracked cache) survives checkout/reset and hides
        # files from status; checkout rebuilds it from base_commit.
        # `reset --hard` updates ORIG_HEAD and follows it if it is a symbolic
        # ref (`ref: refs/heads/<x>`), writing the base sha into whatever
        # that names; unlinked here, it is simply recreated. HEAD needs no
        # unlink: `checkout --detach` below rewrites it without following
        # it, so it must stay before `reset --hard`.
        for name in ("index", "ORIG_HEAD"):
            try:
                os.unlink(name, dir_fd=self.git_dir_fd)
            except FileNotFoundError:
                pass
        self._remove_dot_git_entries(top_level_only=True)
        if not _git_supports_attr_source():
            # No --attr-source: remove an agent-written .gitattributes before
            # checkout and reset read attributes from the work tree.
            self.git(["clean", "-ffdx", "-e", "node_modules"])
        self.git(["checkout", "--detach", "--force", self.base_commit])
        self.git(["reset", "--hard", self.base_commit])
        # -ff: also removes untracked nested repositories.
        self.git(["clean", "-ffdx", "-e", "node_modules"])
        self._verify_identity()
        self._remove_dot_git_entries(top_level_only=False)
        self._verify_checkout()

    def assert_only_expected_dirty(self, expected: list[Path]) -> None:
        """After writing a candidate's files, verify `git status` shows
        exactly the expected paths changed and nothing else. Any surprise
        here means real corruption (or a materialization bug) -- raise
        rather than continue scoring on unverified state. The marker lives
        outside the tree, so nothing else is implicitly expected."""
        flagged = [
            line for line in self.git(["ls-files", "-v"], global_opts=["--no-optional-locks"]).stdout.splitlines()
            if not line.startswith("H ")
        ]
        if flagged:
            raise ScratchWorktreeCorrupted(
                f"scratch index of {self.path} has entries status would not report: {flagged[:20]}"
            )
        status = self.git(["status", "--porcelain", "--ignore-submodules=all"],
                          global_opts=["--no-optional-locks"]).stdout
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
        expected_set = {p.resolve() for p in expected}
        unexpected = dirty - expected_set
        if unexpected:
            raise ScratchWorktreeCorrupted(
                f"scratch tree {self.path} has unexpected changes after "
                f"materializing a candidate: {sorted(str(p) for p in unexpected)}\n"
                f"full `git status --porcelain`:\n{status}"
            )


def _scratch_repo_env(env: Mapping[str, str], git_dir: Path) -> dict[str, str]:
    """`env` plus GIT_COMMON_DIR pinned to G, so a planted G/commondir can
    never redirect config, refs or objects to another directory."""
    return {**env, "GIT_COMMON_DIR": str(git_dir)}


def _git_object_id(data: bytes, oid: str) -> str:
    """Blob object id of `data`, in the hash of `oid` (sha1, or sha256 for
    a sha256 repository)."""
    algo = hashlib.sha256 if len(oid) == 64 else hashlib.sha1
    return algo(b"blob %d\0" % len(data) + data).hexdigest()


def _entry_matches(dir_fd: int, name: bytes, mode: bytes, oid: str) -> bool:
    """Whether the entry `name` of dir_fd is what `git ls-tree` recorded:
    a regular file with the right owner exec bit and blob hash, a symlink
    whose target hashes to the blob, or (gitlink) a directory."""
    try:
        st = _lstat_at(dir_fd, name)
        if mode == b"120000":
            if not stat.S_ISLNK(st.st_mode):
                return False
            data = os.readlink(name, dir_fd=dir_fd)
        elif mode in (b"100644", b"100755"):
            if not stat.S_ISREG(st.st_mode) or bool(st.st_mode & stat.S_IXUSR) != (mode == b"100755"):
                return False
            data = _read_nofollow(name, dir_fd=dir_fd)
        elif mode == b"160000":
            return stat.S_ISDIR(st.st_mode)
        else:
            return False
    except OSError:
        return False
    return _git_object_id(data, oid) == oid


def _record_base_tree(git_dir: Path, base_commit: str, env: Mapping[str, str]) -> tuple[tuple[bytes, str, bytes], ...]:
    """`git ls-tree -r -z <base_commit>` of the freshly created repo, before
    the agent has run: (mode, object id, path) per entry, paths as bytes."""
    out = _run_git(["ls-tree", "-r", "-z", "--full-tree", base_commit], git_dir=git_dir,
                   env=_scratch_repo_env(env, git_dir), text=False).stdout
    entries = []
    for record in out.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, _, oid = meta.split(b" ")
        entries.append((mode, oid.decode("ascii"), path))
    return tuple(entries)


def _create_private_repo(
    source_repo_root: Path, base_commit: str, scratch_path: Path, git_dir: Path, env: Mapping[str, str],
) -> None:
    """Populate the (already created, empty) scratch_path and git_dir: a
    private repo holding only base_commit, checked out detached.

    The `--upload-pack` override sets uploadpack.allowAnySHA1InWant inside
    the upload-pack command itself, because the local transport strips
    client-side `-c` config from the upload-pack child; without it a
    commit reachable from no ref fails to fetch under protocol v0."""
    env = _scratch_repo_env(env, git_dir)
    _run_git(["init", "-q", "--template="], git_dir=git_dir, work_tree=scratch_path, env=env)
    _run_git(
        ["fetch", "-q", "--no-tags", "--depth=1",
         "--upload-pack=git -c uploadpack.allowAnySHA1InWant=true upload-pack",
         str(source_repo_root), base_commit],
        git_dir=git_dir, env=env,
    )
    _run_git(["cat-file", "-e", f"{base_commit}^{{commit}}"], git_dir=git_dir, env=env)
    attr = [f"--attr-source={base_commit}"] if _git_supports_attr_source() else []
    _run_git(
        ["checkout", "-q", "--detach", "--force", base_commit],
        git_dir=git_dir, work_tree=scratch_path, env=env,
        global_opts=[*attr, "-c", "advice.detachedHead=false"],
    )
    # FETCH_HEAD names the source path; nothing needs it after checkout.
    try:
        os.unlink(git_dir / "FETCH_HEAD")
    except FileNotFoundError:
        pass


def _hardened_inspect_command(scratch_path: Path, git_dir: Path, base_commit: str) -> str:
    """The `git status` a human can run on a preserved tree. Nothing
    verifies G any more by then, so it also takes --no-pager and
    --porcelain: a pager set in G/config is never started."""
    parts = [
        "GIT_CONFIG_GLOBAL=" + os.devnull, "GIT_CONFIG_NOSYSTEM=1", f"GIT_COMMON_DIR={git_dir}", "git",
        "--no-pager", f"--git-dir={git_dir}", f"--work-tree={scratch_path}", "--no-optional-locks",
    ]
    if _git_supports_attr_source():
        parts.append(f"--attr-source={base_commit}")
    parts += [*_HARDEN, "status", "--porcelain", "--ignore-submodules=all"]
    return " ".join(shlex.quote(p) for p in parts)


@contextlib.contextmanager
def scratch_worktree(
    source_repo_root: Path,
    *,
    commit: str = "HEAD",
    parent_dir: Path | None = None,
    pi_bin: Path | None = None,
    keep: bool = False,
    orchestrator_only_env: Iterable[str] = (),
) -> Iterator[ScratchWorktree]:
    """Create a disposable private repo of source_repo_root, checked out
    detached at `commit` (resolved to a concrete sha once, so every candidate
    evaluated within the `with` block is scored against one immutable base).
    The tree, its sibling git dir and marker are always removed on exit
    unless `keep=True`, in which case they are left on disk and logged for
    post-mortem -- exit-time cleanup here never masks an exception raised
    inside the `with` block.

    `orchestrator_only_env` names variables ScratchWorktree.env() withholds
    from the agent-under-test, in addition to ALWAYS_ORCHESTRATOR_ONLY_ENV.
    They are also withheld from every git call this function makes.
    """
    source_repo_root = Path(source_repo_root).resolve()
    withheld = frozenset(orchestrator_only_env)
    env = _git_env(withheld)

    base_commit = _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=source_repo_root,
                           env=env).stdout.strip()
    resolved_pi_bin = resolve_pi_bin(source_repo_root, pi_bin)

    root = Path(parent_dir).resolve() if parent_dir else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    scratch_path = root / f"gepa-scratch-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    git_dir = scratch_git_dir(scratch_path)
    marker_path = scratch_marker_path(scratch_path)
    lock_path = scratch_lock_path(scratch_path)
    for p in (scratch_path, git_dir, marker_path, lock_path):
        if os.path.lexists(p):
            raise ScratchWorktreeError(f"scratch path already exists, refusing to reuse: {p}")

    # Taken before anything else exists, so nothing of ours is ever unlocked.
    # The fd is non-inheritable (PEP 446), so only this process holds it.
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)

    def _release_lock(unlink: bool) -> None:
        # The lock file is unlinked only together with the marker: if the
        # tree could not be removed, both stay for gepa_scratch_gc.py (an
        # unheld lock file is not evidence of a live owner).
        if unlink:
            try:
                os.unlink(lock_path)
            except OSError:
                pass
        _close_fd(lock_fd)

    created: list[Path] = []
    #: Each created directory's held fd, closed only after teardown used it.
    held: dict[Path, int] = {}
    marker_created = False

    def _close_held() -> None:
        for fd in held.values():
            _close_fd(fd)
        held.clear()

    def _open_dir(p: Path) -> int:
        fd = os.open(p, _DIR_OPEN_FLAGS)
        held[p] = fd
        return fd

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The marker is written before any directory exists, so there is no
        # window in which an orphan of ours is unmarked.
        _create_marker_exclusive(marker_path, {
            "little_coder_self_improve_scratch": True,
            "layout": MARKER_LAYOUT,
            "pid": os.getpid(),
            "created_at": time.time(),
            "base_commit": base_commit,
            "repo_root": str(source_repo_root),
            "tree_path": str(scratch_path),
        })
        marker_created = True
        # Each path is ours (and so ours to destroy) only once our own mkdir
        # of it succeeded.
        os.mkdir(scratch_path, 0o700)
        created.append(scratch_path)
        tree_fd = _open_dir(scratch_path)
        os.mkdir(git_dir, 0o700)
        created.append(git_dir)
        git_dir_fd = _open_dir(git_dir)
        _create_private_repo(source_repo_root, base_commit, scratch_path, git_dir, env)
        config_bytes = _read_nofollow("config", dir_fd=git_dir_fd)
        # Recorded now, before any agent has run, so G/objects is trusted.
        tracked = _record_base_tree(git_dir, base_commit, env)
        worktree = ScratchWorktree(
            path=scratch_path, source_repo_root=source_repo_root,
            base_commit=base_commit, pi_bin=resolved_pi_bin,
            git_dir=git_dir, marker_path=marker_path,
            orchestrator_only_env=withheld, config_bytes=config_bytes,
            tree_fd=tree_fd, git_dir_fd=git_dir_fd, tracked=tracked,
        )
        # Fail at startup, not at candidate 2, if this git leaves an entry in
        # G that the allowlist does not know, or if the base's checkout
        # differs from its blobs (an eol or filter attribute, say).
        worktree._verify_git_dir()
        worktree._verify_checkout()
    except BaseException:
        removed = _destroy(created, marker_path if marker_created else None, held)
        _close_held()
        _release_lock(unlink=removed)
        raise

    try:
        yield worktree
    finally:
        if keep:
            _close_held()
            print(
                f"SCRATCH WORKTREE PRESERVED FOR POST-MORTEM: {scratch_path}\n"
                f"Inspect with: {_hardened_inspect_command(scratch_path, git_dir, base_commit)}\n"
                f"Remove with: python -m benchmarks.self_improve.gepa_scratch_gc "
                f"--clean --scratch-root {shlex.quote(str(root))}"
            )
            # Released even with keep=True: no live owner remains, so the GC
            # falls back to the marker for a preserved tree.
            _release_lock(unlink=True)
        else:
            removed = _destroy([scratch_path, git_dir], marker_path, held)
            _close_held()
            _release_lock(unlink=removed)
