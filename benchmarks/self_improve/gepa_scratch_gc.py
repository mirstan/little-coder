"""List/clean orphaned self-improve scratch repos left by a crashed or
SIGKILLed live-eval run.

Discovery scans one scratch root (`--scratch-root`, default
`tempfile.gettempdir()` -- the same default scratch_worktree() uses; pass
the run's `--scratch-dir` if it set one) for scratch_worktree()'s sibling
layout: `gepa-scratch-<pid>-<hex8>` (tree), `.git` (private git dir),
`.marker.json` and `.lock`. Nothing is touched without the external marker
and the flag inside it, a symlinked tree or git dir is never followed, a held
owner lock always wins over the marker, and only the four siblings of one
base name are ever removed.

A legacy pass handles worktrees left by older runs, which registered a
`git worktree` in the source repo and kept the marker inside the tree. It
runs only when `--repo-root` (default `.`) is a git repo. A branch-having
(non-detached) worktree is never even considered there, regardless of marker
content: legacy scratch worktrees were always created detached.

Usage:
    python -m benchmarks.self_improve.gepa_scratch_gc --list [--scratch-root DIR]
    python -m benchmarks.self_improve.gepa_scratch_gc --clean [--scratch-root DIR] [--older-than-hours 6] [--yes]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from benchmarks.self_improve.scratch_worktree import (
    LEGACY_SCRATCH_MARKER_NAME,
    _force_rmtree,
    _git_env,
    _read_nofollow,
    prune_stale,
    scratch_git_dir,
    scratch_lock_path,
    scratch_marker_path,
)

#: scratch_worktree.py's own naming scheme: f"gepa-scratch-{pid}-{uuid4().hex[:8]}".
_SCRATCH_DIR_NAME_RE = re.compile(r"^gepa-scratch-\d+-[0-9a-f]{8}$")
#: One base name plus the suffix of each sibling scratch_worktree() creates.
_SCRATCH_ENTRY_RE = re.compile(r"^(gepa-scratch-\d+-[0-9a-f]{8})(\.lock|\.marker\.json|\.git)?$")


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    except OSError:
        return False
    return True


def _lock_held(path: Path) -> bool:
    """Whether a live process holds the tree's owner lock
    (scratch_lock_path). The lock path is derived from the tree path, never
    read from the marker. A missing lock file is not held; any other failure
    to probe it counts as held (fail closed)."""
    try:
        fd = os.open(scratch_lock_path(path), os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False


def _liveness_reason(marker: dict) -> Optional[str]:
    """Why the marker says the scratch tree may still be in use, or None."""
    if _pid_alive(marker.get("pid")):
        return f"still running (pid {marker.get('pid')})"
    # The orchestrator's own pid (above) can die on SIGKILL while a detached
    # exercise subprocess (its own process group, recorded here by
    # PolyglotLiveRunner) is still alive and actively using this tree.
    if _pid_alive(marker.get("active_pid")):
        return (f"owning process is gone, but an exercise subprocess is still running "
                f"(pid {marker.get('active_pid')})")
    # mark_spawn_pending() writes spawn_pending_at just before Popen() and
    # set_active_pid() clears it on every path afterwards, so a lingering one
    # means the orchestrator was killed in the window where a subprocess may
    # have started without its pid ever being recorded. Non-removable
    # permanently, not on a time-based expiry: no later evidence will arrive.
    spawn_pending_at = marker.get("spawn_pending_at")
    if isinstance(spawn_pending_at, (int, float)):
        age_s = time.time() - spawn_pending_at
        return (
            f"a subprocess started spawning {age_s:.0f}s ago and neither pid nor "
            "active_pid is alive -- set_active_pid() never ran to clear spawn_pending_at, "
            "so the orchestrator crashed mid-spawn; not safe to remove"
        )
    return None


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def find_scratch_dirs(scratch_root: Path) -> list[dict]:
    """One dict per scratch base name found directly under scratch_root:
    {"path": Path, "removable": bool, "reason": str, "marker": dict | None,
    "kind": "private-repo"}. Only removable entries are ever touched by
    --clean. A tree holding a `.git` FILE and no external marker is a legacy
    worktree and is left to the legacy pass."""
    root = Path(scratch_root).resolve()
    bases: set[str] = set()
    try:
        with os.scandir(root) as it:
            for entry in it:
                m = _SCRATCH_ENTRY_RE.match(entry.name)
                if m:
                    bases.add(m.group(1))
    except FileNotFoundError:
        return []

    results: list[dict] = []
    for base in sorted(bases):
        tree = root / base
        git_dir = scratch_git_dir(tree)
        marker_path = scratch_marker_path(tree)
        info = {"path": tree, "removable": False, "reason": "", "marker": None, "kind": "private-repo"}

        if _is_symlink(tree) or _is_symlink(git_dir):
            info["reason"] = "tree or git dir is a symlink -- never followed, never touched"
            results.append(info)
            continue

        has_marker = os.path.lexists(marker_path)
        if not has_marker and os.path.isfile(tree / ".git") and not _is_symlink(tree / ".git"):
            continue  # a legacy `git worktree`; the legacy pass reports it

        if _lock_held(tree):
            info["reason"] = f"still running (owner lock {scratch_lock_path(tree).name} is held)"
            results.append(info)
            continue

        if not has_marker:
            info["reason"] = "no external marker -- not ours, never touch"
            results.append(info)
            continue

        try:
            marker = json.loads(_read_nofollow(marker_path))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            info["reason"] = f"marker unreadable ({e}) -- not ours, never touch"
            results.append(info)
            continue
        if not isinstance(marker, dict) or not marker.get("little_coder_self_improve_scratch"):
            info["reason"] = "marker present but missing the expected flag -- not ours, never touch"
            results.append(info)
            continue

        info["marker"] = marker
        reason = _liveness_reason(marker)
        if reason:
            info["reason"] = reason
            results.append(info)
            continue

        info["removable"] = True
        if os.path.lexists(tree) and os.path.lexists(git_dir):
            info["reason"] = "orphaned scratch repo (owning process is gone), safe to remove"
        else:
            info["reason"] = "orphaned (owner gone; tree/gitdir partially present), safe to remove"
        results.append(info)
    return results


def _remove_scratch(root: Path, tree: Path) -> None:
    """Remove one scratch base's siblings: tree and git dir first, then the
    marker and lock only once both are gone (so a partial failure stays
    visible to the next GC run). Raises OSError on failure."""
    root = Path(root).resolve()
    tree = Path(tree)
    if tree.parent.resolve() != root or not _SCRATCH_DIR_NAME_RE.match(tree.name):
        raise OSError(f"refusing to remove {tree}: not a scratch name directly under {root}")
    git_dir = scratch_git_dir(tree)
    for p in (tree, git_dir):
        if _is_symlink(p):
            raise OSError(f"refusing to remove {p}: it is a symlink")
    for p in (tree, git_dir):
        _force_rmtree(p)
    remaining = [p for p in (tree, git_dir) if os.path.lexists(p)]
    if remaining:
        raise OSError(f"could not remove {', '.join(map(str, remaining))}")
    for p in (scratch_marker_path(tree), scratch_lock_path(tree)):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------
# Legacy pass: `git worktree`s registered in the source by older runs
# --------------------------------------------------------------------------


def _is_git_repo(path: Path) -> bool:
    try:
        return subprocess.run(["git", "rev-parse", "--git-dir"], cwd=path, capture_output=True,
                              text=True, env=_git_env()).returncode == 0
    except OSError:
        return False


def _parse_worktree_list(repo_root: Path) -> list[dict]:
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root, capture_output=True, text=True, check=True, env=_git_env(),
    )
    entries: list[dict] = []
    current: dict = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current = {"path": line[len("worktree "):], "branch": None,
                       "detached": False, "prunable": False}
        elif line.startswith("branch "):
            current["branch"] = line[len("branch "):]
        elif line == "detached":
            current["detached"] = True
        elif line.startswith("prunable"):
            current["prunable"] = True
    if current:
        entries.append(current)
    return entries


def find_legacy_worktrees(repo_root: Path, scratch_root: Optional[Path] = None) -> list[dict]:
    """One dict per worktree registered against repo_root:
    {"path", "removable", "reason", "marker", "kind": "legacy-worktree"}.
    Only entries with removable=True are ever touched by --clean."""
    repo_root = Path(repo_root).resolve()
    scratch_root_resolved = Path(scratch_root).resolve() if scratch_root else None
    results: list[dict] = []
    for entry in _parse_worktree_list(repo_root):
        path = Path(entry["path"])
        info = {"path": path, "removable": False, "reason": "", "marker": None, "kind": "legacy-worktree"}

        if scratch_root_resolved is not None:
            try:
                path.resolve().relative_to(scratch_root_resolved)
            except ValueError:
                info["reason"] = "outside configured scratch root"
                results.append(info)
                continue

        if not entry["detached"]:
            info["reason"] = f"has a branch checked out ({entry['branch']}) -- never touch"
            results.append(info)
            continue

        # Checked before anything the marker says: the marker can be
        # rewritten to look dead, but the owner's flock cannot be released.
        if _lock_held(path):
            info["reason"] = f"still running (owner lock {scratch_lock_path(path).name} is held)"
            results.append(info)
            continue

        if not path.exists():
            # The marker lived INSIDE this now-gone directory, so it can never
            # be checked here -- only remove automatically when BOTH the
            # caller scoped us to a known scratch root (verified above) AND
            # the directory's own name matches our naming scheme: containment
            # alone is not ownership evidence (an unrelated tool's detached
            # worktree could live under a shared tmp root), but containment
            # plus an exact name match makes a coincidental false positive
            # practically impossible. Otherwise leave it for a plain
            # `git worktree prune` (safe regardless, the directory is gone).
            name_matches = _SCRATCH_DIR_NAME_RE.match(path.name) is not None
            if entry.get("prunable") and scratch_root_resolved is not None and name_matches:
                info["removable"] = True
                info["reason"] = (
                    "directory gone, git already marks it prunable, under configured scratch "
                    "root, and its name matches our own scratch naming scheme"
                )
            elif entry.get("prunable"):
                info["reason"] = (
                    "directory gone, git marks it prunable, but no marker to verify (and either "
                    "no --scratch-root was given, or the directory name doesn't match our own "
                    "scratch naming scheme) -- run `git worktree prune` directly if this is safe"
                )
            else:
                info["reason"] = "directory gone but git does not mark it prunable -- leave to `git worktree prune`"
            results.append(info)
            continue

        marker_path = path / LEGACY_SCRATCH_MARKER_NAME
        if not marker_path.exists():
            info["reason"] = "no scratch marker file -- not ours, never touch"
            results.append(info)
            continue
        try:
            marker = json.loads(_read_nofollow(marker_path))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            info["reason"] = f"marker file unreadable ({e}) -- not ours, never touch"
            results.append(info)
            continue
        if not isinstance(marker, dict) or not marker.get("little_coder_self_improve_scratch"):
            info["reason"] = "marker file present but missing the expected flag -- not ours, never touch"
            results.append(info)
            continue

        info["marker"] = marker
        reason = _liveness_reason(marker)
        if reason:
            info["reason"] = reason
            results.append(info)
            continue

        info["removable"] = True
        info["reason"] = "orphaned legacy scratch worktree (owning process is gone), safe to remove"
        results.append(info)
    return results


#: Older name, kept for callers of the pre-private-repo API.
find_scratch_worktrees = find_legacy_worktrees


def _remove_worktree(repo_root: Path, path: Path) -> None:
    result = subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                            cwd=repo_root, capture_output=True, text=True, env=_git_env())
    if result.returncode != 0:
        shutil.rmtree(path, ignore_errors=True)
        prune_stale(repo_root)
    # Only removable entries get here, so the lock is not held: it was left
    # by an owner that never reached its own cleanup.
    try:
        scratch_lock_path(path).unlink()
    except OSError:
        pass


def _format_entry(entry: dict) -> str:
    age_note = ""
    if entry["marker"] and isinstance(entry["marker"].get("created_at"), (int, float)):
        age_h = (time.time() - entry["marker"]["created_at"]) / 3600
        age_note = f" (age {age_h:.1f}h)"
    tag = "REMOVABLE" if entry["removable"] else "skip     "
    return f"{tag}  [{entry['kind']}] {entry['path']}{age_note} -- {entry['reason']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".",
                        help="Source repo for the legacy `git worktree` pass (skipped if not a git repo).")
    parser.add_argument("--scratch-root", default=None,
                        help="Directory to scan for scratch repos (default: the system temp dir). "
                             "Must match the run's --scratch-dir if it set one.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--older-than-hours", type=float, default=0.0)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    scan_root = Path(args.scratch_root if args.scratch_root else tempfile.gettempdir()).resolve()
    do_list = args.list or not args.clean

    entries = find_scratch_dirs(scan_root)
    if _is_git_repo(repo_root):
        seen = {os.path.realpath(e["path"]) for e in entries}
        legacy_scope = Path(args.scratch_root) if args.scratch_root else None
        entries += [e for e in find_legacy_worktrees(repo_root, legacy_scope)
                    if os.path.realpath(e["path"]) not in seen]

    print(f"Scanned scratch root: {scan_root}")
    if do_list:
        if not entries:
            print("No scratch directories or worktrees found.")
        for entry in entries:
            print(_format_entry(entry))

    if not args.clean:
        return 0

    to_remove = [e for e in entries if e["removable"]]
    if args.older_than_hours:
        cutoff = time.time() - args.older_than_hours * 3600
        to_remove = [e for e in to_remove if e["marker"]
                     and isinstance(e["marker"].get("created_at"), (int, float))
                     and e["marker"]["created_at"] <= cutoff]

    if not to_remove:
        print("Nothing to clean.")
        return 0

    print(f"\nWill remove {len(to_remove)} scratch entr{'y' if len(to_remove) == 1 else 'ies'}:")
    for entry in to_remove:
        print(f"  [{entry['kind']}] {entry['path']} -- {entry['reason']}")

    if not args.yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return 2

    # No unconditional repo-wide prune_stale() here: `git worktree remove
    # --force` already deregisters each legacy entry it succeeds on, and
    # _remove_worktree() prunes on its own failure fallback -- a blanket
    # prune afterward would also deregister OTHER prunable worktrees never
    # selected by this invocation's filters, silently expanding a scoped
    # cleanup into a repo-wide one.
    removed = 0
    failed = 0
    for entry in to_remove:
        try:
            if entry["kind"] == "private-repo":
                _remove_scratch(scan_root, entry["path"])
            else:
                _remove_worktree(repo_root, entry["path"])
            removed += 1
        except OSError as e:
            failed += 1
            print(f"WARNING: could not remove {entry['path']}: {e} -- its marker is kept; "
                  f"re-run after fixing permissions.", file=sys.stderr)
    print(f"Removed {removed} scratch entr{'y' if removed == 1 else 'ies'}."
          + (f" {failed} could not be removed." if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
