"""List/clean orphaned self-improve scratch worktrees left by a crashed or
SIGKILLed live-eval run.

Never touches anything without scratch_worktree.SCRATCH_MARKER_NAME present
and a matching flag inside it -- that's what makes this safe to run against
a repo with several other real worktrees checked out (this repo currently
has ~10). A branch-having (non-detached) worktree is never even considered,
regardless of marker content: scratch worktrees are always created detached
(scratch_worktree.py), so a branch means "not ours."

Usage:
    python -m benchmarks.self_improve.gepa_scratch_gc --list
    python -m benchmarks.self_improve.gepa_scratch_gc --clean [--older-than-hours 6] [--yes]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from benchmarks.self_improve.scratch_worktree import SCRATCH_MARKER_NAME, prune_stale

#: scratch_worktree.py's own real naming scheme: f"gepa-scratch-{pid}-{uuid4().hex[:8]}".
#: Used as a second piece of evidence (alongside --scratch-root containment)
#: before trusting a directory-gone-but-prunable entry as ours -- containment
#: under a caller-given --scratch-root is not ownership proof by itself (an
#: unrelated tool's own detached worktree could happen to live under the same
#: shared directory), but the two together make a coincidental match
#: practically impossible, while still letting the tool's actual, common
#: purpose -- cleaning up ITS OWN crashed scratch checkouts -- keep working.
_SCRATCH_DIR_NAME_RE = re.compile(r"^gepa-scratch-\d+-[0-9a-f]{8}$")

#: Grace window after PolyglotLiveRunner.mark_spawn_pending() (written just
#: before subprocess.Popen()) before a worktree with no live active_pid is
#: trusted as truly orphaned. Closes a real TOCTOU gap, confirmed by review:
#: set_active_pid(proc.pid) can only run AFTER Popen() returns, so a SIGKILL
#: in that window would otherwise leave no marker evidence a subprocess was
#: ever started, and this tool would wrongly call the worktree removable
#: while that just-spawned subprocess is still alive and writing into it.
#: Generous on purpose -- worst case of setting this too high is a stale
#: worktree lingering a few extra minutes before the next --clean run.
_SPAWN_GRACE_SECONDS = 120.0


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


def _parse_worktree_list(repo_root: Path) -> list[dict]:
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root, capture_output=True, text=True, check=True,
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


def find_scratch_worktrees(repo_root: Path, scratch_root: Optional[Path] = None) -> list[dict]:
    """Returns one dict per worktree registered against repo_root:
    {"path": Path, "removable": bool, "reason": str, "marker": dict | None}.
    Only entries with removable=True are ever touched by --clean."""
    repo_root = Path(repo_root).resolve()
    scratch_root_resolved = Path(scratch_root).resolve() if scratch_root else None
    results: list[dict] = []
    for entry in _parse_worktree_list(repo_root):
        path = Path(entry["path"])
        info = {"path": path, "removable": False, "reason": "", "marker": None}

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

        if not path.exists():
            # The marker lived INSIDE this now-gone directory, so it can
            # never be checked here -- only remove automatically when BOTH
            # the caller explicitly scoped us to a known scratch root
            # (verified above via relative_to) AND the directory's own name
            # matches our exact naming scheme (_SCRATCH_DIR_NAME_RE): path
            # containment alone is NOT real ownership evidence (an unrelated
            # tool's own detached worktree could happen to live under the
            # same --scratch-root, e.g. a shared tmp directory -- real gap,
            # confirmed by review), but containment PLUS an exact name match
            # makes a coincidental false positive practically impossible.
            # Without both, honor the stated "never touch without a marker"
            # invariant literally and leave it for a plain `git worktree
            # prune` (safe regardless, since the directory is already gone).
            name_matches = _SCRATCH_DIR_NAME_RE.match(path.name) is not None
            if entry.get("prunable") and scratch_root_resolved is not None and name_matches:
                info["removable"] = True
                info["reason"] = (
                    "directory gone, git already marks it prunable, under configured scratch "
                    "root, and its name matches our own scratch-worktree naming scheme"
                )
            elif entry.get("prunable"):
                info["reason"] = (
                    "directory gone, git marks it prunable, but no marker to verify (and either "
                    "no --scratch-root was given, or the directory name doesn't match our own "
                    "scratch-worktree naming scheme) -- run `git worktree prune` directly if "
                    "this is safe"
                )
            else:
                info["reason"] = "directory gone but git does not mark it prunable -- leave to `git worktree prune`"
            results.append(info)
            continue

        marker_path = path / SCRATCH_MARKER_NAME
        if not marker_path.exists():
            info["reason"] = "no scratch marker file -- not ours, never touch"
            results.append(info)
            continue

        try:
            marker = json.loads(marker_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            info["reason"] = f"marker file unreadable ({e}) -- not ours, never touch"
            results.append(info)
            continue

        if not isinstance(marker, dict) or not marker.get("little_coder_self_improve_scratch"):
            info["reason"] = "marker file present but missing the expected flag -- not ours, never touch"
            results.append(info)
            continue

        info["marker"] = marker
        if _pid_alive(marker.get("pid")):
            info["reason"] = f"still running (pid {marker.get('pid')})"
            results.append(info)
            continue
        # The orchestrator's own pid (above) can die on SIGKILL while a
        # detached exercise subprocess (its own process group, recorded here
        # by PolyglotLiveRunner) is still alive and actively using this
        # worktree -- checking only the orchestrator's pid would call it
        # orphaned and safe to remove out from under that live process.
        if _pid_alive(marker.get("active_pid")):
            info["reason"] = f"owning process is gone, but an exercise subprocess is still running (pid {marker.get('active_pid')})"
            results.append(info)
            continue

        spawn_pending_at = marker.get("spawn_pending_at")
        if isinstance(spawn_pending_at, (int, float)):
            age_s = time.time() - spawn_pending_at
            if age_s < _SPAWN_GRACE_SECONDS:
                info["reason"] = (
                    f"a subprocess started spawning {age_s:.0f}s ago (active_pid not yet "
                    f"recorded) -- within the {_SPAWN_GRACE_SECONDS:.0f}s spawn grace window, "
                    "not yet safe to remove"
                )
                results.append(info)
                continue

        info["removable"] = True
        info["reason"] = "orphaned scratch worktree (owning process is gone), safe to remove"
        results.append(info)
    return results


def _remove_worktree(repo_root: Path, path: Path) -> None:
    result = subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                             cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(path, ignore_errors=True)
        prune_stale(repo_root)


def _format_entry(entry: dict) -> str:
    age_note = ""
    if entry["marker"] and "created_at" in entry["marker"]:
        age_h = (time.time() - entry["marker"]["created_at"]) / 3600
        age_note = f" (age {age_h:.1f}h)"
    tag = "REMOVABLE" if entry["removable"] else "skip     "
    return f"{tag}  {entry['path']}{age_note} -- {entry['reason']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--scratch-root", default=None,
                         help="Restrict consideration to worktrees under this directory.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--older-than-hours", type=float, default=0.0)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    scratch_root = Path(args.scratch_root) if args.scratch_root else None
    do_list = args.list or not args.clean

    entries = find_scratch_worktrees(repo_root, scratch_root)

    if do_list:
        if not entries:
            print("No worktrees registered.")
        for entry in entries:
            print(_format_entry(entry))

    if not args.clean:
        return 0

    to_remove = [e for e in entries if e["removable"]]
    if args.older_than_hours:
        cutoff = time.time() - args.older_than_hours * 3600
        to_remove = [e for e in to_remove if e["marker"] and e["marker"].get("created_at", 0) <= cutoff]

    if not to_remove:
        print("Nothing to clean.")
        return 0

    print(f"\nWill remove {len(to_remove)} scratch worktree(s):")
    for entry in to_remove:
        print(f"  {entry['path']} -- {entry['reason']}")

    if not args.yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return 2

    # No unconditional repo-wide prune_stale() here: `git worktree remove
    # --force` already deregisters each entry it succeeds on, and
    # _remove_worktree() already prunes on its own failure fallback --
    # a blanket prune afterward would also deregister OTHER prunable
    # worktrees never selected by this invocation's own filters
    # (--scratch-root / --older-than-hours), silently expanding a scoped
    # cleanup into a repo-wide one.
    for entry in to_remove:
        _remove_worktree(repo_root, entry["path"])
    print(f"Removed {len(to_remove)} scratch worktree(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
