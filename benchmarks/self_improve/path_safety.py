"""Shared path-containment guard.

components.py (components.yaml -> repo file) and
ingest/aider_polyglot_ingest.py (a results_full_polyglot.json exercise key ->
a file under log_root) both resolve a caller-controlled relative path against
a trusted base directory and must refuse anything that would escape it. These
share this one check rather than each keeping their own copy (they differ
only in failure behavior: raise vs. log+skip), so a later hardening fix --
symlink traversal handling, say -- can't land on one and miss the other.
"""
from pathlib import Path


def resolve_contained_path(base: Path, rel_path: str) -> Path:
    """Join rel_path onto base and verify the result stays inside it.

    An absolute rel_path is rejected outright, even one that happens to
    resolve inside base: pathlib's `/` discards the left side entirely for
    an absolute right side, and callers build a second path from the same
    inputs a different way (aider_polyglot_ingest.py's _build_trajectory()
    joins `log_root / lang / exercise` as separate components, not the
    combined string validated here), so an absolute value can validate
    against one construction and resolve elsewhere under the other --
    detail files silently going missing instead of a clean rejection.
    A relative rel_path can still escape via `../`, rejected below.
    """
    if Path(rel_path).is_absolute():
        raise ValueError(f"path {rel_path!r} must be relative, not absolute")
    base = Path(base).resolve()
    resolved = (base / rel_path).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"path {rel_path!r} escapes base directory {base}")
    return resolved
