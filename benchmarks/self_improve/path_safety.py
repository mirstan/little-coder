"""Shared path-containment guard.

components.py (components.yaml -> repo file) and
ingest/aider_polyglot_ingest.py (a results_full_polyglot.json exercise key ->
a file under log_root) both resolve a caller-controlled relative path against
a trusted base directory and must refuse anything that would escape it. These
started as two independently-maintained copies of the same resolve() +
is_relative_to() check with different failure behavior (raise vs. log+skip) --
confirmed by review as a real risk: a future hardening fix (e.g. symlink
traversal handling) applied to only one copy leaves the other unpatched.
"""
from pathlib import Path


def resolve_contained_path(base: Path, rel_path: str) -> Path:
    """Join rel_path onto base and verify the result stays inside it.

    An ABSOLUTE rel_path discards base entirely (pathlib's `/` operator
    replaces the left side when the right is absolute), and a relative one
    can still escape via `../` segments -- both are rejected here rather than
    silently resolving outside base.
    """
    base = Path(base).resolve()
    resolved = (base / rel_path).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"path {rel_path!r} escapes base directory {base}")
    return resolved
