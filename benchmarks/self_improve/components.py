"""Read-modify-write logic for the optimizable harness artifacts (AGENTS.md,
PRINCIPLES.md, skill files). Frontmatter is split from body and NEVER
re-serialized through a YAML dumper -- string concatenation only, so a
component whose optimized body is byte-identical to the original round-trips
exactly (TDD_SPEC.md §7.2).

config/components.yaml is the single source of truth mapping pred_name ->
repo-relative file path.
"""
import logging
import math
import re
from pathlib import Path

import yaml

from benchmarks.self_improve.path_safety import resolve_contained_path

_FRONTMATTER_DELIM = "---\n"
#: Same ratio already established for this exact purpose across
#: .pi/extensions/ (thinking-budget's charsToTokens, read-guard's own copy,
#: knowledge-inject's selection logic) -- reused here rather than inventing
#: a different one (a naive len/4 would be off by up to ~99% against real
#: skill files' actual hand-authored token_cost values, which range 3.64-7.96
#: chars/token).
_CHARS_PER_TOKEN = 3.5
_TOKEN_COST_LINE_RE = re.compile(r"(?m)^token_cost:\s*\d+\s*$")


def _estimate_token_cost(body: str) -> int:
    return math.ceil(len(body) / _CHARS_PER_TOKEN)


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Split a file's leading '---\\n...\\n---\\n' YAML block from its body.
    Returns (frontmatter_including_delimiters, body). If the file does not
    start with the delimiter, returns (None, text) unchanged."""
    if not text.startswith(_FRONTMATTER_DELIM):
        return None, text
    end = text.find("\n---\n", len(_FRONTMATTER_DELIM))
    if end == -1:
        return None, text
    end_of_block = end + len("\n---\n")
    frontmatter = text[:end_of_block]
    body = text[end_of_block:]
    return frontmatter, body


def reattach_frontmatter(frontmatter: str | None, body: str) -> str:
    """Exact inverse of split_frontmatter: string concatenation only."""
    if frontmatter is None:
        return body
    return frontmatter + body


def _load_components_yaml(components_yaml_path: Path) -> dict[str, str]:
    return yaml.safe_load(Path(components_yaml_path).read_text()) or {}


def _resolve_component_path(repo_root: Path, rel_path: str) -> Path:
    """Join rel_path onto repo_root and verify the result stays inside it.

    components.yaml is repo-controlled data today, but that alone provides no
    containment (confirmed hardening gap by review) -- this is the only thing
    standing between a components.yaml edit and an arbitrary file
    read/overwrite. See path_safety.resolve_contained_path for the shared
    resolve+validate logic (also used by aider_polyglot_ingest.py).
    """
    try:
        return resolve_contained_path(repo_root, rel_path)
    except ValueError as e:
        raise ValueError(f"component path {rel_path!r} escapes repo_root {Path(repo_root).resolve()}") from e


def load_components(components_yaml_path: Path, repo_root: Path) -> dict[str, str]:
    """Load every component's body text (frontmatter stripped) keyed by
    pred_name, per components.yaml."""
    mapping = _load_components_yaml(components_yaml_path)
    components = {}
    for pred_name, rel_path in mapping.items():
        text = _resolve_component_path(repo_root, rel_path).read_text()
        _frontmatter, body = split_frontmatter(text)
        components[pred_name] = body
    return components


def write_components_back(
    components_yaml_path: Path, repo_root: Path, optimized: dict[str, str]
) -> list[Path]:
    """Write optimized body text back into each component's real file.
    Frontmatter is otherwise preserved untouched EXCEPT its token_cost:
    line, which is recomputed from the new body (see _estimate_token_cost)
    so skill-inject/knowledge-inject's real per-turn injection budget isn't
    silently selecting against a stale, understated cost. Skips (does not
    write) any file whose optimized body is identical to its current body.
    Returns the list of file paths actually changed."""
    mapping = _load_components_yaml(components_yaml_path)
    changed: list[Path] = []
    for pred_name, new_body in optimized.items():
        rel_path = mapping.get(pred_name)
        if rel_path is None:
            # Real gap, confirmed by review: silently dropping this with no
            # signal is how a mismatched components.yaml (e.g. applying a
            # full-scope optimized set against a scoped-down pilot yaml)
            # loses most of a run's results with zero error -- `changed`
            # just comes back shorter than the caller expects.
            logging.getLogger(__name__).warning(
                "write_components_back: %r is not in %s -- skipping (optimized "
                "and components.yaml were likely generated from different scopes)",
                pred_name, components_yaml_path,
            )
            continue
        file_path = _resolve_component_path(repo_root, rel_path)
        current_text = file_path.read_text()
        frontmatter, current_body = split_frontmatter(current_text)
        if new_body == current_body:
            continue
        if frontmatter is not None:
            # Targeted line substitution ONLY -- never a YAML parse+dump.
            # split_frontmatter/reattach_frontmatter's own contract (this
            # module's docstring, TDD_SPEC.md §7.2) guarantees
            # string-concatenation-only round-tripping; a YAML re-dump would
            # risk reformatting quote style/key order/comments on every
            # write and silently break that. A file with no token_cost:
            # line (not expected for real skill files today) is left
            # untouched rather than having one invented.
            new_cost = _estimate_token_cost(new_body)
            frontmatter = _TOKEN_COST_LINE_RE.sub(f"token_cost: {new_cost}", frontmatter, count=1)
        file_path.write_text(reattach_frontmatter(frontmatter, new_body))
        changed.append(file_path)
    return changed
