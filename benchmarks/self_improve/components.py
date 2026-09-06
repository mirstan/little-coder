"""Read-modify-write logic for the optimizable harness artifacts (AGENTS.md,
PRINCIPLES.md, skill files). Frontmatter is split from body and NEVER
re-serialized through a YAML dumper -- string concatenation only, so a
component whose optimized body is byte-identical to the original round-trips
exactly (TDD_SPEC.md §7.2).

config/components.yaml is the single source of truth mapping pred_name ->
repo-relative file path.
"""
import logging
from pathlib import Path

import yaml

from benchmarks.self_improve.path_safety import resolve_contained_path

_FRONTMATTER_DELIM = "---\n"


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
    """Write optimized body text back into each component's real file,
    preserving frontmatter untouched. Skips (does not write) any file whose
    optimized body is identical to its current body. Returns the list of
    file paths actually changed."""
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
        file_path.write_text(reattach_frontmatter(frontmatter, new_body))
        changed.append(file_path)
    return changed
