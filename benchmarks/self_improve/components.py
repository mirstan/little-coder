"""Read-modify-write logic for the optimizable harness artifacts (AGENTS.md,
PRINCIPLES.md, skill files). Frontmatter is split from body and NEVER
re-serialized through a YAML dumper -- string concatenation only, so a
component whose optimized body is byte-identical to the original round-trips
exactly (TDD_SPEC.md §7.2).

config/components.yaml is the single source of truth mapping pred_name ->
repo-relative file path.
"""
import logging
import re
from pathlib import Path

import yaml

from benchmarks.self_improve.path_safety import resolve_contained_path

_FRONTMATTER_DELIM = "---\n"
#: [ \t]* rather than \s* on BOTH sides deliberately: \s matches "\n" too,
#: so a greedy trailing \s*$ would swallow the line's own newline (plus any
#: blank line after it) into the match and the replacement -- silently
#: deleting frontmatter whitespace this module's round-trip contract
#: promises to preserve byte-for-byte. Capture group lets the SAME pattern
#: read the current value (.search + group) and rewrite it (.sub) -- a group
#: is harmless for .sub() since the replacement string never backreferences it.
_TOKEN_COST_LINE_RE = re.compile(r"(?m)^token_cost:[ \t]*(\d+)[ \t]*$")


def _current_token_cost(frontmatter: str) -> int | None:
    m = _TOKEN_COST_LINE_RE.search(frontmatter)
    return int(m.group(1)) if m else None


def _estimate_token_cost(old_cost: int, old_body: str, new_body: str) -> int:
    """Rescale the file's OWN already-correct, hand-calibrated cost
    proportionally to how much the body's length changed, rather than
    deriving an absolute chars/token estimate from scratch.

    Real bug, confirmed by review (and independently verified against
    every real skills/**/*.md file): there is no single reliable
    chars/token ratio for this corpus -- measured token_cost/body_len
    ratios range 3.59-11.16 (median 6.23, mean 6.44) across all 31 files.
    An earlier version of this function used a flat 3.5 (borrowed from
    .pi/extensions/'s chars-to-TOKEN-BUDGET estimator, calibrated for a
    different purpose entirely), which inflated most real files' cost by
    roughly 1.78x even for an UNCHANGED-length body -- pushing several
    skills (edit.md, write.md, dispatch.md, shell_start.md) permanently
    past skill-inject's real injection budget the moment GEPA touched them,
    so a mutation could score as if the skill had vanished from context
    entirely. Rescaling from each file's own baseline sidesteps
    miscalibration by construction: a same-length edit leaves cost
    unchanged, and only genuine growth/shrinkage moves it, in proportion to
    that file's own already-correct human calibration."""
    if not old_body:
        return old_cost
    return max(1, round(old_cost * len(new_body) / len(old_body)))


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


def load_component_token_costs(components_yaml_path: Path, repo_root: Path) -> dict[str, int]:
    """Each component's CURRENT (pristine, hand-authored) token_cost, keyed
    by pred_name -- the stable baseline _estimate_token_cost's
    rescale-from-original approach needs, and what PolyglotGEPAAdapter
    threads through so make_reflective_dataset can report an accurate live
    estimate without depending on scratch-worktree state (which a cache hit
    can leave stale or mid-candidate, or a prior write in the same run can
    already have rescaled). Components with no frontmatter (agents_md) or
    no token_cost line are simply omitted -- callers already treat a
    missing entry as "no budget applies here"."""
    mapping = _load_components_yaml(components_yaml_path)
    costs: dict[str, int] = {}
    for pred_name, rel_path in mapping.items():
        text = _resolve_component_path(repo_root, rel_path).read_text()
        frontmatter, _body = split_frontmatter(text)
        if frontmatter is None:
            continue
        cost = _current_token_cost(frontmatter)
        if cost is not None:
            costs[pred_name] = cost
    return costs


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
            # untouched rather than having one invented. current_text is
            # read fresh at the top of this loop every call -- for the live
            # GEPA loop this is always the pristine worktree copy (reset()
            # runs before every materialize()), so old_cost/current_body
            # here are always the real hand-authored baseline, never a
            # previously-rescaled value from an earlier candidate this run.
            old_cost = _current_token_cost(frontmatter)
            if old_cost is not None:
                new_cost = _estimate_token_cost(old_cost, current_body, new_body)
                frontmatter = _TOKEN_COST_LINE_RE.sub(f"token_cost: {new_cost}", frontmatter, count=1)
        file_path.write_text(reattach_frontmatter(frontmatter, new_body))
        changed.append(file_path)
    return changed
