"""Shared helpers for the per-benchmark ingest modules: parsing skill-inject /
knowledge-inject notification lines into ComponentUsage records, and
summarizing a trajectory's assistant text + tool calls for reflection_lm
context (capped length, error calls prioritized).

Notification message formats are fixed by the emitting TS extensions and must
not be re-derived from guesswork -- see TDD_SPEC.md §0 for the confirmed
source references:
  .pi/extensions/skill-inject/index.ts        'skill-inject: +N ["tool1","tool2"]'
  .pi/extensions/knowledge-inject/index.ts    'knowledge-inject: +N ["topic1","topic2"]'

The bracketed payload is a JSON array of strings. Historical trajectory data
predates that and carries a bare comma-joined list instead, ambiguous
whenever a name itself contains a comma (real for knowledge-inject topics,
which are arbitrary human frontmatter text like "Error handling, retries,
and backoff"). `_parse_notification_payload()` tries JSON first and falls
back to the comma-split for that old data, ambiguity included, since it
can't be recovered after the fact.

The names in a notification are frontmatter FIELDS, not file stems:
skill-inject emits each card's `target_tool`, knowledge-inject each entry's
`topic`. Neither can be turned into a components.yaml pred_name by string
transformation -- see build_knowledge_topic_index(), which must be built
against the real repo and threaded through to resolve either.
"""
import json
import logging
import re
from pathlib import Path

import yaml

from benchmarks.self_improve.schema import ComponentUsage

logger = logging.getLogger(__name__)

_NOTIF_RE = re.compile(
    r"^\[(?P<level>\w+)\]\s+(?P<source>skill-inject|knowledge-inject):"
    r"\s+(?:\+\d+\s+(?P<payload>\[.*\]))?"
)


def _parse_notification_payload(payload: str) -> list[str]:
    """New format: a JSON array of strings -- unambiguous even when a name
    itself contains a comma (see this module's own docstring). Old format
    (real historical trajectory data predates this fix and will keep
    showing up): a bare bracketed, comma-joined list -- kept as an
    exact-behavior fallback, including its known ambiguity for a name
    containing a literal comma, since that data is already ambiguous and
    can't be recovered after the fact."""
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return [n.strip() for n in parsed if n.strip()]

    stripped = payload.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    return [n.strip() for n in stripped.split(",") if n.strip()]

_FRONTMATTER_BLOCK_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

# (subdir, pred_name prefix, frontmatter field, notification source).
#
# knowledge and protocols share the "knowledge-inject" source because
# knowledge-inject/index.ts's own dirs() feeds both into ONE registry;
# later directories win on a topic collision, matching that TS Map's
# insertion-order-overwrite semantics.
#
# `field` is the frontmatter key holding the string the extension actually
# emits (skill-inject: target_tool; knowledge-inject: topic, falling back to
# name). Keys are namespaced by source via _index_key() because the two
# vocabularies are independent short human strings: an unnamespaced index
# would let a tool's target_tool overwrite an equal knowledge topic, and
# parse_notification_line (which looks up by name) would misattribute usage
# to the wrong component.
_INDEXED_DIRS = (
    ("tools", "skills_tools_", "target_tool", "skill-inject"),
    ("knowledge", "skills_knowledge_", "topic", "knowledge-inject"),
    ("protocols", "skills_protocols_", "topic", "knowledge-inject"),
)


def _index_key(source: str, name: str) -> str:
    """Namespaces an index key by notification source -- see _INDEXED_DIRS'
    own comment for why. NUL is not valid in YAML frontmatter scalar text
    (and not realistically typeable into a `name`/`topic`/`target_tool`
    field), so it can't collide with either vocabulary's own content."""
    return f"{source}\x00{name}"


def build_knowledge_topic_index(repo_root: Path) -> dict[str, str]:
    """Despite its name, this indexes THREE directories, not just
    knowledge/protocols: skills/tools/*.md too, keyed by each card's
    `target_tool` frontmatter field.

    target_tool only sometimes equals the file stem (true for "bash"), so a
    name cannot be turned into a pred_name by prefixing: several real tool
    skills declare a CamelCase target_tool while the file stem -- and
    config/components.yaml's pred_name -- is snake_case, e.g.
    skills/tools/browser_click.md declaring `target_tool: BrowserClick`.
    knowledge-inject's `topic` has no textual relation to its file at all.

    Scans frontmatter exactly as the corresponding TS loader does for each
    directory (skill-inject's cards, knowledge-inject's loadEntries():
    topic = fm.topic or fm.name) and maps each real emitted-name string to
    the pred_name parse_notification_line needs. Missing/malformed files
    are skipped, not raised -- ingest must still work even if this index
    comes back partial or empty."""
    repo_root = Path(repo_root)
    index: dict[str, str] = {}
    for subdir, prefix, field, source in _INDEXED_DIRS:
        dir_path = repo_root / "skills" / subdir
        if not dir_path.is_dir():
            continue
        for file in sorted(dir_path.glob("*.md")):
            try:
                text = file.read_text()
            except (OSError, UnicodeDecodeError) as e:
                # UnicodeDecodeError is a ValueError, not an OSError: caught
                # here so one non-UTF-8 skill file is skipped rather than
                # aborting the whole ingest run (gaia/aider both call this
                # once at the top of load()).
                logger.warning("build_knowledge_topic_index: failed to read %s: %s", file, e)
                continue
            m = _FRONTMATTER_BLOCK_RE.match(text)
            if not m:
                continue
            try:
                frontmatter = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError as e:
                logger.warning("build_knowledge_topic_index: malformed frontmatter in %s: %s", file, e)
                continue
            if not isinstance(frontmatter, dict):
                # Syntactically valid YAML that isn't a mapping (a bare list
                # or scalar) would make frontmatter.get() raise
                # AttributeError, aborting the run like the read failure
                # above.
                logger.warning("build_knowledge_topic_index: frontmatter in %s is not a mapping", file)
                continue
            key = frontmatter.get(field)
            if field == "topic":
                key = key or frontmatter.get("name")
            if not isinstance(key, str) or not key:
                continue
            index[_index_key(source, key)] = f"{prefix}{file.stem}"
    return index


def parse_notification_line(
    line: str, knowledge_topic_index: dict[str, str] | None = None
) -> list[ComponentUsage]:
    """Parse one notifications.txt line into zero or more ComponentUsage
    records (invocation_count=1 each -- caller aggregates via
    merge_component_usage). Never raises; unrelated/unparseable lines
    (quality-monitor, thinking-budget, garbage) return [].

    knowledge_topic_index (from build_knowledge_topic_index(), despite its
    name -- see that function's own docstring) resolves BOTH skill-inject
    (target_tool) and knowledge-inject (topic) entries to a real pred_name.

    A knowledge-inject entry missing from it is dropped with a warning: a
    topic has no textual relation to its pred_name, so there is no
    reasonable fallback. A skill-inject entry missing from it falls back to
    a blind "skills_tools_" + name guess, which is right only when
    target_tool equals the file stem, so that a caller with no index at all
    (every unit test here) keeps usable output instead of losing every
    skill-inject record. That fallback warns only when a NON-EMPTY index was
    provided, i.e. resolution was possible and this one name was missing --
    aider_polyglot_ingest.py and gaia_ingest.py pass an empty dict as their
    own "no repo_root, no resolution possible" sentinel, so truthiness, not
    `is not None`, is what separates the two cases.

    Index keys are namespaced by source (see _index_key()) so a tool's
    target_tool can never collide with an unrelated knowledge/protocol
    topic or name that happens to share the same raw string."""
    m = _NOTIF_RE.match(line)
    if not m:
        return []
    payload = m.group("payload")
    if not payload:
        return []
    names = _parse_notification_payload(payload)
    if not names:
        return []
    source = m.group("source")

    # Captured separately from `index` below because the fallback warning
    # must distinguish "no resolution was possible" from "resolution was
    # possible and this name was missing" -- see the docstring for why
    # truthiness, not `is not None`, is the right test.
    index_was_provided = bool(knowledge_topic_index)
    index = knowledge_topic_index or {}
    usages = []
    for name in names:
        pred_name = index.get(_index_key(source, name))
        if pred_name is None:
            if source == "skill-inject":
                if index_was_provided:
                    logger.warning(
                        "parse_notification_line: skill-inject target_tool %r not found "
                        "in the provided tool-skill index (e.g. a renamed/deleted tool "
                        "skill) -- falling back to a blind name->pred_name guess "
                        "(skills_tools_%s), which is wrong whenever target_tool differs "
                        "from the file stem", name, name,
                    )
                pred_name = f"skills_tools_{name}"
            else:
                logger.warning(
                    "parse_notification_line: knowledge-inject topic %r not found in "
                    "skills/knowledge or skills/protocols -- dropping usage record", name,
                )
                continue
        usages.append(ComponentUsage(pred_name=pred_name, invocation_count=1))
    return usages


def merge_component_usage(
    lines: list[str],
    follows_error: bool = False,
    knowledge_topic_index: dict[str, str] | None = None,
) -> list[ComponentUsage]:
    """Parse every line and aggregate by pred_name: sum invocation_count,
    OR was_error_context across contributing lines."""
    counts: dict[str, int] = {}
    error_flags: dict[str, bool] = {}
    for line in lines:
        for usage in parse_notification_line(line, knowledge_topic_index):
            counts[usage.pred_name] = counts.get(usage.pred_name, 0) + usage.invocation_count
            error_flags[usage.pred_name] = error_flags.get(usage.pred_name, False) or follows_error
    return [
        ComponentUsage(
            pred_name=name,
            invocation_count=count,
            was_error_context=error_flags[name],
        )
        for name, count in counts.items()
    ]


def summarize_for_reflection(
    assistant_text: str, tool_calls: list[dict], cap: int = 8_000
) -> str:
    """Build a bounded-length summary for reflection_lm: error tool calls
    first (they're the material reflection actually needs), then backfill
    with the tail of assistant_text. Never raises on empty input."""
    if not assistant_text and not tool_calls:
        return ""

    parts: list[str] = []
    remaining = cap

    error_calls = [tc for tc in tool_calls if tc.get("is_error")]
    for tc in error_calls:
        if remaining <= 0:
            break
        excerpt = (
            f"[ERROR] {tc.get('name')}({tc.get('args')}) -> {tc.get('result_text')}"
        )
        excerpt = excerpt[:remaining]
        parts.append(excerpt)
        remaining -= len(excerpt)

    if assistant_text and remaining > 0:
        tail = assistant_text[-remaining:]
        parts.append(tail)
        remaining -= len(tail)

    out = "\n".join(parts)
    return out[:cap]
