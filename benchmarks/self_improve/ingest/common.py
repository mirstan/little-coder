"""Shared helpers for the per-benchmark ingest modules: parsing skill-inject /
knowledge-inject notification lines into ComponentUsage records, and
summarizing a trajectory's assistant text + tool calls for reflection_lm
context (capped length, error calls prioritized).

Notification message formats are fixed by the emitting TS extensions and must
not be re-derived from guesswork -- see TDD_SPEC.md §0 for the confirmed
source references:
  .pi/extensions/skill-inject/index.ts:385      "skill-inject: +N [tool1,tool2]"
  .pi/extensions/knowledge-inject/index.ts:156   "knowledge-inject: +N [topic1,topic2]"

skill-inject's bracketed names are the TOOL NAME itself (e.g. "bash"), which
is also the file stem, so pred_name = "skills_tools_" + name works directly.
knowledge-inject's bracketed names are each entry's `topic` FRONTMATTER FIELD
(e.g. "Binary Search", "State-Space Search") -- an arbitrary human string
independent of the file's `name`/stem (confirmed against
.pi/extensions/knowledge-inject/index.ts:49 and real skills/knowledge/*.md,
skills/protocols/*.md frontmatter by review). A topic string can NEVER be
turned into the right pred_name by string transformation alone (e.g.
"State-Space Search" -> file bfs_state_space.md, no textual relation) --
build_knowledge_topic_index() must be called against the real repo and
threaded through to resolve it.
"""
import logging
import re
from pathlib import Path

import yaml

from benchmarks.self_improve.schema import ComponentUsage

logger = logging.getLogger(__name__)

_NOTIF_RE = re.compile(
    r"^\[(?P<level>\w+)\]\s+(?P<source>skill-inject|knowledge-inject):"
    r"\s+(?:\+\d+\s+\[(?P<names>[^\]]*)\])?"
)

_SOURCE_PREFIX = {
    "skill-inject": "skills_tools_",
    "knowledge-inject": "skills_knowledge_",
}

_FRONTMATTER_BLOCK_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

# Mirrors .pi/extensions/knowledge-inject/index.ts's dirs(): both directories
# feed the SAME topic->entry registry there, so both must feed the same
# lookup index here. Later directories win on a topic collision, exactly
# matching the TS Map's insertion-order-overwrite semantics.
_KNOWLEDGE_DIRS = (("knowledge", "skills_knowledge_"), ("protocols", "skills_protocols_"))


def build_knowledge_topic_index(repo_root: Path) -> dict[str, str]:
    """Scan skills/knowledge/*.md and skills/protocols/*.md frontmatter,
    exactly as knowledge-inject's loadEntries() does (topic = fm.topic or
    fm.name), and map each real topic string to the pred_name
    parse_notification_line needs. Missing/malformed files are skipped, not
    raised -- ingest must still work for skill-inject-only data even if this
    index comes back partial or empty."""
    repo_root = Path(repo_root)
    index: dict[str, str] = {}
    for subdir, prefix in _KNOWLEDGE_DIRS:
        dir_path = repo_root / "skills" / subdir
        if not dir_path.is_dir():
            continue
        for file in sorted(dir_path.glob("*.md")):
            try:
                text = file.read_text()
            except (OSError, UnicodeDecodeError) as e:
                # Real bug, confirmed by review: UnicodeDecodeError is NOT an
                # OSError subclass (it's a ValueError) -- a non-UTF-8 skill
                # file previously propagated all the way out of this
                # function, aborting the whole ingest run (gaia/aider both
                # call this once at the top of load()) instead of just
                # skipping the one unreadable file.
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
                # Real bug, confirmed by review: syntactically valid YAML
                # that isn't a mapping (e.g. a bare list or scalar) made
                # frontmatter.get() raise AttributeError, aborting the whole
                # ingest run the same way as the read failure above.
                logger.warning("build_knowledge_topic_index: frontmatter in %s is not a mapping", file)
                continue
            topic = frontmatter.get("topic") or frontmatter.get("name")
            if not isinstance(topic, str) or not topic:
                continue
            index[topic] = f"{prefix}{file.stem}"
    return index


def parse_notification_line(
    line: str, knowledge_topic_index: dict[str, str] | None = None
) -> list[ComponentUsage]:
    """Parse one notifications.txt line into zero or more ComponentUsage
    records (invocation_count=1 each -- caller aggregates via
    merge_component_usage). Never raises; unrelated/unparseable lines
    (quality-monitor, thinking-budget, garbage) return [].

    knowledge_topic_index (from build_knowledge_topic_index()) is REQUIRED to
    resolve knowledge-inject entries to a real pred_name -- without it (or
    for a topic missing from it, e.g. a renamed/deleted skill file), those
    entries are dropped with a warning rather than guessed at."""
    m = _NOTIF_RE.match(line)
    if not m:
        return []
    names = m.group("names")
    if not names:
        return []
    source = m.group("source")

    if source == "knowledge-inject":
        index = knowledge_topic_index or {}
        usages = []
        for name in names.split(","):
            name = name.strip()
            if not name:
                continue
            pred_name = index.get(name)
            if pred_name is None:
                logger.warning(
                    "parse_notification_line: knowledge-inject topic %r not found in "
                    "skills/knowledge or skills/protocols -- dropping usage record", name,
                )
                continue
            usages.append(ComponentUsage(pred_name=pred_name, invocation_count=1))
        return usages

    prefix = _SOURCE_PREFIX[source]
    return [
        ComponentUsage(pred_name=f"{prefix}{name.strip()}", invocation_count=1)
        for name in names.split(",")
        if name.strip()
    ]


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
