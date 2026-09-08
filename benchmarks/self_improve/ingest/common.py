"""Shared helpers for the per-benchmark ingest modules: parsing skill-inject /
knowledge-inject notification lines into ComponentUsage records, and
summarizing a trajectory's assistant text + tool calls for reflection_lm
context (capped length, error calls prioritized).

Notification message formats are fixed by the emitting TS extensions and must
not be re-derived from guesswork -- see TDD_SPEC.md §0 for the confirmed
source references:
  .pi/extensions/skill-inject/index.ts:~388      'skill-inject: +N ["tool1","tool2"]'
  .pi/extensions/knowledge-inject/index.ts:~164   'knowledge-inject: +N ["topic1","topic2"]'

The bracketed payload is a JSON array of strings (real bug, fixed: it used to
be a bare comma-joined list, `[tool1,tool2]` -- ambiguous whenever a name
itself contained a literal comma, which is a real risk for knowledge-inject's
topic strings, arbitrary human frontmatter text like "Error handling,
retries, and backoff"). `_parse_notification_payload()` below tries JSON
first and falls back to the old comma-split ONLY for historical trajectory
data written before this fix -- that data is already ambiguous and can't be
recovered after the fact, so the fallback keeps its old (imperfect) behavior
rather than trying to guess.

Real bug, confirmed by review: skill-inject's names are each card's
`target_tool` FRONTMATTER FIELD (.pi/extensions/skill-inject/index.ts:387,
`selected.map((s) => s.targetTool)`), NOT the file stem -- they only
coincide for tools whose target_tool happens to already be snake_case
(e.g. "bash" -> skills/tools/bash.md). For several real tool skills,
target_tool is CamelCase while the file stem (and config/components.yaml's
pred_name) is snake_case -- e.g. skills/tools/browser_click.md declares
`target_tool: BrowserClick`, so blindly prefixing the emitted name
("skills_tools_" + "BrowserClick") produced a pred_name that matches
nothing in config/components.yaml, silently dropping real usage signal for
every such tool. knowledge-inject's names are each entry's `topic`
FRONTMATTER FIELD (e.g. "Binary Search", "State-Space Search") -- an
arbitrary human string independent of the file's `name`/stem (confirmed
against .pi/extensions/knowledge-inject/index.ts:49 and real
skills/knowledge/*.md, skills/protocols/*.md frontmatter by review). Both
cases need the SAME fix: neither an emitted name can be turned into the
right pred_name by string transformation alone -- build_knowledge_topic_index()
(despite its name, predating this fix -- it indexes BOTH skills/tools/*.md's
target_tool and skills/knowledge|protocols/*.md's topic into one
name->pred_name map) must be called against the real repo and threaded
through to resolve either.
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

# Mirrors .pi/extensions/knowledge-inject/index.ts's dirs(): both directories
# feed the SAME topic->entry registry there, so both must feed the same
# lookup index here. Later directories win on a topic collision, exactly
# matching the TS Map's insertion-order-overwrite semantics. `field` names
# which frontmatter key holds the string the TS extension actually emits in
# a notification line (skill-inject emits target_tool; knowledge-inject
# emits topic, falling back to name) -- see build_knowledge_topic_index()'s
# own docstring for the real bug this fixes.
#: Namespace prefix for each directory's index keys -- the `source` value
#: _NOTIF_RE itself captures ("skill-inject" for tools; knowledge and
#: protocols share "knowledge-inject", exactly matching how
#: knowledge-inject/index.ts's own dirs() feeds both into ONE registry).
#: Real gap, confirmed by review: an earlier version of this index used
#: bare, unnamespaced keys shared across all three directories -- if a
#: tool's target_tool ever happened to equal a knowledge/protocol
#: topic/name (plausible: both are short human-ish strings drawn from
#: independent vocabularies with no coordination), one entry would
#: silently overwrite the other and parse_notification_line (which looks
#: up by name only, without checking `source`) could misattribute usage
#: to the wrong component. `_index_key()` below scopes every lookup/store
#: to its own source, so the two vocabularies can never collide even if
#: their raw strings do.
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
    """Despite its name (predates this fix, kept to avoid an unnecessary
    rename across every call site), this indexes THREE directories, not
    just knowledge/protocols: skills/tools/*.md too, keyed by each card's
    `target_tool` frontmatter field.

    Real bug, confirmed by review: skill-inject notifications report each
    card's `target_tool` (.pi/extensions/skill-inject/index.ts:387,
    `s.targetTool`), which parse_notification_line used to resolve to a
    pred_name by blindly prefixing it ("skills_tools_" + target_tool) --
    correct only when target_tool happens to already equal the file stem
    (true for e.g. "bash"). Several real tool skills declare a CamelCase
    target_tool while the file stem (and config/components.yaml's
    pred_name) is snake_case -- e.g. skills/tools/browser_click.md declares
    `target_tool: BrowserClick` -- so the blind-prefix pred_name matched
    nothing in components.yaml and real usage signal for those tools was
    silently dropped. Exactly the same class of bug knowledge-inject's
    topic already needed this index for; skill-inject needed it too.

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
    knowledge-inject entries missing from it (e.g. a renamed/deleted skill
    file, or no index passed at all) are dropped with a warning rather
    than guessed at -- a topic has no textual relation to its pred_name,
    so there is no reasonable fallback. skill-inject entries missing from
    it instead fall back to the pre-fix blind-prefix guess
    ("skills_tools_" + name): real bug, confirmed by review, an earlier
    version used that blind prefix UNCONDITIONALLY, correct only for tools
    whose target_tool happens to equal the file stem -- several real tool
    skills declare a CamelCase target_tool (e.g. "BrowserClick") while the
    file stem/pred_name is snake_case, so real skill-inject usage for
    those tools was silently attributed to a pred_name matching nothing in
    config/components.yaml. The index (built from real
    skills/tools/*.md target_tool fields) fixes exactly that case; the
    fallback is kept only so a caller that doesn't build/pass an index at
    all (every unit test in this module, and any future caller not yet
    updated) keeps its old, still-usually-correct behavior instead of
    losing every skill-inject usage record outright -- and now logs a
    warning whenever it fires WITH a real, non-empty index that just
    doesn't contain this name (real gap, confirmed by review, three
    rounds: an earlier version warned only on the knowledge-inject drop
    path, so a skill-inject name silently missing from a REAL, present
    index -- e.g. a renamed/deleted tool skill -- corrupted usage signal
    with no trace; the next fix warned UNCONDITIONALLY, collapsing right
    back into being indistinguishable from -- and noisy for -- the
    expected no-index-passed case; the fix after THAT checked the
    caller's argument for `is not None`, which still miscounted an
    intentionally-EMPTY dict (aider_polyglot_ingest.py's/gaia_ingest.py's
    own `build_knowledge_topic_index(repo_root) if repo_root else {}`
    no-repo_root sentinel) as "a real index was provided". Truthiness
    (an empty dict is falsy, same as None) is what actually distinguishes
    "no resolution possible at all" from "resolution possible but this
    specific name is missing").

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

    # Real gap, confirmed by review: `knowledge_topic_index or {}` collapses
    # "no index passed at all" (every unit test in this module; the
    # expected, benign case the fallback exists FOR) and "a real index was
    # passed but doesn't contain this name" (the actually-concerning case
    # -- a renamed/deleted tool skill silently corrupting usage signal)
    # into the exact same `index = {}` -- a warning fired for both would
    # be indistinguishable in the logs (defeating the point) AND spam every
    # no-index caller. Keep the distinction: only warn when the caller
    # actually opted into indexed resolution.
    #
    # Real gap, confirmed by review (second round): `is not None` isn't
    # enough either -- aider_polyglot_ingest.py's and gaia_ingest.py's own
    # `build_knowledge_topic_index(repo_root) if repo_root else {}` pass a
    # genuinely EMPTY dict (not None) as their own "no repo_root given, no
    # resolution possible" sentinel, which `is not None` wrongly counted
    # as "a real index was provided", spamming a warning for every
    # skill-inject usage record in that (common, legitimate) no-repo_root
    # case. An index that resolves nothing at all is just as much "no
    # index" as one that was never passed -- truthiness (empty dict is
    # falsy) treats both sentinels the same way.
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
