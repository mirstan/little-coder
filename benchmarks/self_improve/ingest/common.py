"""Shared helpers for the per-benchmark ingest modules: parsing skill-inject /
knowledge-inject notification lines into ComponentUsage records, and
summarizing a trajectory's assistant text + tool calls for reflection_lm
context (capped length, error calls prioritized).

Notification message formats are fixed by the emitting TS extensions and must
not be re-derived from guesswork -- see TDD_SPEC.md §0 for the confirmed
source references:
  .pi/extensions/skill-inject/index.ts:385      "skill-inject: +N [tool1,tool2]"
  .pi/extensions/knowledge-inject/index.ts:156   "knowledge-inject: +N [topic1,topic2]"
"""
import re

from benchmarks.self_improve.schema import ComponentUsage

_NOTIF_RE = re.compile(
    r"^\[(?P<level>\w+)\]\s+(?P<source>skill-inject|knowledge-inject):"
    r"\s+(?:\+\d+\s+\[(?P<names>[^\]]*)\])?"
)

_SOURCE_PREFIX = {
    "skill-inject": "skills_tools_",
    "knowledge-inject": "skills_knowledge_",
}


def parse_notification_line(line: str) -> list[ComponentUsage]:
    """Parse one notifications.txt line into zero or more ComponentUsage
    records (invocation_count=1 each -- caller aggregates via
    merge_component_usage). Never raises; unrelated/unparseable lines
    (quality-monitor, thinking-budget, garbage) return []."""
    m = _NOTIF_RE.match(line)
    if not m:
        return []
    names = m.group("names")
    if not names:
        return []
    prefix = _SOURCE_PREFIX[m.group("source")]
    return [
        ComponentUsage(pred_name=f"{prefix}{name.strip()}", invocation_count=1)
        for name in names.split(",")
        if name.strip()
    ]


def merge_component_usage(
    lines: list[str], follows_error: bool = False
) -> list[ComponentUsage]:
    """Parse every line and aggregate by pred_name: sum invocation_count,
    OR was_error_context across contributing lines."""
    counts: dict[str, int] = {}
    error_flags: dict[str, bool] = {}
    for line in lines:
        for usage in parse_notification_line(line):
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
