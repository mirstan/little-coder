from benchmarks.self_improve.ingest.common import (
    merge_component_usage,
    parse_notification_line,
    summarize_for_reflection,
)
from benchmarks.self_improve.schema import ComponentUsage


def test_parse_skill_inject_notification_with_tools():
    usages = parse_notification_line("[info] skill-inject: +2 [bash,read]")
    assert usages == [
        ComponentUsage(pred_name="skills_tools_bash", invocation_count=1),
        ComponentUsage(pred_name="skills_tools_read", invocation_count=1),
    ]


def test_parse_skill_inject_notification_research_directive_only():
    """selected.length === 0 case: no bracket group present at all."""
    usages = parse_notification_line("[info] skill-inject: +research-directive")
    assert usages == []


def test_parse_knowledge_inject_notification():
    usages = parse_notification_line("[info] knowledge-inject: +2 [binary_search,two_pointer]")
    assert usages == [
        ComponentUsage(pred_name="skills_knowledge_binary_search", invocation_count=1),
        ComponentUsage(pred_name="skills_knowledge_two_pointer", invocation_count=1),
    ]


def test_parse_notification_line_ignores_unrelated_lines():
    """quality-monitor / thinking-budget notifications exist but aren't
    component-usage signals for THIS purpose -- must not raise, must return []."""
    assert parse_notification_line("[warn] quality-monitor: empty response detected") == []
    assert parse_notification_line("garbage line with no brackets at all") == []


def test_merge_component_usage_sums_counts_across_lines():
    """A skill injected in 3 different turns of one trajectory must
    aggregate into ONE ComponentUsage with invocation_count=3, not three
    separate entries."""
    lines = [
        "[info] skill-inject: +1 [bash]",
        "[info] skill-inject: +1 [bash]",
        "[info] skill-inject: +1 [read]",
    ]
    merged = merge_component_usage(lines)
    by_name = {u.pred_name: u.invocation_count for u in merged}
    assert by_name == {"skills_tools_bash": 2, "skills_tools_read": 1}


def test_mark_error_context_flags_usage_immediately_after_tool_error():
    """was_error_context=True when merge_component_usage is told this batch
    of lines follows a tool error."""
    notif_lines = ["[info] skill-inject: +1 [bash]"]
    merged = merge_component_usage(notif_lines, follows_error=True)
    assert merged[0].was_error_context is True


def test_summarize_for_reflection_caps_length():
    long_text = "x" * 50_000
    out = summarize_for_reflection(assistant_text=long_text, tool_calls=[], cap=8_000)
    assert len(out) <= 8_000


def test_summarize_for_reflection_prioritizes_error_tool_calls():
    """When truncating, error tool calls must survive in the summary even if
    they occurred early in a long trajectory -- this is the material
    reflection_lm actually needs."""
    tool_calls = [
        {"name": "bash", "args": {"command": "flaky"}, "result_text": "boom", "is_error": True},
        *[{"name": "read", "args": {}, "result_text": "ok", "is_error": False} for _ in range(500)],
    ]
    out = summarize_for_reflection(assistant_text="", tool_calls=tool_calls, cap=2_000)
    assert "boom" in out
    assert "flaky" in out


def test_summarize_for_reflection_handles_empty_input():
    assert summarize_for_reflection(assistant_text="", tool_calls=[], cap=8_000) == ""
