from pathlib import Path

from benchmarks.self_improve.ingest.common import (
    build_knowledge_topic_index,
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


def test_parse_knowledge_inject_notification_resolves_via_topic_index():
    """Real notification format, confirmed by review: knowledge-inject's
    bracketed names are each entry's `topic` FRONTMATTER FIELD (e.g. "Binary
    Search"), an arbitrary human string independent of the file's name/stem
    -- never a slug. pred_name can only be resolved via a topic index built
    from the real skill files (build_knowledge_topic_index()), never by
    string transformation of the topic itself."""
    index = {
        "Binary Search": "skills_knowledge_binary_search",
        "Two Pointers": "skills_knowledge_two_pointers",
    }
    usages = parse_notification_line(
        "[info] knowledge-inject: +2 [Binary Search,Two Pointers]", knowledge_topic_index=index,
    )
    assert usages == [
        ComponentUsage(pred_name="skills_knowledge_binary_search", invocation_count=1),
        ComponentUsage(pred_name="skills_knowledge_two_pointers", invocation_count=1),
    ]


def test_parse_knowledge_inject_notification_drops_unresolved_topic():
    """Without a topic index (or a topic missing from it -- e.g. a renamed
    skill file), the usage record is dropped rather than guessed at."""
    assert parse_notification_line("[info] knowledge-inject: +1 [Some Unknown Topic]") == []


def _write_skill_file(path, name, topic=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = f"name: {name}\n" + (f"topic: {topic}\n" if topic else "")
    path.write_text(f"---\n{fm}---\nBody text.\n")


def test_build_knowledge_topic_index_maps_topic_field_to_pred_name(tmp_path):
    _write_skill_file(tmp_path / "skills" / "knowledge" / "binary_search.md",
                       "binary-search", topic="Binary Search")
    index = build_knowledge_topic_index(tmp_path)
    assert index == {"Binary Search": "skills_knowledge_binary_search"}


def test_build_knowledge_topic_index_falls_back_to_name_field_when_no_topic(tmp_path):
    """skills/protocols/*.md files have no `topic:` field at all (confirmed
    against real files) -- topic falls back to `name`, exactly matching
    .pi/extensions/knowledge-inject/index.ts:49-50."""
    _write_skill_file(tmp_path / "skills" / "protocols" / "cite_before_answer.md",
                       "cite-before-answer")
    index = build_knowledge_topic_index(tmp_path)
    assert index == {"cite-before-answer": "skills_protocols_cite_before_answer"}


def test_build_knowledge_topic_index_handles_missing_directories(tmp_path):
    assert build_knowledge_topic_index(tmp_path) == {}


def test_build_knowledge_topic_index_against_real_repo_files():
    """End-to-end against the REAL skills/knowledge and skills/protocols
    files, not fixtures -- confirms the index actually resolves the same
    topic strings real notification lines carry."""
    real_repo_root = Path(__file__).parent.parent.parent.parent  # little-coder-self-improve/
    index = build_knowledge_topic_index(real_repo_root)
    assert index["Binary Search"] == "skills_knowledge_binary_search"
    assert index["cite-before-answer"] == "skills_protocols_cite_before_answer"


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
