import json
from pathlib import Path

from benchmarks.self_improve.ingest.common import (
    build_knowledge_topic_index,
    merge_component_usage,
    parse_notification_line,
    summarize_for_reflection,
)
from benchmarks.self_improve.schema import ComponentUsage


def test_parse_skill_inject_notification_with_tools():
    """No index passed -- falls back to the pre-fix blind-prefix guess,
    which happens to be correct here since "bash"/"read" are also their
    own file stems. See test_parse_skill_inject_notification_resolves_camelcase_target_tool_via_index
    below for the real bug this index exists to fix."""
    usages = parse_notification_line("[info] skill-inject: +2 [bash,read]")
    assert usages == [
        ComponentUsage(pred_name="skills_tools_bash", invocation_count=1),
        ComponentUsage(pred_name="skills_tools_read", invocation_count=1),
    ]


def test_parse_skill_inject_notification_resolves_camelcase_target_tool_via_index():
    """Real bug, confirmed by review: skill-inject notifications report
    each card's `target_tool` frontmatter field (.pi/extensions/skill-inject/index.ts:387),
    NOT the file stem -- an earlier version resolved it by blindly
    prefixing the emitted name, correct only when target_tool happens to
    equal the file stem. Several real tool skills declare a CamelCase
    target_tool (e.g. skills/tools/browser_click.md's `target_tool:
    BrowserClick`) while the file stem/pred_name is snake_case, so real
    usage for those tools was silently attributed to a pred_name matching
    nothing in config/components.yaml. With an index available (built from
    the real skills/tools/*.md files), it must resolve correctly instead
    of falling back to the blind (wrong) guess."""
    index = {"BrowserClick": "skills_tools_browser_click"}
    usages = parse_notification_line(
        "[info] skill-inject: +1 [BrowserClick]", knowledge_topic_index=index,
    )
    assert usages == [ComponentUsage(pred_name="skills_tools_browser_click", invocation_count=1)]


def test_parse_skill_inject_notification_falls_back_when_target_tool_missing_from_index():
    """A target_tool not in the index (e.g. no index passed at all, or a
    renamed/deleted tool skill) degrades to the old blind-prefix guess
    rather than dropping the record -- unlike knowledge-inject, which has
    no reasonable fallback at all (a topic has no textual relation to its
    pred_name), skill-inject's blind prefix is at least USUALLY correct,
    so keeping it as a fallback (rather than always dropping) avoids
    regressing every caller that hasn't been updated to pass an index."""
    usages = parse_notification_line(
        "[info] skill-inject: +1 [bash]", knowledge_topic_index={"BrowserClick": "skills_tools_browser_click"},
    )
    assert usages == [ComponentUsage(pred_name="skills_tools_bash", invocation_count=1)]


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


def test_parse_skill_inject_notification_new_json_format():
    """Real notification format going forward -- both emitting TS extensions
    now JSON.stringify() the name list instead of comma-joining it (see
    ingest/common.py's own module docstring)."""
    payload = json.dumps(["bash", "read"])
    usages = parse_notification_line(f"[info] skill-inject: +2 {payload}")
    assert usages == [
        ComponentUsage(pred_name="skills_tools_bash", invocation_count=1),
        ComponentUsage(pred_name="skills_tools_read", invocation_count=1),
    ]


def test_parse_knowledge_inject_notification_handles_a_topic_containing_a_literal_comma():
    """The exact bug this JSON switch fixes: a knowledge-inject topic is
    arbitrary human frontmatter text and can contain a literal comma (e.g.
    "Error handling, retries, and backoff"). The old bare comma-join format
    was genuinely ambiguous here; JSON.stringify()/json.loads() are not."""
    index = {
        "Error handling, retries, and backoff": "skills_knowledge_error_handling",
        "Binary Search": "skills_knowledge_binary_search",
    }
    payload = json.dumps(["Error handling, retries, and backoff", "Binary Search"])
    usages = parse_notification_line(
        f"[info] knowledge-inject: +2 {payload}", knowledge_topic_index=index,
    )
    assert usages == [
        ComponentUsage(pred_name="skills_knowledge_error_handling", invocation_count=1),
        ComponentUsage(pred_name="skills_knowledge_binary_search", invocation_count=1),
    ]


def test_parse_knowledge_inject_notification_still_reads_old_comma_joined_format():
    """Backward compat: real historical trajectory data written before the
    JSON switch used a bare comma-joined bracket list and will keep showing
    up. Must still parse (with its old, already-baked-in ambiguity for a
    topic containing a literal comma -- that data can't be recovered after
    the fact, only new data is fixed)."""
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


def _write_skill_file(path, name, topic=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = f"name: {name}\n" + (f"topic: {topic}\n" if topic else "")
    path.write_text(f"---\n{fm}---\nBody text.\n")


def _write_tool_skill_file(path, name, target_tool):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ntarget_tool: {target_tool}\n---\nBody text.\n")


def test_build_knowledge_topic_index_maps_target_tool_field_to_pred_name(tmp_path):
    """Real bug, confirmed by review: this index (despite its name) must
    also cover skills/tools/*.md, keyed by target_tool -- not just
    skills/knowledge and skills/protocols -- since skill-inject
    notifications need the SAME target_tool->pred_name resolution
    knowledge-inject's topic already required."""
    _write_tool_skill_file(tmp_path / "skills" / "tools" / "browser_click.md",
                            "browser-click-guidance", target_tool="BrowserClick")
    index = build_knowledge_topic_index(tmp_path)
    assert index == {"BrowserClick": "skills_tools_browser_click"}


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


def test_build_knowledge_topic_index_skips_non_utf8_file(tmp_path):
    """Real bug, confirmed by review: UnicodeDecodeError is NOT an OSError
    subclass -- a non-UTF-8 skill file previously propagated all the way out
    of this function, aborting the whole ingest run instead of just skipping
    the one unreadable file."""
    good_dir = tmp_path / "skills" / "knowledge"
    good_dir.mkdir(parents=True)
    _write_skill_file(good_dir / "good.md", "good-skill", topic="Good Topic")
    (good_dir / "bad.md").write_bytes(b"---\nname: bad\ntopic: Bad\n---\n\xff\xfe not utf-8 \x80")
    index = build_knowledge_topic_index(tmp_path)
    assert index == {"Good Topic": "skills_knowledge_good"}


def test_build_knowledge_topic_index_skips_non_mapping_frontmatter(tmp_path):
    """Real bug, confirmed by review: syntactically valid YAML that isn't a
    mapping (e.g. a bare list) made frontmatter.get() raise AttributeError,
    aborting the whole ingest run."""
    knowledge_dir = tmp_path / "skills" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "list_frontmatter.md").write_text("---\n- not\n- a\n- mapping\n---\nBody.\n")
    assert build_knowledge_topic_index(tmp_path) == {}


def test_build_knowledge_topic_index_skips_non_string_topic(tmp_path):
    knowledge_dir = tmp_path / "skills" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "weird.md").write_text("---\nname: weird\ntopic: [not, a, string]\n---\nBody.\n")
    assert build_knowledge_topic_index(tmp_path) == {}


def test_build_knowledge_topic_index_against_real_repo_files():
    """End-to-end against the REAL skills/knowledge, skills/protocols, and
    skills/tools files, not fixtures -- confirms the index actually
    resolves the same topic/target_tool strings real notification lines
    carry."""
    real_repo_root = Path(__file__).parent.parent.parent.parent  # little-coder-self-improve/
    index = build_knowledge_topic_index(real_repo_root)
    assert index["Binary Search"] == "skills_knowledge_binary_search"
    assert index["cite-before-answer"] == "skills_protocols_cite_before_answer"
    # The real bug this index exists to fix for skill-inject: several real
    # tool skills declare a CamelCase target_tool that does NOT match their
    # snake_case file stem/pred_name.
    assert index["BrowserClick"] == "skills_tools_browser_click"
    assert index["ShellSession"] == "skills_tools_shell_session"
    # And the common case (target_tool already equals the stem) resolves
    # the same way, via the same index.
    assert index["bash"] == "skills_tools_bash"


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
