"""turn_ledger.observe/summarize: pi RPC events (shapes from pi-ai dist/types.d.ts
and pi-coding-agent agent-session.d.ts) folded into per-turn records."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import turn_ledger as L  # noqa: E402


def usage(input=0, output=0, cache_read=0, cache_write=0, reasoning=None):
    u = {"input": input, "output": output, "cacheRead": cache_read, "cacheWrite": cache_write,
         "totalTokens": input + output + cache_read + cache_write,
         "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}
    if reasoning is not None:
        u["reasoning"] = reasoning
    return u


def assistant(content, u=None, stop="toolUse", error=None):
    msg = {"role": "assistant", "content": content, "api": "openai-completions",
           "provider": "omlx", "model": "tiel-coder-oq6e-fp16",
           "usage": u if u is not None else usage(), "stopReason": stop,
           "timestamp": 1759860247398}
    if error is not None:
        msg["errorMessage"] = error
    return msg


def telemetry(data):
    return {"type": "entry_appended", "entry": {
        "type": "custom", "customType": "lc-telemetry", "data": {"v": 1, **data},
        "id": "a1b2c3d4", "parentId": None, "timestamp": "2026-10-07T18:04:07.398Z"}}


def retention(demoted, large=8):
    return telemetry({"kind": "shell_retention", "pairs": large, "large": large, "due": demoted,
                      "prefix": demoted, "demoted": demoted, "flushed": False, "signed": 0,
                      "skippedNoShrink": 0, "skippedArchive": 0,
                      "bytesBefore": demoted * 4096, "bytesAfter": demoted * 900})


def turn_end(msg, results=()):
    return {"type": "turn_end", "message": msg, "toolResults": list(results)}


def run(events):
    st = L.new_ledger()
    recs = []
    for now, ev in events:
        recs.extend(L.observe(st, ev, now))
    return st, recs


def test_an_empty_ledger_summarizes_to_zeros():
    assert L.summarize(L.LedgerState()) == {
        "v": 1, "n_turns": 0, "orphan_turns": 0, "peak_prompt_tokens": 0,
        "peak_context_tokens": 0, "max_output_tokens": 0, "max_tool_arg_bytes": 0,
        "max_ttft_s": None, "stop_reasons": {}, "telemetry_seen": False,
        "max_large_pairs": 0, "max_demoted": 0, "demoted_new_total": 0,
        "max_bytes_saved": 0, "max_signed": 0, "stubbed_new_total": 0,
        "guard_fires": {}, "compactions": {"pi": 0, "harness": 0, "harness_failed": 0},
    }


def test_a_turn_record_carries_usage_timing_and_content_sizes():
    msg = assistant(
        [{"type": "thinking", "thinking": "t" * 50, "thinkingSignature": "reasoning_content"},
         {"type": "text", "text": "hello"},
         {"type": "toolCall", "id": "call_1", "name": "ShellSession", "arguments": {"command": "ls -la"}}],
        usage(input=1000, output=900, cache_read=45000, reasoning=300),
    )
    tool_result = {"role": "toolResult", "toolCallId": "call_1", "toolName": "ShellSession",
                   "content": [{"type": "text", "text": "total 0"}], "isError": True, "timestamp": 1}
    _, recs = run([
        (100.0, {"type": "turn_start"}),
        (101.0, {"type": "message_start", "message": msg}),
        (103.0, {"type": "message_update", "assistantMessageEvent": {"type": "thinking_start", "contentIndex": 0}, "message": msg}),
        (104.0, {"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": 0, "delta": "t"}, "message": msg}),
        (110.0, {"type": "message_end", "message": msg}),
        (110.5, {"type": "message_start", "message": tool_result}),
        (110.6, {"type": "message_end", "message": tool_result}),
        (112.0, turn_end(msg, [tool_result])),
    ])
    assert len(recs) == 1
    r = recs[0]
    assert (r["v"], r["turn"]) == (1, 1)
    assert (r["ts_start"], r["ts_end"], r["iso_end"]) == (100.0, 112.0, "1970-01-01T00:01:52.000Z")
    assert (r["headers_s"], r["ttft_s"], r["gen_s"], r["turn_s"]) == (1.0, 3.0, 10.0, 12.0)
    assert (r["stop_reason"], r["error_message"], r["usage_reported"]) == ("toolUse", None, True)
    assert (r["input"], r["cache_read"], r["cache_write"], r["output"], r["reasoning"]) == (1000, 45000, 0, 900, 300)
    assert (r["prompt_tokens"], r["context_tokens"], r["cache_hit"]) == (46000, 46900, 0.9783)
    assert (r["text_chars"], r["thinking_chars"]) == (5, 50)
    assert r["tool_calls"] == [{"name": "ShellSession", "arg_bytes": 20}]
    assert (r["n_tool_calls"], r["max_tool_arg_bytes"], r["tool_errors"]) == (1, 20, 1)
    assert (r["retention"], r["demoted_new"], r["stubs"], r["stubbed_new"]) == (None, None, None, None)
    assert (r["guards"], r["compactions"], r["retries_before"]) == ([], [], 0)


def test_a_length_stop_keeps_the_salvaged_tool_call_size():
    big = {"command": "echo " + "QUJD" * 20_000}
    msg = assistant([{"type": "toolCall", "id": "c", "name": "ShellSession", "arguments": big}],
                    usage(input=200, cache_read=45_500, output=80_000), stop="length")
    _, recs = run([(0.0, {"type": "turn_start"}), (5.0, turn_end(msg))])
    r = recs[0]
    assert r["stop_reason"] == "length"
    assert r["max_tool_arg_bytes"] == len('{"command":"echo ') + 80_000 + len('"}')
    assert (r["headers_s"], r["ttft_s"], r["gen_s"]) == (None, None, None)


def test_retention_snapshots_attach_to_their_turn_and_demoted_new_is_the_growth():
    ok = assistant([], usage(input=10))
    events, t = [], 0.0
    for demoted in (4, 8, 8, 4, 8):
        events += [(t, {"type": "turn_start"}), (t + 0.1, retention(demoted)), (t + 1, turn_end(ok))]
        t += 2
    st, recs = run(events)
    assert [r["retention"]["demoted"] for r in recs] == [4, 8, 8, 4, 8]
    assert [r["demoted_new"] for r in recs] == [4, 4, 0, 0, 4]
    s = L.summarize(st)
    assert (s["max_demoted"], s["demoted_new_total"], s["max_large_pairs"]) == (8, 12, 8)
    assert s["max_bytes_saved"] == 8 * (4096 - 900)
    assert s["telemetry_seen"] is True


def test_telemetry_outside_a_turn_waits_for_the_next_record():
    ok = assistant([], usage(input=10), stop="aborted")
    abort = telemetry({"kind": "guard_abort", "trigger": "toolcall_cap", "tool": "ShellSession", "argChars": 32001, "capChars": 32000})
    echo = telemetry({"kind": "echo_block", "source": "shell_retention", "tool": "ShellSession", "ids": 1})
    st, recs = run([
        (0.0, {"type": "turn_start"}), (0.5, abort), (1.0, turn_end(ok)),
        (1.5, echo),
        (2.0, {"type": "turn_start"}), (3.0, turn_end(ok)),
    ])
    assert [g["kind"] for g in recs[0]["guards"]] == ["guard_abort"]
    assert recs[0]["guards"][0]["argChars"] == 32001
    assert [g["kind"] for g in recs[1]["guards"]] == ["echo_block"]
    assert L.summarize(st)["guard_fires"] == {"echo_block:shell_retention": 1, "guard_abort:toolcall_cap": 1}


def test_only_lc_telemetry_custom_entries_count():
    ok = assistant([], usage(input=10))
    other = {"type": "entry_appended", "entry": {"type": "custom", "customType": "someone-else",
             "data": {"kind": "guard_abort"}, "id": "x", "parentId": None, "timestamp": "t"}}
    label = {"type": "entry_appended", "entry": {"type": "label", "id": "y", "parentId": None, "timestamp": "t"}}
    bad = {"type": "entry_appended", "entry": {"type": "custom", "customType": "lc-telemetry",
           "data": "oops", "id": "z", "parentId": None, "timestamp": "t"}}
    st, recs = run([(0, {"type": "turn_start"}), (0.1, other), (0.2, label), (0.3, bad), (1, turn_end(ok))])
    assert (recs[0]["guards"], recs[0]["retention"]) == ([], None)
    assert L.summarize(st)["telemetry_seen"] is False


def test_compactions_attach_to_the_next_turn_and_are_counted_by_source():
    ok = assistant([], usage(input=10))
    st, recs = run([
        (0, {"type": "turn_start"}), (1, turn_end(ok)),
        (2, {"type": "compaction_start", "reason": "threshold"}),
        (3, {"type": "compaction_end", "reason": "threshold", "aborted": False, "willRetry": False,
             "result": {"summary": "s", "firstKeptEntryId": "e1", "tokensBefore": 230000, "estimatedTokensAfter": 41000}}),
        (4, {"type": "lc_harness_compaction", "n": 1, "ok": True, "tokens_before": 221000, "tokens_after": 38000, "error": None}),
        (5, {"type": "lc_harness_compaction", "n": 2, "ok": False, "tokens_before": None, "tokens_after": None,
             "error": "RuntimeError: pi rejected compact: Already compacted"}),
        (6, {"type": "compaction_end", "reason": "manual", "result": None, "aborted": True, "willRetry": False}),
        (7, {"type": "turn_start"}), (8, turn_end(ok)),
    ])
    assert recs[0]["compactions"] == []
    assert recs[1]["compactions"] == [
        {"source": "pi", "reason": "threshold", "ok": True, "tokens_before": 230000, "tokens_after": 41000, "error": None},
        {"source": "harness", "reason": "manual", "ok": True, "tokens_before": 221000, "tokens_after": 38000, "error": None},
        {"source": "harness", "reason": "manual", "ok": False, "tokens_before": None, "tokens_after": None,
         "error": "RuntimeError: pi rejected compact: Already compacted"},
        {"source": "pi", "reason": "manual", "ok": False, "tokens_before": None, "tokens_after": None, "error": None},
    ]
    assert L.summarize(st)["compactions"] == {"pi": 1, "harness": 1, "harness_failed": 1}


def test_an_aborted_turn_without_usage_is_marked_unreported():
    msg = assistant([{"type": "text", "text": "partial"}], stop="aborted")
    _, recs = run([(0, {"type": "turn_start"}), (2, turn_end(msg))])
    r = recs[0]
    assert (r["usage_reported"], r["prompt_tokens"], r["cache_hit"], r["reasoning"]) == (False, 0, None, None)
    assert (r["stop_reason"], r["text_chars"], r["turn_s"]) == ("aborted", 7, 2.0)


def test_turn_numbers_run_across_agent_runs_and_retries_are_counted():
    ok = assistant([], usage(input=10))
    err = assistant([], stop="error", error="upstream 500")
    st, recs = run([
        (0, {"type": "agent_start"}), (0, {"type": "turn_start"}), (1, turn_end(err)),
        (1, {"type": "agent_end", "messages": [], "willRetry": True}),
        (2, {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 2000, "errorMessage": "upstream 500"}),
        (4, {"type": "agent_start"}), (4, {"type": "turn_start"}), (5, turn_end(ok)),
    ])
    assert [r["turn"] for r in recs] == [1, 2]
    assert recs[0]["error_message"] == "upstream 500"
    assert recs[1]["retries_before"] == 1
    assert L.summarize(st)["stop_reasons"] == {"error": 1, "toolUse": 1}


def test_summary_folds_peaks_histogram_ttft_and_stubs():
    a = assistant([{"type": "toolCall", "id": "c", "name": "write", "arguments": {"content": "x" * 100}}],
                  usage(input=1000, cache_read=40000, output=500))
    b = assistant([], usage(input=2000, cache_read=60000, output=17000), stop="length")
    stub = telemetry({"kind": "length_stub", "stubbed": 1, "bytesBefore": 90000, "bytesAfter": 1200})
    st, recs = run([
        (0, {"type": "turn_start"}),
        (0.5, {"type": "message_update", "assistantMessageEvent": {"type": "text_start", "contentIndex": 0}, "message": a}),
        (1, turn_end(a)),
        (2, {"type": "turn_start"}), (2.1, stub), (9, turn_end(b)),
    ])
    s = L.summarize(st)
    assert s["n_turns"] == 2
    assert (s["peak_prompt_tokens"], s["peak_context_tokens"], s["max_output_tokens"]) == (62000, 79000, 17000)
    assert s["max_tool_arg_bytes"] == len('{"content":"') + 100 + len('"}')
    assert s["stop_reasons"] == {"length": 1, "toolUse": 1}
    assert (s["stubbed_new_total"], recs[1]["stubbed_new"], recs[1]["stubs"]["stubbed"]) == (1, 1, 1)
    assert s["max_ttft_s"] == 0.5


def test_a_stub_after_compaction_counts_as_new_once_a_zero_snapshot_arrives():
    ok = assistant([], usage(input=10))

    def stub(n):
        return telemetry({"kind": "length_stub", "stubbed": n, "bytesBefore": n * 9000, "bytesAfter": n * 1200})

    st, recs = run([
        (0, {"type": "turn_start"}), (0.1, stub(3)), (1, turn_end(ok)),
        (2, {"type": "lc_harness_compaction", "n": 1, "ok": True, "tokens_before": 221000, "tokens_after": 38000, "error": None}),
        (3, {"type": "turn_start"}), (3.1, stub(0)), (4, turn_end(ok)),
        (5, {"type": "turn_start"}), (5.1, stub(1)), (6, turn_end(ok)),
    ])
    assert [r["stubbed_new"] for r in recs] == [3, 0, 1]
    assert L.summarize(st)["stubbed_new_total"] == 4


def test_a_turn_start_without_turn_end_is_counted_and_its_telemetry_kept():
    ok = assistant([], usage(input=10))
    st, recs = run([
        (0, {"type": "turn_start"}), (0.1, retention(4)),
        (1, {"type": "turn_start"}), (2, turn_end(ok)),
    ])
    assert len(recs) == 1 and recs[0]["retention"]["demoted"] == 4
    assert L.summarize(st)["orphan_turns"] == 1


@pytest.mark.parametrize("ev", [
    None, "turn_end", {}, {"type": "message_start"}, {"type": "entry_appended", "entry": None},
    {"type": "compaction_end", "result": "x"},
    {"type": "turn_end", "message": "garbage", "toolResults": "nope"},
    {"type": "turn_end", "message": {"usage": {"input": "12", "output": None},
                                     "content": [None, {"type": "toolCall", "arguments": {1, 2}}]}},
])
def test_malformed_events_never_raise(ev):
    st = L.LedgerState()
    L.observe(st, ev, 1.0)
    L.summarize(st)
