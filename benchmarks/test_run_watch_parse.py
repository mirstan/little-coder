"""run_watch parsers over real omlx / Splash log lines (brief, omlx.log and
scratchpad splash/serve-*-tb.log) and turns.jsonl records."""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_watch as W  # noqa: E402

OMLX_PREFIX = (
    "2026-10-07 18:04:07,398 - omlx.scheduler - INFO - prefix cache: request "
    "964228db-3670-4e00-b06b-efba8572ae77 re-prefills 5220 of 71530 tokens (reused 66310); "
    "closest stored sequence 0fbf9c00-2858-4e1e-9f2f-3847d517846d shares the first 68144 of "
    "69632 comparable tokens before diverging"
)
OMLX_RESTORE = (
    "2026-10-07 18:16:45,928 - omlx.scheduler - INFO - Prefix cache restore for "
    "a85ff18d-fc4e-411a-8098-c0f92eb5ace2: source=paged cached=95193 suffix=478 blocks=24 "
    "lookup=4.737ms reconstruct=164.598ms promote=True"
)
OMLX_DONE_OLD = (
    "2026-09-05 17:46:32,973 - omlx.server - INFO - Chat completion: model=tiel-coder-oq4e, "
    "394 tokens in 16.56s (56.2 tok/s), prompt: 8093, finish_reason=tool_calls, "
    "max_tokens=32768, request_max_tokens=32768"
)
OMLX_DONE_NEW = (
    "2026-10-02 09:37:52,868 - omlx.server - INFO - Chat completion: model=tiel-coder-oq6e-fp16, "
    "108 tokens in 6.84s (56.4 tok/s), prompt: 3918, finish_reason=tool_calls, max_tokens=80000, "
    "request_max_tokens=80000, stream_model_ttft=4.92s, stream_visible_ttft=4.93s"
)
OMLX_WARN = (
    "2026-09-08 20:31:45,910 - omlx.scheduler - WARNING - [guard:chunked_step] context too large "
    "at progress=4096 kv_len=69632"
)
# Shaped like the WARNING lines above, at ERROR level.
OMLX_ERROR = "2026-09-08 20:31:46,002 - omlx.server - ERROR - prefill_memory_exceeded: request rejected"
SPLASH_DONE = "10:27:09 Done · input 3,623 · cached 3,360 · output 120 · tools 4·a72fb0c4 · TTFT 0.8s · 70.5 tok/s"
SPLASH_DONE_BARE = "01:42:14 Done · input 83 · cached 32 · output 250 · TTFT 1.0s · 21.9 tok/s"
SPLASH_CANCELLED = "10:57:37 Cancelled · input 3,473 · cached 3,040 · output 48,227 · tools 4·a72fb0c4 · TTFT 1.1s · 76.4 tok/s"
SPLASH_ERROR = "20:45:10 Error · not_found · GET /api/status"
SPLASH_READY = "10:26:47 Ready · incoai/Qwen3.6-35B-A3B-Splash · context 256K · http://127.0.0.1:8000"


def tod(h, m, s):
    return h * 3600 + m * 60 + s


def test_omlx_prefix_cache_line():
    assert W.parse_omlx_line(OMLX_PREFIX) == {
        "src": "omlx", "kind": "prefix_cache",
        "ts": datetime(2026, 10, 7, 18, 4, 7).timestamp() + 0.398,
        "request": "964228db-3670-4e00-b06b-efba8572ae77",
        "reprefill": 5220, "prompt": 71530, "reused": 66310, "shared": 68144, "comparable": 69632,
    }


def test_omlx_completion_lines_old_and_new_format():
    old = W.parse_omlx_line(OMLX_DONE_OLD)
    assert old == {
        "src": "omlx", "kind": "completion", "ts": datetime(2026, 9, 5, 17, 46, 32).timestamp() + 0.973,
        "model": "tiel-coder-oq4e", "prompt": 8093, "cached": None, "output": 394, "duration_s": 16.56,
        "finish_reason": "tool_calls", "status": None, "ttft_s": None,
    }
    new = W.parse_omlx_line(OMLX_DONE_NEW)
    assert (new["kind"], new["prompt"], new["output"], new["ttft_s"]) == ("completion", 3918, 108, 4.92)


def test_omlx_restore_and_warning_lines_are_not_events_but_errors_are():
    assert W.parse_omlx_line(OMLX_RESTORE) is None
    assert W.parse_omlx_line(OMLX_WARN) is None
    err = W.parse_omlx_line(OMLX_ERROR)
    assert (err["kind"], err["text"]) == ("server_error", "prefill_memory_exceeded: request rejected")
    assert W.parse_omlx_line(SPLASH_DONE) is None
    assert W.parse_omlx_line("Traceback (most recent call last):") is None


def test_splash_done_and_cancelled_lines():
    assert W.parse_splash_line(SPLASH_DONE) == {
        "src": "splash", "kind": "completion", "tod": tod(10, 27, 9), "status": "done",
        "prompt": 3623, "cached": 3360, "output": 120, "ttft_s": 0.8, "tok_s": 70.5,
    }
    bare = W.parse_splash_line(SPLASH_DONE_BARE)
    assert (bare["tod"], bare["prompt"], bare["cached"], bare["output"], bare["ttft_s"]) == (tod(1, 42, 14), 83, 32, 250, 1.0)
    cancelled = W.parse_splash_line(SPLASH_CANCELLED)
    assert (cancelled["status"], cancelled["output"]) == ("cancelled", 48227)


def test_splash_error_lifecycle_and_noise():
    assert W.parse_splash_line(SPLASH_ERROR) == {
        "src": "splash", "kind": "server_error", "tod": tod(20, 45, 10), "text": "not_found · GET /api/status"}
    assert W.parse_splash_line(SPLASH_READY)["kind"] == "lifecycle"
    assert W.parse_splash_line("Preparing weights: target/layer-0.bin") is None
    assert W.parse_splash_line("02:41:35 Chat template · patched to render later system messages in place") is None
    assert W.parse_splash_line(OMLX_PREFIX) is None


def test_a_backlog_spanning_midnight_is_dated_from_the_file_mtime():
    tods = [tod(23, 58, 10), tod(23, 59, 50), tod(0, 0, 30), tod(0, 1, 0)]
    mtime = datetime(2026, 10, 8, 0, 1, 0, 500000).timestamp()
    assert W.anchor_splash_batch(tods, mtime) == [
        datetime(2026, 10, 7, 23, 58, 10).timestamp(),
        datetime(2026, 10, 7, 23, 59, 50).timestamp(),
        datetime(2026, 10, 8, 0, 0, 30).timestamp(),
        datetime(2026, 10, 8, 0, 1, 0).timestamp(),
    ]


def test_a_line_stamped_before_midnight_but_flushed_after_keeps_its_day():
    mtime = datetime(2026, 10, 8, 0, 0, 0, 500000).timestamp()
    assert W.anchor_splash_batch([tod(23, 59, 59)], mtime) == [datetime(2026, 10, 7, 23, 59, 59).timestamp()]


def test_a_same_day_batch_and_an_empty_batch():
    mtime = datetime(2026, 10, 7, 10, 30, 0).timestamp()
    assert W.anchor_splash_batch([tod(10, 27, 9), tod(10, 29, 0)], mtime) == [
        datetime(2026, 10, 7, 10, 27, 9).timestamp(), datetime(2026, 10, 7, 10, 29, 0).timestamp()]
    assert W.anchor_splash_batch([], mtime) == []


def test_detect_server_kind():
    assert W.detect_server_kind(["noise", OMLX_DONE_OLD]) == "omlx"
    assert W.detect_server_kind(["Preparing weights: x", SPLASH_READY]) == "splash"
    assert W.detect_server_kind(["noise", "more noise"]) is None


def test_parse_turn_record():
    assert W.parse_turn_record(json.dumps({"v": 1, "turn": 3, "stop_reason": "length"})) == {
        "v": 1, "turn": 3, "stop_reason": "length"}
    assert W.parse_turn_record('{"v": 2, "turn": 3}') is None
    assert W.parse_turn_record('{"v": 1, "turn": "3"}') is None
    assert W.parse_turn_record('{"v": 1, "tu') is None
    assert W.parse_turn_record("[1, 2]") is None
