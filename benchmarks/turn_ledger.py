"""Per-turn JSONL ledger for one Harbor trial, built from pi's RPC events.

One record per model turn (pi turn_start ... turn_end) goes to
<trial>/agent/turns.jsonl; benchmarks/run_watch.py tails it while the job runs,
and the Harbor adapter copies summarize()'s output into result.json metadata
under "turn_ledger".

Sources, all authoritative, none parsed from prose:
- turn_end.message is pi-ai's AssistantMessage (pi-ai dist/types.d.ts:260-302):
  usage {input, output, cacheRead, cacheWrite, reasoning?, totalTokens},
  stopReason, errorMessage, content[] of text / thinking / toolCall.
- lc-telemetry custom entries the extensions append (entry_appended; see
  .pi/extensions/_shared/telemetry.ts). Their counts are per-request
  snapshots, because pi hands the context hook stored history on every
  request; demoted_new / stubbed_new are diffs of consecutive snapshots.
- compaction_end from pi, and lc_harness_compaction from rpc_client.py (one
  per deliberate harness compaction).
- Timing is this process's clock at event receipt: pi's RPC turn_start and
  turn_end carry no timestamp or turn index (pi-coding-agent
  agent-session.js:353 forwards the raw agent event; turnIndex lives only on
  the extension event, :435-452, and resets every agent_start).

observe() is pure: state and one event in, finished records out. The LedgerSink
half at the bottom is the only IO, and it never raises into a trial.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

LEDGER_FILENAME = "turns.jsonl"
SCHEMA_VERSION = 1
TELEMETRY_CUSTOM_TYPE = "lc-telemetry"
#: rpc_client.HARNESS_COMPACTION_EVENT, spelled out so this module imports
#: nothing from the repo.
HARNESS_COMPACTION_EVENT = "lc_harness_compaction"


@dataclass
class LedgerState:
    turn: int = 0
    open_turn: Optional[dict] = None
    pending: dict = field(default_factory=lambda: {
        "retention": None, "stubs": None, "guards": [], "compactions": [], "retries": 0})
    prev_demoted: int = 0
    prev_stubbed: int = 0
    orphan_turns: int = 0
    peak_prompt_tokens: int = 0
    peak_context_tokens: int = 0
    max_output_tokens: int = 0
    max_tool_arg_bytes: int = 0
    max_ttft_s: Optional[float] = None
    stop_reasons: dict = field(default_factory=dict)
    telemetry_seen: bool = False
    max_large_pairs: int = 0
    max_demoted: int = 0
    demoted_new_total: int = 0
    max_bytes_saved: int = 0
    max_signed: int = 0
    stubbed_new_total: int = 0
    guard_fires: dict = field(default_factory=dict)
    compactions_pi: int = 0
    compactions_harness: int = 0
    compactions_harness_failed: int = 0


def new_ledger() -> LedgerState:
    return LedgerState()


def _bucket() -> dict:
    return {"retention": None, "stubs": None, "guards": [], "compactions": [], "retries": 0}


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _count(v: Any) -> int:
    x = _num(v)
    return int(x) if x is not None and x > 0 else 0


def _opt_int(v: Any) -> Optional[int]:
    x = _num(v)
    return int(x) if x is not None else None


def _since(start: float, at: Optional[float]) -> Optional[float]:
    return None if at is None else round(at - start, 3)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_bytes(v: Any) -> int:
    try:
        return len(json.dumps(v, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _bucket_now(state: LedgerState) -> dict:
    return state.open_turn["bucket"] if state.open_turn is not None else state.pending


def _merge(into: dict, src: dict) -> None:
    for key in ("retention", "stubs"):
        if src[key] is not None:
            into[key] = src[key]
    into["guards"].extend(src["guards"])
    into["compactions"].extend(src["compactions"])
    into["retries"] += src["retries"]


def _observe_telemetry(state: LedgerState, entry: Any) -> None:
    if (not isinstance(entry, dict) or entry.get("type") != "custom"
            or entry.get("customType") != TELEMETRY_CUSTOM_TYPE):
        return
    data = entry.get("data")
    if not isinstance(data, dict):
        return
    state.telemetry_seen = True
    bucket = _bucket_now(state)
    kind = data.get("kind")
    if kind == "shell_retention":
        bucket["retention"] = data
    elif kind == "length_stub":
        bucket["stubs"] = data
    else:
        bucket["guards"].append(data)
        label = f"{kind}:{data.get('trigger') or data.get('source') or '?'}"
        state.guard_fires[label] = state.guard_fires.get(label, 0) + 1


def observe(state: LedgerState, ev: Any, now: float) -> list[dict]:
    """Fold one RPC event in; return the record a turn_end completes, else []."""
    if not isinstance(ev, dict):
        return []
    kind = ev.get("type")
    if kind == "turn_start":
        if state.open_turn is not None:
            # turn_start with no turn_end before it: the lost turn's telemetry
            # rides on the record that follows.
            state.orphan_turns += 1
            _merge(state.pending, state.open_turn["bucket"])
        state.open_turn = {"ts_start": now, "msg_start": None, "first_update": None,
                           "msg_end": None, "bucket": state.pending}
        state.pending = _bucket()
        return []
    if kind in ("message_start", "message_end"):
        msg = ev.get("message")
        # user and toolResult messages emit these too (agent-loop.js:52-53, :546-547)
        if state.open_turn is not None and isinstance(msg, dict) and msg.get("role") == "assistant":
            slot = "msg_start" if kind == "message_start" else "msg_end"
            if state.open_turn[slot] is None:
                state.open_turn[slot] = now
        return []
    if kind == "message_update":
        if state.open_turn is not None and state.open_turn["first_update"] is None:
            state.open_turn["first_update"] = now
        return []
    if kind == "entry_appended":
        _observe_telemetry(state, ev.get("entry"))
        return []
    if kind == "compaction_end":
        result = ev.get("result")
        ok = isinstance(result, dict) and not ev.get("aborted")
        info = result if isinstance(result, dict) else {}
        reason = str(ev.get("reason") or "")
        _bucket_now(state)["compactions"].append({
            "source": "pi", "reason": reason, "ok": ok,
            "tokens_before": _opt_int(info.get("tokensBefore")),
            "tokens_after": _opt_int(info.get("estimatedTokensAfter")),
            "error": ev.get("errorMessage") or None,
        })
        # "manual" is the harness's own compact request (rpc-mode.js:415-417),
        # counted from lc_harness_compaction; listed here, not counted twice.
        if ok and reason != "manual":
            state.compactions_pi += 1
        return []
    if kind == HARNESS_COMPACTION_EVENT:
        ok = ev.get("ok") is True
        _bucket_now(state)["compactions"].append({
            "source": "harness", "reason": "manual", "ok": ok,
            "tokens_before": _opt_int(ev.get("tokens_before")),
            "tokens_after": _opt_int(ev.get("tokens_after")),
            "error": ev.get("error") or None,
        })
        if ok:
            state.compactions_harness += 1
        else:
            state.compactions_harness_failed += 1
        return []
    if kind == "auto_retry_start":
        _bucket_now(state)["retries"] += 1
        return []
    if kind == "turn_end":
        return [_close_turn(state, ev, now)]
    return []


def _close_turn(state: LedgerState, ev: dict, now: float) -> dict:
    opened = state.open_turn
    if opened is None:
        opened = {"ts_start": now, "msg_start": None, "first_update": None,
                  "msg_end": None, "bucket": state.pending}
        state.pending = _bucket()
    state.open_turn = None
    bucket = opened["bucket"]
    msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
    usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
    inp, out = _count(usage.get("input")), _count(usage.get("output"))
    cache_read, cache_write = _count(usage.get("cacheRead")), _count(usage.get("cacheWrite"))
    prompt = inp + cache_read + cache_write
    content = msg.get("content") if isinstance(msg.get("content"), list) else []
    blocks = [b for b in content if isinstance(b, dict)]
    text_chars = sum(len(b["text"]) for b in blocks if b.get("type") == "text" and isinstance(b.get("text"), str))
    thinking_chars = sum(len(b["thinking"]) for b in blocks
                         if b.get("type") == "thinking" and isinstance(b.get("thinking"), str))
    tool_calls = [{"name": str(b.get("name") or ""), "arg_bytes": _json_bytes(b.get("arguments"))}
                  for b in blocks if b.get("type") == "toolCall"]
    results = ev.get("toolResults") if isinstance(ev.get("toolResults"), list) else []

    retention, stubs = bucket["retention"], bucket["stubs"]
    demoted_new = stubbed_new = None
    if retention is not None:
        current = _count(retention.get("demoted"))
        demoted_new = max(0, current - state.prev_demoted)
        state.prev_demoted = current
    if stubs is not None:
        current = _count(stubs.get("stubbed"))
        stubbed_new = max(0, current - state.prev_stubbed)
        state.prev_stubbed = current

    state.turn += 1
    start = opened["ts_start"]
    record = {
        "v": SCHEMA_VERSION,
        "turn": state.turn,
        "ts_start": round(start, 3),
        "ts_end": round(now, 3),
        "iso_end": _iso(now),
        "headers_s": _since(start, opened["msg_start"]),
        "ttft_s": _since(start, opened["first_update"]),
        "gen_s": _since(start, opened["msg_end"]),
        "turn_s": round(now - start, 3),
        "stop_reason": str(msg.get("stopReason") or ""),
        "error_message": msg.get("errorMessage") or None,
        "usage_reported": (inp + out + cache_read + cache_write) > 0,
        "input": inp,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "output": out,
        "reasoning": _opt_int(usage.get("reasoning")),
        "prompt_tokens": prompt,
        "context_tokens": prompt + out,
        "cache_hit": round(cache_read / prompt, 4) if prompt > 0 else None,
        "text_chars": text_chars,
        "thinking_chars": thinking_chars,
        "n_tool_calls": len(tool_calls),
        "max_tool_arg_bytes": max((c["arg_bytes"] for c in tool_calls), default=0),
        "tool_calls": tool_calls,
        "tool_errors": sum(1 for r in results if isinstance(r, dict) and r.get("isError") is True),
        "retention": retention,
        "demoted_new": demoted_new,
        "stubs": stubs,
        "stubbed_new": stubbed_new,
        "guards": bucket["guards"],
        "compactions": bucket["compactions"],
        "retries_before": bucket["retries"],
    }
    _fold(state, record)
    return record


def _fold(state: LedgerState, r: dict) -> None:
    state.peak_prompt_tokens = max(state.peak_prompt_tokens, r["prompt_tokens"])
    state.peak_context_tokens = max(state.peak_context_tokens, r["context_tokens"])
    state.max_output_tokens = max(state.max_output_tokens, r["output"])
    state.max_tool_arg_bytes = max(state.max_tool_arg_bytes, r["max_tool_arg_bytes"])
    if r["ttft_s"] is not None:
        state.max_ttft_s = r["ttft_s"] if state.max_ttft_s is None else max(state.max_ttft_s, r["ttft_s"])
    reason = r["stop_reason"] or "unknown"
    state.stop_reasons[reason] = state.stop_reasons.get(reason, 0) + 1
    ret = r["retention"]
    if ret is not None:
        state.max_large_pairs = max(state.max_large_pairs, _count(ret.get("large")))
        state.max_demoted = max(state.max_demoted, _count(ret.get("demoted")))
        state.max_signed = max(state.max_signed, _count(ret.get("signed")))
        state.max_bytes_saved = max(state.max_bytes_saved,
                                    _count(ret.get("bytesBefore")) - _count(ret.get("bytesAfter")))
        state.demoted_new_total += r["demoted_new"] or 0
    if r["stubbed_new"]:
        state.stubbed_new_total += r["stubbed_new"]


def summarize(state: LedgerState) -> dict:
    """The result.json metadata "turn_ledger" block (the sink adds ledger_errors)."""
    return {
        "v": SCHEMA_VERSION,
        "n_turns": state.turn,
        "orphan_turns": state.orphan_turns,
        "peak_prompt_tokens": state.peak_prompt_tokens,
        "peak_context_tokens": state.peak_context_tokens,
        "max_output_tokens": state.max_output_tokens,
        "max_tool_arg_bytes": state.max_tool_arg_bytes,
        "max_ttft_s": state.max_ttft_s,
        "stop_reasons": dict(sorted(state.stop_reasons.items())),
        "telemetry_seen": state.telemetry_seen,
        "max_large_pairs": state.max_large_pairs,
        "max_demoted": state.max_demoted,
        "demoted_new_total": state.demoted_new_total,
        "max_bytes_saved": state.max_bytes_saved,
        "max_signed": state.max_signed,
        "stubbed_new_total": state.stubbed_new_total,
        "guard_fires": dict(sorted(state.guard_fires.items())),
        "compactions": {"pi": state.compactions_pi, "harness": state.compactions_harness,
                        "harness_failed": state.compactions_harness_failed},
    }


# ── IO: the sink the Harbor adapter drives ─────────────────────────────────


@dataclass
class LedgerSink:
    path: Optional[Path]
    log: Callable[[str], None]
    clock: Callable[[], float]
    state: LedgerState = field(default_factory=LedgerState)
    fh: Any = None
    errors: int = 0
    logged: bool = False


def make_ledger_sink(
    path: Optional[Path],
    log: Callable[[str], None],
    clock: Callable[[], float] = time.time,
) -> LedgerSink:
    return LedgerSink(path=path, log=log, clock=clock)


def _sink_failed(sink: LedgerSink, where: str, exc: BaseException) -> None:
    """Count every failure; log only the first, so a broken disk cannot flood the trial log."""
    sink.errors += 1
    if sink.logged:
        return
    sink.logged = True
    try:
        sink.log(f"turn ledger error in {where} (logged once; later errors are only counted): "
                 f"{type(exc).__name__}: {exc}")
    except Exception:
        pass


def sink_on_event(sink: LedgerSink, ev: Any) -> None:
    """Feed one RPC event; append any finished record. Never raises."""
    try:
        records = observe(sink.state, ev, sink.clock())
    except Exception as exc:
        _sink_failed(sink, "event handling", exc)
        return
    if not records or sink.path is None:
        return
    try:
        if sink.fh is None:
            sink.fh = open(sink.path, "a", encoding="utf-8")
        for record in records:
            sink.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.fh.flush()
    except Exception as exc:
        _sink_failed(sink, "file write", exc)


def sink_metadata(sink: LedgerSink) -> dict:
    """summarize() plus the error count; just the count if summarizing fails."""
    try:
        meta = summarize(sink.state)
        meta["ledger_errors"] = sink.errors
        return meta
    except Exception as exc:
        _sink_failed(sink, "summary", exc)
        return {"ledger_errors": sink.errors}


def sink_close(sink: LedgerSink) -> None:
    try:
        if sink.fh is not None:
            sink.fh.close()
    except Exception as exc:
        _sink_failed(sink, "close", exc)
    finally:
        sink.fh = None
