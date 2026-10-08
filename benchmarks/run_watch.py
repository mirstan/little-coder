#!/usr/bin/env python3
"""Watch a running Harbor job and say when a trial goes wrong, while it runs.

Tails every trial's agent/turns.jsonl (written by benchmarks/turn_ledger.py)
and, with --server-log, the model server's log (omlx's
/opt/homebrew/var/log/omlx.log, or a Splash serve log). Prints one line per
alert, a status line every --status-interval seconds, and appends every alert
to alerts.jsonl in the job dir.

    python3 run_watch.py [JOB_OR_JOBS_DIR] [--server-log PATH]
        [--server-kind auto|omlx|splash] [--once] [--interval S]
        [--status-interval S] [--server-backlog-bytes N] [rule flags]

Without JOB_OR_JOBS_DIR it watches $RUN_WATCH_JOBS_DIR, else
./benchmarks/harbor_runs under the current directory (so a copy of this file
run from the repo root works), and exits 2 if that does not exist yet. Given a
jobs dir, the newest job (by its config.json mtime) is watched, re-checked
every poll. Server-log lines stamped before that job's config.json mtime feed
the status line only and never alert, so a long omlx.log backlog cannot page
about earlier jobs. Every RuleConfig field is a flag and an env var, flag
first: --context-jump-tokens / RUN_WATCH_CONTEXT_JUMP_TOKENS, and so on. A
trial whose agent/little_coder.log is non-empty (the adapter writes it once
the agent has finished) is "verifying": no stall alerts.

An omlx prefix-cache divergence deep in history is joined to the turn whose
request produced it (same trial, timestamp inside the turn, same prompt size
when the turn reported usage; see classify_divergence) and reported as expected_divergence when that turn
demoted, stubbed or followed a compaction, as divergence_in_output when it
split the previous turn's generated output (jundot/omlx#4353), and as a
prefix_divergence warning only when the ledger explains nothing.

--once makes one pass over everything already written and exits 1 when it
raised a crit alert, 2 when the pass itself failed or its alerts could not be
written to alerts.jsonl, else 0. Alerts already in alerts.jsonl are not raised
again, so cron can call it; an alert whose write failed is not recorded as
raised, so the next pass raises it again.

Run a copy. CPython compiles this whole file before running it, and it
imports only the standard library, all at the top (test_run_watch_io.py
checks both, and runs a lone copy), so editing it or switching branches under
a running watcher changes nothing the watcher does. Launch from a copy anyway
so tracebacks show the code that is actually running:

    d=$(mktemp -d) && cp benchmarks/run_watch.py "$d/" && \
      python3 "$d/run_watch.py" benchmarks/harbor_runs \
        --server-log /opt/homebrew/var/log/omlx.log &

Parsing and rules read no files and take the clock as an argument; all
file IO lives below the IO section marker.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

#: A Splash stamp that jumps BACK by more than this between consecutive
#: lines crossed a midnight; real lines are never this far out of order.
CLOCK_SLACK_S = 60.0

_OMLX_LINE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) - (\S+) - ([A-Z]+) - (.*)$")
_OMLX_PREFIX_RE = re.compile(
    r"prefix cache: request (\S+) re-prefills (\d+) of (\d+) tokens \(reused (\d+)\); "
    r"closest stored sequence \S+ shares the first (\d+) of (\d+) comparable tokens"
)
_OMLX_TOKENS_RE = re.compile(r"^(\d+) tokens in ([\d.]+)s")
_SPLASH_LINE_RE = re.compile(r"^(\d\d):(\d\d):(\d\d) (.+)$")


def _int(text: Any) -> Optional[int]:
    try:
        return int(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _secs(text: Any) -> Optional[float]:
    if text is None:
        return None
    t = str(text).strip()
    if t.endswith("s"):
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        return None


def parse_omlx_line(line: str) -> Optional[dict]:
    """One omlx.log line -> event dict, or None for lines no rule reads."""
    m = _OMLX_LINE_RE.match(line.rstrip("\r\n"))
    if m is None:
        return None
    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000.0
    level, msg = m.group(4), m.group(5)
    p = _OMLX_PREFIX_RE.search(msg)
    if p is not None:
        return {"src": "omlx", "kind": "prefix_cache", "ts": ts, "request": p.group(1),
                "reprefill": int(p.group(2)), "prompt": int(p.group(3)), "reused": int(p.group(4)),
                "shared": int(p.group(5)), "comparable": int(p.group(6))}
    marker = "Chat completion: "
    if marker in msg:
        kv: dict[str, str] = {}
        output = duration = None
        for part in msg.split(marker, 1)[1].split(", "):
            tokens = _OMLX_TOKENS_RE.match(part)
            if tokens is not None:
                output, duration = int(tokens.group(1)), float(tokens.group(2))
                continue
            key, sep, value = part.partition("=")
            if not sep:
                key, sep, value = part.partition(": ")
            if sep:
                kv[key.strip()] = value.strip()
        return {"src": "omlx", "kind": "completion", "ts": ts, "model": kv.get("model"),
                "prompt": _int(kv.get("prompt")), "cached": None, "output": output,
                "duration_s": duration, "finish_reason": kv.get("finish_reason"), "status": None,
                "ttft_s": _secs(kv.get("stream_model_ttft"))}
    if level in ("ERROR", "CRITICAL"):
        return {"src": "omlx", "kind": "server_error", "ts": ts, "text": msg[:300]}
    return None


def parse_splash_line(line: str) -> Optional[dict]:
    """One Splash serve-log line -> event dict with `tod` (seconds since local midnight)."""
    m = _SPLASH_LINE_RE.match(line.rstrip("\r\n"))
    if m is None:
        return None
    tod = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    parts = m.group(4).split(" · ")
    head = parts[0]
    if head in ("Done", "Cancelled"):
        ev: dict = {"src": "splash", "kind": "completion", "tod": tod, "status": head.lower(),
                    "prompt": None, "cached": None, "output": None, "ttft_s": None, "tok_s": None}
        for part in parts[1:]:
            if part.endswith(" tok/s"):
                ev["tok_s"] = _secs(part[: -len(" tok/s")])
                continue
            key, _, value = part.partition(" ")
            if key == "input":
                ev["prompt"] = _int(value)
            elif key == "cached":
                ev["cached"] = _int(value)
            elif key == "output":
                ev["output"] = _int(value)
            elif key == "TTFT":
                ev["ttft_s"] = _secs(value)
        return ev
    if head == "Error":
        return {"src": "splash", "kind": "server_error", "tod": tod, "text": " · ".join(parts[1:])[:300]}
    if head in ("Loading", "Ready", "Stopping"):
        return {"src": "splash", "kind": "lifecycle", "tod": tod, "text": m.group(4)[:300]}
    return None


def anchor_splash_batch(tods: list[int], mtime: float) -> list[float]:
    """Epoch seconds for consecutive Splash stamps (oldest first), dated from the file's mtime.

    The last line was written no later than mtime, so it takes mtime's date,
    or the day before when its time of day is later than mtime's (written just
    before midnight, flushed just after). Walking backwards, an earlier stamp
    later than its successor by more than CLOCK_SLACK_S crossed a midnight.
    """
    if not tods:
        return []
    anchor = datetime.fromtimestamp(mtime)
    anchor_tod = anchor.hour * 3600 + anchor.minute * 60 + anchor.second + anchor.microsecond / 1e6
    day = anchor.date()
    if tods[-1] > anchor_tod + CLOCK_SLACK_S:
        day -= timedelta(days=1)
    days = [day] * len(tods)
    for i in range(len(tods) - 2, -1, -1):
        if tods[i] > tods[i + 1] + CLOCK_SLACK_S:
            day -= timedelta(days=1)
        days[i] = day
    return [datetime.combine(d, dtime()).timestamp() + t for d, t in zip(days, tods)]


def detect_server_kind(lines: list[str]) -> Optional[str]:
    for line in lines:
        if _OMLX_LINE_RE.match(line):
            return "omlx"
        if parse_splash_line(line) is not None:
            return "splash"
    return None


def parse_turn_record(line: str) -> Optional[dict]:
    """One turns.jsonl line (benchmarks/turn_ledger.py, schema v1), or None."""
    try:
        rec = json.loads(line)
    except ValueError:
        return None
    if not isinstance(rec, dict) or rec.get("v") != 1:
        return None
    turn = rec.get("turn")
    if not isinstance(turn, int) or isinstance(turn, bool):
        return None
    return rec


# ── Rules ─────────────────────────────────────────────────────────────────

ENV_PREFIX = "RUN_WATCH_"


@dataclass
class RuleConfig:
    context_jump_tokens: int = 30_000
    max_output_tokens: int = 16_000
    no_demotion_large_pairs: int = 12
    no_demotion_turns: int = 40
    no_demotion_context_tokens: int = 60_000
    #: Due pairs a trial must have reached before the long-context branch of
    #: no_demotions applies: fewer than a batch (LITTLE_CODER_SHELL_DEMOTE_BATCH,
    #: default 4) never demote by design.
    no_demotion_min_due: int = 4
    divergence_reprefill_tokens: int = 16_384
    divergence_gap_tokens: int = 8_192
    #: How long a divergence waits for the turn record that produced it.
    divergence_match_wait_s: float = 1800.0
    #: |server prompt - ledger prompt_tokens| still counted as the same request
    #: (only for turns that reported usage; see classify_divergence).
    divergence_match_tokens: int = 16
    cache_min_prompt_tokens: int = 40_000
    cache_min_hit_ratio: float = 0.5
    ttft_max_s: float = 120.0
    stall_min: float = 20.0
    cooldown_s: float = 600.0
    telemetry_grace_turns: int = 10


@dataclass
class Alert:
    ts: float
    level: str
    rule: str
    trial: Optional[str]
    subject: str
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class TrialState:
    name: str
    start_ts: float
    end_ts: Optional[float] = None
    finished: bool = False
    reward: Optional[float] = None
    verifying: bool = False
    n_turns: int = 0
    last_turn_ts: Optional[float] = None
    last_prompt_tokens: Optional[int] = None
    peak_prompt_tokens: int = 0
    compactions: int = 0
    telemetry_seen: bool = False
    max_large_pairs: int = 0
    max_demoted: int = 0
    max_due: int = 0
    #: True once a shell_retention snapshot arrived (telemetry_seen is set by any extension).
    retention_seen: bool = False
    #: True once shell-retention's cost gate kept a due jump raw (skippedCost).
    cost_deferred: bool = False
    turns_over_ctx: int = 0
    cache_reported: bool = False
    fired_once: set = field(default_factory=set)
    #: Recent turns as {turn, ts_start, ts_end, prompt, output, usage_reported,
    #: explained_by}; the
    #: join key for server-side prefix divergences (classify_divergence).
    turns: list = field(default_factory=list)
    #: Classified divergences by rule name.
    divergences: dict = field(default_factory=dict)


@dataclass
class DedupState:
    seen: set = field(default_factory=set)
    last_emit: dict = field(default_factory=dict)
    suppressed: dict = field(default_factory=dict)


def config_from_env(env: dict, overrides: dict) -> RuleConfig:
    """Defaults, then RUN_WATCH_<FIELD> from env, then non-None overrides (CLI flags)."""
    cfg = RuleConfig()
    for f in fields(RuleConfig):
        cast = type(getattr(cfg, f.name))
        raw = env.get(ENV_PREFIX + f.name.upper())
        if isinstance(raw, str) and raw.strip():
            try:
                setattr(cfg, f.name, cast(raw.strip()))
            except ValueError:
                pass
        if overrides.get(f.name) is not None:
            setattr(cfg, f.name, cast(overrides[f.name]))
    return cfg


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _n(v: Any) -> int:
    x = _num(v)
    return int(x) if x is not None and x > 0 else 0


def _task(trial_name: str) -> str:
    return trial_name.split("__", 1)[0]


def evaluate_turn(st: TrialState, rec: dict, cfg: RuleConfig) -> list[Alert]:
    """Fold one turns.jsonl record into the trial's state; return the alerts it raises."""
    ts = _num(rec.get("ts_end")) or 0.0
    subject = f"turn {rec.get('turn')}"
    alerts: list[Alert] = []

    def add(level: str, rule: str, message: str, subj: str = subject, data: Optional[dict] = None) -> None:
        alerts.append(Alert(ts, level, rule, st.name, subj, message, data or {}))

    def once(rule: str) -> bool:
        if rule in st.fired_once:
            return False
        st.fired_once.add(rule)
        return True

    comps = [c for c in rec.get("compactions") or [] if isinstance(c, dict)]
    if comps:
        st.compactions += sum(1 for c in comps if c.get("ok")
                              and (c.get("source") == "harness" or c.get("reason") != "manual"))
        # Only a compaction that happened rewrites history; a failed one
        # leaves the prompt growing from the same baseline.
        if any(c.get("ok") for c in comps):
            st.last_prompt_tokens = None
            # The cost gate is meant to open before compaction; reaching one
            # with every due pair still raw means it never did.
            if st.cost_deferred and st.max_demoted == 0 and once("deferred_into_compaction"):
                add("warn", "deferred_into_compaction",
                    f"compacted with {st.max_due} due shell pair(s) never demoted: the cost gate "
                    f"deferred them all the way (check LITTLE_CODER_SHELL_DEMOTE_OPEN_AT_PERCENT and the window)",
                    subj="once", data={"due": st.max_due})

    prompt, output = _n(rec.get("prompt_tokens")), _n(rec.get("output"))
    if rec.get("usage_reported"):
        prev = st.last_prompt_tokens
        if prev is not None and prompt - prev > cfg.context_jump_tokens:
            add("warn", "context_jump", f"prompt grew {prev:,} -> {prompt:,} tokens in one turn",
                data={"before": prev, "after": prompt})
        hit = _num(rec.get("cache_hit"))
        if (st.cache_reported and prompt > cfg.cache_min_prompt_tokens
                and hit is not None and hit < cfg.cache_min_hit_ratio):
            add("warn", "cache_collapse", f"cache hit {hit:.0%} on a {prompt:,}-token prompt",
                data={"prompt": prompt, "cache_hit": hit})
        if _n(rec.get("cache_read")) > 0:
            st.cache_reported = True
        st.last_prompt_tokens = prompt
        st.peak_prompt_tokens = max(st.peak_prompt_tokens, prompt)
        if prompt > cfg.no_demotion_context_tokens:
            st.turns_over_ctx += 1

    if rec.get("stop_reason") == "length":
        add("crit", "length_stop", f"response hit the output limit ({output:,} tokens; "
            f"largest tool call {_n(rec.get('max_tool_arg_bytes')):,} bytes)", data={"output": output})
    if output > cfg.max_output_tokens:
        add("warn", "big_output", f"one response produced {output:,} output tokens", data={"output": output})
    ttft = _num(rec.get("ttft_s"))
    if ttft is not None and ttft > cfg.ttft_max_s:
        add("warn", "ttft_spike", f"time to first token {ttft:.0f}s", data={"ttft_s": ttft})

    ret = rec.get("retention")
    if isinstance(ret, dict):
        st.telemetry_seen = True
        st.retention_seen = True
        large, prefix, demoted, signed = (_n(ret.get(k)) for k in ("large", "prefix", "demoted", "signed"))
        st.max_large_pairs = max(st.max_large_pairs, large)
        st.max_demoted = max(st.max_demoted, demoted)
        st.max_due = max(st.max_due, _n(ret.get("due")))
        skipped = _n(ret.get("skippedCost"))
        if skipped > 0:
            st.cost_deferred = True
            if once("demotion_deferred"):
                est = {k: _n(ret.get(k)) for k in ("estSaveTokens", "estReprefillTokens", "estContextTokens")}
                add("info", "demotion_deferred",
                    f"cost gate kept {skipped} due shell pair(s) raw: demoting would save "
                    f"~{est['estSaveTokens']:,} tokens but re-prefill ~{est['estReprefillTokens']:,} "
                    f"(context ~{est['estContextTokens']:,})",
                    subj="once", data={"skippedCost": skipped, **est})
        if prefix >= 1 and demoted == 0 and once("retention_stalled"):
            add("crit", "retention_stalled",
                f"{prefix} shell pair(s) due for demotion and none demoted (signed={signed}, "
                f"no-shrink={_n(ret.get('skippedNoShrink'))}, archive-refused={_n(ret.get('skippedArchive'))})",
                subj="once", data={"large": large, "prefix": prefix, "signed": signed})
        if signed > 0 and once("retention_signed"):
            add("warn", "retention_signed",
                f"{signed} due shell pair(s) treated as signed: their commands are never demoted",
                subj="once", data={"signed": signed})

    for g in rec.get("guards") or []:
        if not isinstance(g, dict):
            continue
        st.telemetry_seen = True
        kind = str(g.get("kind") or "telemetry")
        which = str(g.get("trigger") or g.get("source") or "?")
        detail = " ".join(f"{k}={v}" for k, v in sorted(g.items()) if k not in ("v", "kind", "trigger", "source"))
        add("info", kind, f"{which} {detail}".strip(), subj=f"{subject} {kind}:{which}",
            data={k: v for k, v in g.items() if k != "v"})

    if isinstance(rec.get("stubs"), dict):
        st.telemetry_seen = True
    stubbed_new = _n(rec.get("stubbed_new"))
    if stubbed_new > 0:
        add("info", "stub_applied", f"{stubbed_new} length-truncated response(s) stubbed out of the prompt",
            data={"stubbed_new": stubbed_new})

    st.n_turns += 1
    st.last_turn_ts = ts
    explained_by = []
    if _n(rec.get("demoted_new")) > 0:
        explained_by.append("demotion")
    if stubbed_new > 0:
        explained_by.append("stub")
    if any(c.get("ok") for c in comps):
        explained_by.append("compaction")
    st.turns.append({"turn": rec.get("turn"), "ts_start": _num(rec.get("ts_start")) or ts, "ts_end": ts,
                     "prompt": prompt, "output": output, "usage_reported": bool(rec.get("usage_reported")),
                     "explained_by": explained_by})
    del st.turns[:-MAX_REMEMBERED_TURNS]

    # A cost-gate deferral is the intended reason for none demoted. Without
    # shell_retention snapshots nothing says how many pairs were due, so the
    # long-context branch still stands in for it.
    if st.max_demoted == 0 and not st.cost_deferred and "no_demotions" not in st.fired_once:
        if st.max_large_pairs >= cfg.no_demotion_large_pairs:
            st.fired_once.add("no_demotions")
            add("crit", "no_demotions",
                f"{st.max_large_pairs} large shell outputs in history and none ever demoted",
                subj="once", data={"large": st.max_large_pairs})
        elif st.turns_over_ctx >= cfg.no_demotion_turns and (
                not st.retention_seen or st.max_due >= cfg.no_demotion_min_due):
            st.fired_once.add("no_demotions")
            add("warn", "no_demotions",
                f"{st.turns_over_ctx} turns over {cfg.no_demotion_context_tokens:,} prompt tokens "
                f"and no shell output ever demoted", subj="once", data={"turns": st.turns_over_ctx})
    if not st.telemetry_seen and st.n_turns >= cfg.telemetry_grace_turns and once("telemetry_missing"):
        add("info", "telemetry_missing",
            f"{st.n_turns} turns and no lc-telemetry: the demotion rules cannot see this trial", subj="once")
    return alerts


MAX_REMEMBERED_TURNS = 200
#: Clock slack when placing a server line inside a turn's [ts_start, ts_end].
DIVERGENCE_WINDOW_SLACK_S = 5.0
#: Tokens of slack around the previous output's span for divergence_in_output.
OUTPUT_SPAN_SLACK_TOKENS = 64


def is_mid_history_divergence(ev: dict, cfg: RuleConfig) -> bool:
    """An omlx prefix-cache line that diverged deep in history and re-prefills a lot."""
    if ev.get("kind") != "prefix_cache":
        return False
    reprefill, shared, comparable = (_n(ev.get(k)) for k in ("reprefill", "shared", "comparable"))
    return comparable - shared >= cfg.divergence_gap_tokens and reprefill >= cfg.divergence_reprefill_tokens


def classify_divergence(ev: dict, st: TrialState, cfg: RuleConfig, now: float) -> Optional[list[Alert]]:
    """Join one mid-history divergence to the turn whose request produced it.

    The match: the server line's timestamp falls inside a turn's
    [ts_start, ts_end] (+/- DIVERGENCE_WINDOW_SLACK_S) AND its prompt size
    ("re-prefills X of Y tokens", Y) is that turn's prompt_tokens within
    divergence_match_tokens. A turn that reported no usage (aborted or
    errored, prompt_tokens 0) matches on the window alone, used only when no
    turn matches on prompt size. omlx logs the line when prefill starts and the
    ledger writes the record at turn_end, so a divergence with no prompt-size
    match waits (returns None) until the record lands: a retry or follow-up can
    start inside an aborted turn's trailing slack before its own record exists.
    Once a later turn has started after the line, or after
    divergence_match_wait_s, it falls back to a window-alone match, else is
    decided as unmatched.

    Classes, by what the matched turn's own record says:
    - expected_divergence (info): the turn demoted shell pairs, stubbed a
      length-truncated message, or followed a successful compaction; each
      rewrites history on purpose (shell-retention batching).
    - divergence_in_output (info): the shared prefix ends inside the previous
      turn's generated output (the nearest earlier turn that reported usage),
      where omlx re-tokenizes the sampled tokens
      (jundot/omlx#4353); costly but not a harness defect.
    - prefix_divergence (warn): nothing in the ledger explains it.
    """
    ts = _num(ev.get("ts")) or 0.0
    prompt, shared, comparable, reprefill = (_n(ev.get(k)) for k in ("prompt", "shared", "comparable", "reprefill"))
    gap = comparable - shared
    slack = DIVERGENCE_WINDOW_SLACK_S
    by_prompt = by_window = None
    for i, t in enumerate(st.turns):
        if not t["ts_start"] - slack <= ts <= t["ts_end"] + slack:
            continue
        if not t["usage_reported"]:
            by_window = (i, t)
        elif abs(t["prompt"] - prompt) <= cfg.divergence_match_tokens:
            by_prompt = (i, t)
    matched = by_prompt or by_window
    data = {"reprefill": reprefill, "shared": shared, "comparable": comparable, "prompt": prompt}
    subject = f"{ev.get('src')} {ts:.3f}"
    detail = f"{gap:,} tokens before its end; re-prefilling {reprefill:,} of {prompt:,}"

    def out(level: str, rule: str, message: str, extra: dict) -> list[Alert]:
        st.divergences[rule] = st.divergences.get(rule, 0) + 1
        return [Alert(ts, level, rule, st.name, subject, message, {**data, **extra})]

    if by_prompt is None:
        later = any(t["ts_start"] - slack > ts for t in st.turns)
        if not later and now - ts <= cfg.divergence_match_wait_s:
            return None
    if matched is None:
        return out("warn", "prefix_divergence",
                   f"prompt diverged from the cached prefix {detail} (no matching turn record)", {"turn": None})
    i, t = matched
    if t["explained_by"]:
        return out("info", "expected_divergence",
                   f"turn {t['turn']} rewrote history ({', '.join(t['explained_by'])}): diverged {detail}",
                   {"turn": t["turn"], "explained_by": list(t["explained_by"])})
    prev = next((p for p in reversed(st.turns[:i]) if p["usage_reported"]), None)
    if (prev is not None and prev["output"] > 0
            and prev["prompt"] - OUTPUT_SPAN_SLACK_TOKENS <= shared
            <= prev["prompt"] + prev["output"] + OUTPUT_SPAN_SLACK_TOKENS):
        return out("info", "divergence_in_output",
                   f"turn {t['turn']} diverged inside the previous output (tokens {prev['prompt']:,}-"
                   f"{prev['prompt'] + prev['output']:,}): omlx re-tokenized it (jundot/omlx#4353); {detail}",
                   {"turn": t["turn"]})
    return out("warn", "prefix_divergence", f"prompt diverged from the cached prefix {detail}", {"turn": t["turn"]})


def evaluate_server_event(ev: dict, trial: Optional[str], cfg: RuleConfig) -> list[Alert]:
    """Alerts from one parsed server-log event (needs `ts`); stateless."""
    ts = _num(ev.get("ts"))
    if ts is None:
        return []
    subject = f"{ev.get('src')} {ts:.3f}"
    out: list[Alert] = []

    def add(level: str, rule: str, message: str, data: dict) -> None:
        out.append(Alert(ts, level, rule, trial, subject, message, data))

    kind = ev.get("kind")
    if kind == "prefix_cache":
        reprefill, shared, comparable, prompt = (_n(ev.get(k)) for k in ("reprefill", "shared", "comparable", "prompt"))
        gap = comparable - shared
        # A normal turn diverges near the tail (the new suffix) and re-prefills
        # roughly the new tokens; only a gap deep into cached history means
        # history was rewritten mid-prompt.
        if gap >= cfg.divergence_gap_tokens and reprefill >= cfg.divergence_reprefill_tokens:
            add("warn", "prefix_divergence",
                f"prompt diverged from the cached prefix {gap:,} tokens before its end; "
                f"re-prefilling {reprefill:,} of {prompt:,}",
                {"reprefill": reprefill, "shared": shared, "comparable": comparable, "prompt": prompt})
    elif kind == "completion":
        ttft = _num(ev.get("ttft_s"))
        if ttft is not None and ttft > cfg.ttft_max_s:
            add("warn", "ttft_spike", f"server time to first token {ttft:.0f}s", {"ttft_s": ttft})
        prompt, cached = ev.get("prompt"), ev.get("cached")
        if (isinstance(prompt, int) and isinstance(cached, int) and prompt > cfg.cache_min_prompt_tokens
                and cached / prompt < cfg.cache_min_hit_ratio):
            add("warn", "cache_collapse", f"server cache hit {cached / prompt:.0%} on a {prompt:,}-token prompt",
                {"prompt": prompt, "cached": cached})
        output = ev.get("output")
        if isinstance(output, int) and output > cfg.max_output_tokens:
            add("warn", "big_output",
                f"server finished a {output:,}-token response ({ev.get('status') or ev.get('finish_reason')})",
                {"output": output})
    elif kind == "server_error":
        add("info", "server_error", str(ev.get("text") or ""), {})
    return out


def evaluate_stall(st: TrialState, now: float, last_turn_ts: float, live_log_ts: Optional[float],
                   cfg: RuleConfig) -> list[Alert]:
    """One alert per idle period of an unfinished trial with no new turn for stall_min minutes."""
    limit = cfg.stall_min * 60
    if st.finished or now - last_turn_ts <= limit:
        return []
    if live_log_ts is not None and now - live_log_ts <= limit:
        note = f"live log written {(now - live_log_ts) / 60:.0f} min ago: slow, not hung"
    else:
        note = "live log quiet too: likely hung"
    idle_min = (now - last_turn_ts) / 60
    return [Alert(now, "warn", "stall", st.name, f"idle since {last_turn_ts:.0f}",
                  f"no new turn for {idle_min:.0f} min ({note})", {"idle_min": round(idle_min, 1)})]


def admit(state: DedupState, alert: Alert, cooldown_s: float, undo: Optional[list] = None) -> bool:
    """False for an identical (rule, trial, subject) seen before, or a non-crit within cooldown.

    An admitted alert appends what it changed to `undo`, for unadmit.
    """
    ident = (alert.rule, alert.trial, alert.subject)
    if ident in state.seen:
        return False
    state.seen.add(ident)
    key = (alert.rule, alert.trial)
    last = state.last_emit.get(key)
    if alert.level != "crit" and last is not None and alert.ts - last < cooldown_s:
        state.suppressed[key] = state.suppressed.get(key, 0) + 1
        return False
    held = state.suppressed.pop(key, 0)
    if held:
        alert.data["suppressed_before"] = held
    if undo is not None:
        undo.append((alert, ident, key, last, held))
    state.last_emit[key] = alert.ts
    return True


def unadmit(state: DedupState, undo: list) -> None:
    """Reverse admit() for the alerts in `undo`, newest first, so they can be admitted again."""
    for alert, ident, key, last, held in reversed(undo):
        state.seen.discard(ident)
        if last is None:
            state.last_emit.pop(key, None)
        else:
            state.last_emit[key] = last
        if held:
            state.suppressed[key] = state.suppressed.get(key, 0) + held
            alert.data.pop("suppressed_before", None)


def seed_dedup(state: DedupState, rows: list) -> None:
    """Mark alerts.jsonl rows as already raised, so a restart or --once never repeats them."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        rule, trial, subject = row.get("rule"), row.get("trial"), row.get("subject")
        state.seen.add((rule, trial, subject))
        ts = _num(row.get("ts"))
        if ts is not None:
            key = (rule, trial)
            state.last_emit[key] = max(ts, state.last_emit.get(key, ts))


def format_alert(a: Alert) -> str:
    stamp = datetime.fromtimestamp(a.ts).strftime("%H:%M:%S")
    held = a.data.get("suppressed_before")
    tail = f" (+{held} similar suppressed)" if held else ""
    return f"[{a.level.upper()} {stamp}] {_task(a.trial) if a.trial else '-'} {a.rule}: {a.message}{tail}"


def alert_to_row(a: Alert) -> dict:
    return {"ts": round(a.ts, 3), "iso": datetime.fromtimestamp(a.ts).astimezone().isoformat(timespec="seconds"),
            "level": a.level, "rule": a.rule, "trial": a.trial, "subject": a.subject,
            "message": a.message, "data": a.data}


# ── IO ────────────────────────────────────────────────────────────────────

#: turn_ledger.LEDGER_FILENAME; spelled out because this file imports nothing from the repo.
LEDGER_FILENAME = "turns.jsonl"
LIVE_LOG_FILENAME = "little_coder.live.log"
#: Created empty at trial start; non-empty only once the agent run has returned.
FINAL_LOG_FILENAME = "little_coder.log"
ALERTS_FILENAME = "alerts.jsonl"
ENV_JOBS_DIR = "RUN_WATCH_JOBS_DIR"
#: How far back from EOF the server tail starts; job_start_ts then drops
#: anything older than the watched job.
DEFAULT_SERVER_BACKLOG_BYTES = 1024 * 1024


@dataclass
class TailState:
    path: Path
    #: Server logs only: on first open, start this many bytes before EOF.
    start_at_end_bytes: Optional[int] = None
    inode: Optional[int] = None
    offset: int = 0
    partial: bytes = b""


@dataclass
class TrialWatch:
    dir: Path
    state: TrialState
    tail: TailState


@dataclass
class WatchState:
    target: Path
    cfg: RuleConfig
    server: Optional[TailState] = None
    server_kind: str = "auto"
    status_interval_s: float = 300.0
    job_dir: Optional[Path] = None
    job_start_ts: Optional[float] = None
    trials: dict = field(default_factory=dict)
    dedup: DedupState = field(default_factory=DedupState)
    last_status_ts: Optional[float] = None
    server_last: dict = field(default_factory=dict)
    #: (event, trial name) divergences waiting for their turn record.
    pending_divergences: list = field(default_factory=list)
    #: What the last poll_once's admit() calls changed, for _emit to undo.
    admit_undo: list = field(default_factory=list)
    #: Alerts whose write to alerts.jsonl failed; offered to admit() again next poll.
    unwritten: list = field(default_factory=list)


def read_new_lines(st: TailState) -> list[str]:
    """Complete lines appended since the last call.

    Reopens from the top when the inode changes (rotation) or the file shrinks
    (truncation); holds a partial trailing line until its newline arrives; a
    missing file reads as nothing.
    """
    try:
        fh = open(st.path, "rb")
    except OSError:
        return []
    with fh:
        info = os.fstat(fh.fileno())
        drop_first = False
        if st.inode != info.st_ino or info.st_size < st.offset:
            first_open = st.inode is None
            st.inode, st.offset, st.partial = info.st_ino, 0, b""
            if first_open and st.start_at_end_bytes is not None and info.st_size > st.start_at_end_bytes:
                st.offset = info.st_size - st.start_at_end_bytes
                fh.seek(st.offset - 1)
                drop_first = fh.read(1) != b"\n"
        fh.seek(st.offset)
        data = fh.read()
    st.offset += len(data)
    *complete, st.partial = (st.partial + data).split(b"\n")
    if drop_first and complete:
        complete = complete[1:]
    return [c.decode("utf-8", "replace").rstrip("\r") for c in complete]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _has_key(path: Path, key: str) -> bool:
    data = _read_json(path / "config.json")
    return isinstance(data, dict) and key in data


def resolve_job_dir(target: Path) -> Optional[Path]:
    """`target` itself when it is a Harbor job dir, else its newest job child."""
    if _has_key(target, "job_name"):
        return target
    try:
        jobs = [p for p in target.iterdir() if p.is_dir() and _has_key(p, "job_name")]
    except OSError:
        return None
    if not jobs:
        return None
    return max(jobs, key=lambda p: (_mtime(p / "config.json") or 0.0, p.name))


def discover_trials(job_dir: Path) -> list[Path]:
    try:
        return sorted(p for p in job_dir.iterdir() if p.is_dir() and _has_key(p, "trial_name"))
    except OSError:
        return []


def read_trial_result(trial_dir: Path) -> Optional[dict]:
    """{"reward", "passed", "finished_ts"} once result.json has finished_at, else None."""
    data = _read_json(trial_dir / "result.json")
    if not isinstance(data, dict) or not data.get("finished_at"):
        return None
    verifier = data.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    return {"reward": reward, "passed": reward == 1.0, "finished_ts": _mtime(trial_dir / "result.json")}


def attribute_trial(ts: float, windows: list) -> Optional[str]:
    """Name of the (name, start, end) window holding ts; the latest start wins."""
    best: Optional[tuple] = None
    for name, start, end in windows:
        if start <= ts <= end and (best is None or start > best[1]):
            best = (name, start)
    return best[0] if best else None


def _load_alert_rows(path: Path) -> list:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _append_alerts(job_dir: Path, alerts: list) -> None:
    with open(job_dir / ALERTS_FILENAME, "a", encoding="utf-8") as fh:
        for a in alerts:
            fh.write(json.dumps(alert_to_row(a), ensure_ascii=False) + "\n")


def _poll_server(ws: WatchState, now: float) -> list:
    assert ws.server is not None
    raw = read_new_lines(ws.server)
    if not raw:
        return []
    if ws.server_kind == "auto":
        ws.server_kind = detect_server_kind(raw) or ("omlx" if "omlx" in ws.server.path.name else "auto")
        if ws.server_kind == "auto":
            return []
    if ws.server_kind == "omlx":
        events = [e for e in map(parse_omlx_line, raw) if e is not None]
    else:
        events = [e for e in map(parse_splash_line, raw) if e is not None]
        stamps = anchor_splash_batch([e["tod"] for e in events], _mtime(ws.server.path) or now)
        for e, ts in zip(events, stamps):
            e["ts"] = ts
    windows = [(tw.state.name, tw.state.start_ts, tw.state.end_ts if tw.state.end_ts is not None else now)
               for tw in ws.trials.values()]
    # No job yet means nothing to attribute to: the backlog is all earlier work.
    floor = ws.job_start_ts if ws.job_start_ts is not None else float("inf")
    out = []
    for e in events:
        if e.get("kind") == "completion":
            ws.server_last = {"ttft_s": e.get("ttft_s"), "prompt": e.get("prompt"), "cached": e.get("cached")}
        if e["ts"] < floor:
            continue  # before this job: status context only, never an alert
        trial = attribute_trial(e["ts"], windows)
        if trial is not None and is_mid_history_divergence(e, ws.cfg):
            ws.pending_divergences.append((e, trial))  # classified against the ledger in poll_once
            continue
        out.extend(evaluate_server_event(e, trial, ws.cfg))
    return out


def _resolve_divergences(ws: WatchState, now: float) -> list:
    out, waiting = [], []
    for ev, trial in ws.pending_divergences:
        tw = ws.trials.get(trial)
        got = classify_divergence(ev, tw.state, ws.cfg, now) if tw is not None else None
        if got is None and tw is not None and tw.state.finished:
            got = classify_divergence(ev, tw.state, ws.cfg, float("inf"))
        if got is None:
            waiting.append((ev, trial))
        else:
            out.extend(got)
    ws.pending_divergences = waiting
    return out


def _k(n: Optional[int]) -> str:
    return "?" if n is None else f"{n / 1000:.0f}K"


def format_status(ws: WatchState, now: float) -> str:
    stamp = datetime.fromtimestamp(now).strftime("%H:%M")
    if ws.job_dir is None:
        return f"[status {stamp}] no Harbor job under {ws.target}"
    trials = [tw.state for tw in ws.trials.values()]
    done = [t for t in trials if t.finished]
    passed = sum(1 for t in done if t.reward == 1.0)
    config = _read_json(ws.job_dir / "config.json")
    try:
        total = len(config["datasets"][0]["task_names"])
    except (TypeError, KeyError, IndexError):
        total = len(trials)
    running = [t for t in trials if not t.finished]
    current = "none"
    if running:
        t = max(running, key=lambda s: s.start_ts)
        current = (f"{_task(t.name)} ({(now - t.start_ts) / 60:.0f} min, turn {t.n_turns}, "
                   f"ctx {_k(t.last_prompt_tokens)} peak {_k(t.peak_prompt_tokens)}, "
                   f"{t.compactions} cmp, demoted {t.max_demoted})"
                   f"{' verifying' if t.verifying else ''}")
    divs = {}
    for t in trials:
        for k, v in t.divergences.items():
            divs[k] = divs.get(k, 0) + v
    if divs:
        current += (f" | div {divs.get('expected_divergence', 0)} expected/"
                    f"{divs.get('divergence_in_output', 0)} output/{divs.get('prefix_divergence', 0)} unexplained")
    server = ""
    ttft, prompt, cached = (ws.server_last.get(k) for k in ("ttft_s", "prompt", "cached"))
    if isinstance(ttft, (int, float)):
        server += f" ttft {ttft:.1f}s"
    if isinstance(prompt, int) and isinstance(cached, int) and prompt > 0:
        server += f" cache {cached / prompt:.0%}"
    if server:
        server = " | server" + server
    return (f"[status {stamp}] {ws.job_dir.name}: {len(done)}/{total} done, pass={passed} "
            f"fail={len(done) - passed}; current: {current}{server}")


def poll_once(ws: WatchState, now: float, force_status: bool = False) -> tuple[list[Alert], list[str]]:
    """One pass: new turn records, finished trials, server lines, stalls; admitted alerts and output lines."""
    lines: list[str] = []
    job = resolve_job_dir(ws.target)
    if job is not None and job != ws.job_dir:
        ws.job_dir, ws.trials, ws.dedup, ws.unwritten = job, {}, DedupState(), []
        ws.job_start_ts = _mtime(job / "config.json")
        seed_dedup(ws.dedup, _load_alert_rows(job / ALERTS_FILENAME))
        lines.append(f"watching job {job}")
    alerts: list[Alert] = ws.unwritten
    ws.unwritten = []
    if ws.job_dir is not None:
        for tdir in discover_trials(ws.job_dir):
            tw = ws.trials.get(tdir.name)
            if tw is None:
                tw = TrialWatch(dir=tdir,
                                state=TrialState(name=tdir.name, start_ts=_mtime(tdir / "config.json") or now),
                                tail=TailState(path=tdir / "agent" / LEDGER_FILENAME))
                ws.trials[tdir.name] = tw
            for raw in read_new_lines(tw.tail):
                record = parse_turn_record(raw)
                if record is not None:
                    alerts.extend(evaluate_turn(tw.state, record, ws.cfg))
            if not tw.state.finished:
                res = read_trial_result(tdir)
                if res is not None:
                    tw.state.finished, tw.state.reward, tw.state.end_ts = True, res["reward"], res["finished_ts"]
                    lines.append(f"trial {_task(tdir.name)} finished {'PASS' if res['passed'] else 'FAIL'} "
                                 f"(reward={res['reward']}, {tw.state.n_turns} turns, "
                                 f"peak {_k(tw.state.peak_prompt_tokens)})")
            if (not tw.state.finished and not tw.state.verifying
                    and _size(tdir / "agent" / FINAL_LOG_FILENAME) > 0):
                tw.state.verifying = True
                lines.append(f"trial {_task(tdir.name)} agent done, verifying")
    if ws.server is not None:
        alerts.extend(_poll_server(ws, now))
    alerts.extend(_resolve_divergences(ws, now))
    for tw in ws.trials.values():
        if not tw.state.finished and not tw.state.verifying:
            last_turn = max(tw.state.start_ts, _mtime(tw.tail.path) or 0.0)
            live = _mtime(tw.dir / "agent" / LIVE_LOG_FILENAME)
            alerts.extend(evaluate_stall(tw.state, now, last_turn, live, ws.cfg))
    ws.admit_undo = []
    admitted = [a for a in alerts if admit(ws.dedup, a, ws.cfg.cooldown_s, ws.admit_undo)]
    lines.extend(format_alert(a) for a in admitted)
    if force_status or ws.last_status_ts is None or now - ws.last_status_ts >= ws.status_interval_s:
        lines.append(format_status(ws, now))
        ws.last_status_ts = now
    return admitted, lines


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Watch a running Harbor job and raise alerts.")
    p.add_argument("target", nargs="?", type=Path, default=None,
                   help=f"a Harbor job dir, or a jobs dir whose newest job is watched "
                        f"(default ${ENV_JOBS_DIR}, else ./benchmarks/harbor_runs)")
    p.add_argument("--server-log", type=Path, help="omlx.log or a Splash serve log to tail")
    p.add_argument("--server-kind", choices=["auto", "omlx", "splash"], default="auto")
    p.add_argument("--once", action="store_true", help="one pass, then exit (tests, cron)")
    p.add_argument("--interval", type=float, default=15.0, help="seconds between polls (default 15)")
    p.add_argument("--status-interval", type=float, default=300.0, help="seconds between status lines (default 300)")
    p.add_argument("--server-backlog-bytes", type=int, default=DEFAULT_SERVER_BACKLOG_BYTES,
                   help="how far back from the end of the server log to start (default 1 MiB)")
    for f in fields(RuleConfig):
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=None,
                       help=f"default {f.default}; env {ENV_PREFIX}{f.name.upper()}")
    return p


def _emit(ws: WatchState, admitted: list, lines: list, out: Any) -> bool:
    """Append admitted alerts to alerts.jsonl and print the lines; False when the append failed.

    A failed append undoes those alerts' admission and queues them for the
    next poll, so they are raised again rather than lost.
    """
    ok = True
    if admitted and ws.job_dir is not None:
        try:
            _append_alerts(ws.job_dir, admitted)
        except OSError as exc:
            ok = False
            unadmit(ws.dedup, ws.admit_undo)
            ws.unwritten = list(admitted)
            print(f"could not write {ALERTS_FILENAME}: {exc}", file=out, flush=True)
    for line in lines:
        print(line, file=out, flush=True)
    return ok


def main(argv: Optional[list[str]] = None, *, now: Callable[[], float] = time.time, out: Any = None) -> int:
    out = out if out is not None else sys.stdout
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not (0 < args.interval < float("inf")):
        parser.error("--interval must be a positive, finite number of seconds")
    cfg = config_from_env(dict(os.environ), {f.name: getattr(args, f.name) for f in fields(RuleConfig)})
    server = (TailState(path=args.server_log, start_at_end_bytes=args.server_backlog_bytes)
              if args.server_log else None)
    target = args.target or Path(os.environ.get(ENV_JOBS_DIR) or Path.cwd() / "benchmarks" / "harbor_runs")
    if not target.exists():
        print(f"run_watch: {target} does not exist; pass a job or jobs dir, or set {ENV_JOBS_DIR}",
              file=sys.stderr, flush=True)
        return 2
    ws = WatchState(target=target, cfg=cfg, server=server, server_kind=args.server_kind,
                    status_interval_s=args.status_interval)
    if args.once:
        try:
            admitted, lines = poll_once(ws, now(), force_status=True)
        except Exception as exc:
            print(f"poll failed: {type(exc).__name__}: {exc}", file=out, flush=True)
            return 2
        if not _emit(ws, admitted, lines, out):
            return 2
        return 1 if any(a.level == "crit" for a in admitted) else 0

    stopping = False

    def _stop(signum: int, frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while not stopping:
        try:
            admitted, lines = poll_once(ws, now())
        except Exception as exc:  # one bad poll must not end the watch
            admitted, lines = [], [f"poll failed: {type(exc).__name__}: {exc}"]
        _emit(ws, admitted, lines, out)
        deadline = time.monotonic() + args.interval
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
