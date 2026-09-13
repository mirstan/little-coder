#!/usr/bin/env python3
"""Post-hoc per-turn omlx report for one Harbor trial.

Reads the trial's own time window out of result.json, slices ~/.omlx/logs/server.log
to it, and writes <trial-dir>/omlx_turns.csv plus a short stdout summary:

    benchmarks/omlx_metrics_report.py --trial-dir benchmarks/harbor_runs/<ts>/<task>__<id>

server.log rotates daily, so an older trial needs --server-log pointed at the
matching ~/.omlx/logs/server.log.<date>. Nothing here talks to a live server.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SERVER_LOG = Path.home() / ".omlx" / "logs" / "server.log"
CSV_FILENAME = "omlx_turns.csv"
CSV_COLUMNS = [
    "turn_idx",
    "prompt_tokens",
    "reused_tokens",
    "reprefill_tokens",
    "decode_tok_s",
    "turn_wall_s",
    "throttle_events",
    "reclaimed_gb",
    "finish_reason",
    "mtp_rounds",
    "mtp_accepted",
    "mtp_tokens_per_round",
    "mtp_block_size",
]

_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) ")
_CHAT_RE = re.compile(
    r"Chat completion: model=(?P<model>[^,]+), (?P<out>\d+) tokens in (?P<gen_s>[\d.]+)s "
    r"\((?P<tok_s>[\d.]+) tok/s\), prompt: (?P<prompt>\d+), finish_reason=(?P<finish>[^,\s]+)"
)
_PREFIX_RE = re.compile(
    r"prefix cache: request (?P<req>\S+) re-prefills (?P<reprefill>\d+) of (?P<total>\d+) tokens "
    r"\(reused (?P<reused>\d+)\)"
)
_THROTTLE_RE = re.compile(r"needs prefill headroom before throttling")
_RECLAIM_RE = re.compile(r"Reclaimed (?P<gb>[\d.]+)GB of pooled Metal buffers")
_REJECT_RE = re.compile(
    r"Prefill context too large for available memory \(pre-chunk guard at (?P<chunk>\d+) tokens, "
    r"kv_len=(?P<kv_len>\d+)\): predicted peak would exceed prefill safety cap (?P<cap_gb>[\d.]+)GB "
    r"\([^)]*ceiling (?P<ceiling_gb>[\d.]+)GB\)"
)
_MTP_RE = re.compile(
    r"vlm_mtp stats: request=(?P<req>\S+) finish=(?P<finish>\S+) rounds=(?P<rounds>\d+) "
    r"accepted=(?P<accepted>\d+)/(?P<proposed>\d+) \((?P<pct>[\d.]+)%\) "
    r"tokens_per_round=(?P<tpr>[\d.]+) emitted=(?P<emitted>\d+) block_size=(?P<block>\d+)"
)
_TURN_MARKER_RE = re.compile(r"^=== turn (\d+) start ===", re.MULTILINE)


def parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_log_ts(line: str) -> datetime | None:
    """Timestamp of a server.log line, as UTC.

    server.log stamps local wall-clock with no offset while result.json is UTC,
    so the naive value is localized first -- astimezone() on a naive datetime
    applies the host's offset *for that date*, which keeps DST boundaries right.
    Comparing the two clocks unconverted silently shifts every window by hours.
    """
    match = _LOG_TS_RE.match(line)
    if not match:
        return None
    naive = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
    return naive.astimezone(timezone.utc)


def read_window(trial_dir: Path) -> tuple[datetime, datetime]:
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    execution = result["agent_execution"]
    return parse_iso_utc(execution["started_at"]), parse_iso_utc(execution["finished_at"])


def read_turn_labels(trial_dir: Path) -> list[int]:
    live_log = trial_dir / "agent" / "little_coder.live.log"
    if not live_log.exists():
        return []
    text = live_log.read_text(encoding="utf-8", errors="replace")
    return [int(n) for n in _TURN_MARKER_RE.findall(text)]


def collect_events(server_log: Path, started_at: datetime, finished_at: datetime) -> list[tuple[datetime, str, dict]]:
    events: list[tuple[datetime, str, dict]] = []
    with server_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            ts = parse_log_ts(line)
            if ts is None or ts < started_at or ts > finished_at:
                continue
            chat = _CHAT_RE.search(line)
            if chat:
                events.append((ts, "chat", chat.groupdict()))
                continue
            prefix = _PREFIX_RE.search(line)
            if prefix:
                events.append((ts, "prefix", prefix.groupdict()))
                continue
            mtp = _MTP_RE.search(line)
            if mtp:
                events.append((ts, "mtp", mtp.groupdict()))
                continue
            reject = _REJECT_RE.search(line)
            if reject:
                events.append((ts, "reject", reject.groupdict()))
                continue
            reclaim = _RECLAIM_RE.search(line)
            if reclaim:
                events.append((ts, "reclaim", reclaim.groupdict()))
                continue
            if _THROTTLE_RE.search(line):
                events.append((ts, "throttle", {}))
    return events


def build_turns(events, turn_labels: list[int], started_at: datetime) -> tuple[list[dict], list[dict]]:
    turns: list[dict] = []
    rejections: list[dict] = []
    pending_prefix: dict | None = None
    pending_mtp: dict | None = None
    throttles = 0
    reclaimed_gb = 0.0
    window_start = started_at

    for ts, kind, fields in events:
        if kind == "prefix":
            pending_prefix = fields
        elif kind == "mtp":
            pending_mtp = fields
        elif kind == "throttle":
            throttles += 1
        elif kind == "reclaim":
            reclaimed_gb += float(fields["gb"])
        elif kind == "reject":
            rejection = {
                "ts": ts,
                "kv_len": int(fields["kv_len"]),
                "chunk_tokens": int(fields["chunk"]),
                "cap_gb": float(fields["cap_gb"]),
                "ceiling_gb": float(fields["ceiling_gb"]),
            }
            # One rejection is logged twice -- once by omlx.scheduler and again by
            # omlx.server relaying it to the client -- with the same text and
            # timestamp, so an identical back-to-back match is the same event.
            if not rejections or not _same_rejection(rejections[-1], rejection):
                rejections.append(rejection)
        elif kind == "chat":
            prompt_tokens = int(fields["prompt"])
            # omlx logs no shared request id across `prefix cache:` and `Chat
            # completion:`, so the pairing is positional: the prefix-cache line
            # immediately preceding a completion whose `total` equals that
            # completion's `prompt` is taken to describe the same request. It
            # holds in practice but is a heuristic, not a guaranteed join --
            # a completion with no matching prefix line gets blank cache columns
            # rather than a guess.
            matched = pending_prefix if pending_prefix and int(pending_prefix["total"]) == prompt_tokens else None
            turns.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": int(fields["out"]),
                    "reused_tokens": int(matched["reused"]) if matched else "",
                    "reprefill_tokens": int(matched["reprefill"]) if matched else "",
                    "decode_tok_s": float(fields["tok_s"]),
                    "turn_wall_s": round((ts - window_start).total_seconds(), 2),
                    "throttle_events": throttles,
                    "reclaimed_gb": round(reclaimed_gb, 2),
                    "finish_reason": fields["finish"],
                    "mtp_rounds": int(pending_mtp["rounds"]) if pending_mtp else "",
                    "mtp_accepted": int(pending_mtp["accepted"]) if pending_mtp else "",
                    "mtp_tokens_per_round": float(pending_mtp["tpr"]) if pending_mtp else "",
                    "mtp_block_size": int(pending_mtp["block"]) if pending_mtp else "",
                }
            )
            pending_prefix = None
            pending_mtp = None
            throttles = 0
            reclaimed_gb = 0.0
            window_start = ts

    _assign_turn_idx(turns, turn_labels)
    return turns, rejections


def _same_rejection(previous: dict, current: dict) -> bool:
    return (
        previous["kv_len"] == current["kv_len"]
        and previous["ceiling_gb"] == current["ceiling_gb"]
        and abs((current["ts"] - previous["ts"]).total_seconds()) < 2.0
    )


def _assign_turn_idx(turns: list[dict], turn_labels: list[int]) -> None:
    """Label each completion with a turn number.

    A live-log `=== turn N start ===` marker is an agent-loop turn, which usually
    spans several model completions, and the live log carries no timestamps to
    split them by -- so the labels are only usable when there is exactly one
    marker per completion. Otherwise rows fall back to their own 1-based position
    in the window, which is what `turn_idx` then means.
    """
    aligned = len(turn_labels) == len(turns)
    for i, turn in enumerate(turns):
        turn["turn_idx"] = turn_labels[i] if aligned else i + 1


def write_csv(path: Path, turns: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for turn in turns:
            writer.writerow(turn)


def print_summary(turns: list[dict], rejections: list[dict], csv_path: Path) -> None:
    print(f"wrote {csv_path}")
    if not turns:
        print("turns: 0 (no `Chat completion:` lines in the trial's window -- wrong --server-log?)")
    else:
        decode = [t["decode_tok_s"] for t in turns]
        print(f"turns: {len(turns)}")
        print(f"median decode: {statistics.median(decode):.1f} tok/s")
        print(f"max prompt: {max(t['prompt_tokens'] for t in turns)} tokens")
        print(f"throttle events: {sum(t['throttle_events'] for t in turns)}")
    if rejections:
        print(f"prefill rejections: {len(rejections)}")
        for rejection in rejections:
            print(
                f"  {rejection['ts'].isoformat()} kv_len={rejection['kv_len']} "
                f"cap={rejection['cap_gb']}GB ceiling={rejection['ceiling_gb']}GB"
            )
    else:
        print("prefill rejections: none")


def report(trial_dir: Path, server_log: Path) -> int:
    started_at, finished_at = read_window(trial_dir)
    events = collect_events(server_log, started_at, finished_at)
    turns, rejections = build_turns(events, read_turn_labels(trial_dir), started_at)
    csv_path = trial_dir / CSV_FILENAME
    write_csv(csv_path, turns)
    print_summary(turns, rejections, csv_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trial-dir", required=True, type=Path, help="a single Harbor trial directory")
    parser.add_argument(
        "--server-log",
        type=Path,
        default=DEFAULT_SERVER_LOG,
        help=f"omlx server log to parse (default: {DEFAULT_SERVER_LOG})",
    )
    args = parser.parse_args(argv)
    if not args.server_log.exists():
        parser.error(f"server log not found: {args.server_log}")
    return report(args.trial_dir, args.server_log)


if __name__ == "__main__":
    sys.exit(main())
