#!/usr/bin/env python3
"""Post-hoc per-turn omlx report for one Harbor trial.

Reads the trial's own time window out of result.json, slices ~/.omlx/logs/server.log
to it, and writes <trial-dir>/omlx_turns.csv plus a short stdout summary:

    benchmarks/omlx_metrics_report.py --trial-dir benchmarks/harbor_runs/<job>/<task>__<id>

server.log rotates daily, so an older trial needs --server-log pointed at the
matching ~/.omlx/logs/server.log.<date>; a window crossing local midnight is
read from the given file and its rotated siblings together. Nothing here talks
to a live server.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_SERVER_LOG = Path.home() / ".omlx" / "logs" / "server.log"
CSV_FILENAME = "omlx_turns.csv"
CSV_COLUMNS = [
    "turn_idx",
    "prompt_tokens",
    "output_tokens",
    "reused_tokens",
    "reprefill_tokens",
    "decode_tok_s",
    "turn_wall_s",
    "throttle_events",
    "reclaimed_gb",
    "finish_reason",
    "mtp_rounds",
    "mtp_accepted",
    "mtp_proposed",
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
# omlx names a rotated log for the day it covers: server.log.2026-09-13.
_ROTATED_SUFFIX_RE = re.compile(r"\.\d{4}-\d{2}-\d{2}$")


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
    """The trial's [started_at, finished_at] as UTC.

    Harbor only fills agent_execution once environment and agent setup have
    succeeded, and a run-level result.json never has it at all, so its absence
    is a routine input -- say which file is wrong instead of raising TypeError
    or KeyError from a subscript.
    """
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    execution = result.get("agent_execution")
    started_at = execution.get("started_at") if isinstance(execution, dict) else None
    finished_at = execution.get("finished_at") if isinstance(execution, dict) else None
    if started_at is None or finished_at is None:
        raise SystemExit(
            f"{trial_dir / 'result.json'} has no agent_execution timing (trial likely failed "
            "during setup, or this is a run directory rather than a trial directory) "
            "-- nothing to report"
        )
    return parse_iso_utc(started_at), parse_iso_utc(finished_at)


def rotated_siblings(
    server_log: Path, started_at: datetime, finished_at: datetime
) -> tuple[list[Path], list]:
    """Logs holding the part of the window the given file cannot, newest last,
    plus any day in the window whose log is simply gone.

    omlx rotates server.log at local midnight, so a window crossing midnight is
    split across files: each past day lands in a server.log.<date> sibling,
    while the newest day is normally still in the live server.log. Empty for a
    same-day window, and never names the file already being read.

    Only files still on disk are returned as siblings. ~/.omlx/settings.json
    sets logging.retention_days, so a day in an old window may have been
    deleted already -- the rest of the window is still worth reporting on, but
    the caller needs to know a day is simply missing rather than assume the
    merge is complete. The newest day is exempt: it always resolves to
    whatever file is current (server_log itself, or its own dated file), never
    "missing" in this sense.
    """
    first = started_at.astimezone().date()
    last = finished_at.astimezone().date()
    if last <= first:
        return [], []
    base = _ROTATED_SUFFIX_RE.sub("", server_log.name)
    days = [first + timedelta(days=offset) for offset in range((last - first).days + 1)]
    holders = [server_log.parent / f"{base}.{day}" for day in days]
    # server_log itself is never "missing" here -- whether it exists is the
    # caller's problem (collect_events/main already check that), not a gap in
    # the merge this function is reporting on.
    missing_days = [
        day for day, holder in zip(days[:-1], holders[:-1]) if holder != server_log and not holder.exists()
    ]
    if not holders[-1].exists():
        holders[-1] = server_log.parent / base
    existing = [path for path in holders if path != server_log and path.exists()]
    return existing, missing_days


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


def collect_window_events(
    server_log: Path, started_at: datetime, finished_at: datetime
) -> list[tuple[datetime, str, dict]]:
    """Events from every log holding part of the window, oldest first.

    build_turns walks the list statefully -- pending prefix/MTP stats and the
    throttle tally carry forward to the next completion -- so events merged out
    of several files are re-sorted by timestamp rather than left in file order.
    """
    siblings, missing_days = rotated_siblings(server_log, started_at, finished_at)
    logs = [server_log, *siblings]
    if siblings or missing_days:
        first = started_at.astimezone().date()
        last = finished_at.astimezone().date()
        named = ", ".join(str(path) for path in logs)
        print(
            f"trial window spans {first} to {last} locally and server.log rotates at local "
            f"midnight, so reading {len(logs)} log files to cover it: {named}",
            file=sys.stderr,
        )
    if missing_days:
        named_days = ", ".join(str(day) for day in missing_days)
        print(
            f"INCOMPLETE: no log on disk for {named_days} (past logging.retention_days) -- "
            "events from that day are not in this report, do not treat it as a full window",
            file=sys.stderr,
        )
    events: list[tuple[datetime, str, dict]] = []
    for log in logs:
        events.extend(collect_events(log, started_at, finished_at))
    events.sort(key=lambda event: event[0])
    return events


def build_turns(events, started_at: datetime) -> tuple[list[dict], list[dict], dict]:
    """Turn rows, deduplicated rejections, and whatever trailed the last turn.

    Throttles and reclaims are attributed to the completion that follows them,
    so ones logged after the last completion -- a window ending in a prefill
    rejection or a kill mid-throttle -- have no row to land in. They are
    returned separately rather than dropped, which otherwise reads as a run
    that never throttled. `turn_idx` is a row's 1-based position in the
    window, not an agent-loop turn number.
    """
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
            # The rejected request never reaches a Chat completion line, so its
            # cache/MTP stats would otherwise wait and attach to whichever later
            # completion happens to share its prompt length.
            pending_prefix = None
            pending_mtp = None
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
                    "mtp_proposed": int(pending_mtp["proposed"]) if pending_mtp else "",
                    "mtp_tokens_per_round": float(pending_mtp["tpr"]) if pending_mtp else "",
                    "mtp_block_size": int(pending_mtp["block"]) if pending_mtp else "",
                }
            )
            pending_prefix = None
            pending_mtp = None
            throttles = 0
            reclaimed_gb = 0.0
            window_start = ts

    for i, turn in enumerate(turns):
        turn["turn_idx"] = i + 1
    trailing = {"throttle_events": throttles, "reclaimed_gb": round(reclaimed_gb, 2)}
    return turns, rejections, trailing


def _same_rejection(previous: dict, current: dict) -> bool:
    """Whether two rejection lines describe one event, not two.

    Every field of the message is compared: a client retrying the same
    over-length prompt, or two requests hitting the same cap at once, produces
    genuinely distinct rejections that agree on some of them.
    """
    return (
        previous["kv_len"] == current["kv_len"]
        and previous["chunk_tokens"] == current["chunk_tokens"]
        and previous["cap_gb"] == current["cap_gb"]
        and previous["ceiling_gb"] == current["ceiling_gb"]
        and abs((current["ts"] - previous["ts"]).total_seconds()) < 2.0
    )


def write_csv(path: Path, turns: list[dict]) -> None:
    # Default extrasaction ("raise"): a measurement build_turns computes but
    # CSV_COLUMNS forgets is a bug, and ignoring it loses the column silently.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for turn in turns:
            writer.writerow(turn)


def print_summary(turns: list[dict], rejections: list[dict], trailing: dict, csv_path: Path) -> None:
    print(f"wrote {csv_path}")
    if not turns:
        print("turns: 0 (no `Chat completion:` lines in the trial's window -- wrong --server-log?)")
    else:
        decode = [t["decode_tok_s"] for t in turns]
        print(f"turns: {len(turns)}")
        print(f"median decode: {statistics.median(decode):.1f} tok/s")
        print(f"max prompt: {max(t['prompt_tokens'] for t in turns)} tokens")
        print(f"throttle events: {sum(t['throttle_events'] for t in turns)}")
    if trailing["throttle_events"] or trailing["reclaimed_gb"]:
        counts = f"{trailing['throttle_events']} throttle events, {trailing['reclaimed_gb']}GB reclaimed"
        if turns:
            print(f"after the last completion: {counts} (no turn row in {csv_path.name} holds these)")
        else:
            # "after the last completion" would name a completion that never happened.
            print(f"{counts} (no completions in this window at all)")
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
    events = collect_window_events(server_log, started_at, finished_at)
    turns, rejections, trailing = build_turns(events, started_at)
    csv_path = trial_dir / CSV_FILENAME
    write_csv(csv_path, turns)
    print_summary(turns, rejections, trailing, csv_path)
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
    if not args.trial_dir.is_dir():
        parser.error(f"trial directory not found: {args.trial_dir}")
    if not args.server_log.exists():
        parser.error(f"server log not found: {args.server_log}")
    return report(args.trial_dir, args.server_log)


if __name__ == "__main__":
    sys.exit(main())
