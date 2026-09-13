#!/usr/bin/env python3
"""Sample omlx server health + host memory pressure for the duration of a Harbor run.

Typically launched alongside benchmarks/harbor_pilot.sh, backgrounded, and killed
when the run finishes:

    benchmarks/omlx_metrics_poller.py --run-dir <dir> &

harbor_pilot.sh only points harbor at benchmarks/harbor_runs as its --jobs-dir;
harbor names the per-job directory under it, and the names vary by job type
(harbor_status.sh looks for tb2-*, leaderboard-*, full-* and harbor-* alongside
plain launch timestamps), so no fixed path can be given here. Any directory
works -- it is created if missing.

Writes one JSON object per sample to <run-dir>/omlx_metrics.jsonl, and raises a
SWAP_TRIPWIRE_FIRED sentinel in the run dir when host swap stays far above where
it started. Detect-and-flag only: it never restarts omlx or edits any config.
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

STATUS_URL = "http://127.0.0.1:8000/api/status"
METRICS_FILENAME = "omlx_metrics.jsonl"
SENTINEL_FILENAME = "SWAP_TRIPWIRE_FIRED"
SWAP_TRIPWIRE_MB = 2048.0
SWAP_TRIPWIRE_HOLD_S = 300.0

_SWAP_USED_RE = re.compile(r"used\s*=\s*([0-9.]+)\s*([KMGT])", re.IGNORECASE)
_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_PAGES_FREE_RE = re.compile(r"^Pages free:\s+(\d+)", re.MULTILINE)
_SWAP_UNIT_MB = {"K": 1.0 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}


def run_command(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=15).stdout


def parse_swap_used_mb(text: str) -> float:
    """Pull the used figure out of `sysctl vm.swapusage`.

    macOS normally reports M, but the unit is part of the format rather than a
    constant, so it is read rather than assumed.
    """
    match = _SWAP_USED_RE.search(text)
    if not match:
        raise ValueError(f"no `used =` field in vm.swapusage output: {text!r}")
    return float(match.group(1)) * _SWAP_UNIT_MB[match.group(2).upper()]


def parse_vm_stat(text: str) -> tuple[int, int]:
    """Return (pages_free, page_size_bytes) from `vm_stat` output.

    The page size is 16384 on Apple Silicon and 4096 on Intel, and vm_stat states
    it in its own header line, so it is taken from there rather than hardcoded --
    pages_free is meaningless without the matching multiplier.
    """
    page_match = _PAGE_SIZE_RE.search(text)
    if not page_match:
        raise ValueError(f"no page size in vm_stat header: {text.splitlines()[:1]}")
    free_match = _PAGES_FREE_RE.search(text)
    if not free_match:
        raise ValueError("no `Pages free:` line in vm_stat output")
    return int(free_match.group(1)), int(page_match.group(1))


def fetch_status(url: str = STATUS_URL, timeout: float = 5.0):
    """Return the parsed /api/status body, or None if omlx is not answering.

    omlx gets restarted mid-run often enough that an unreachable server is a
    normal sample, not an error -- the caller keeps polling either way. A
    half-started server answering with a truncated or malformed response raises
    HTTPException, which is outside OSError/ValueError; letting it out would
    lose the whole sample row, including the swap figure the tripwire runs on.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, http.client.HTTPException):
        return None


class SwapTripwire:
    """Fires once host swap has stayed >threshold_mb above baseline continuously.

    A single sample back under the threshold resets the clock rather than merely
    pausing it: the signal worth flagging is sustained pressure, and transient
    spikes during model load/unload are expected and uninteresting.
    """

    def __init__(self, threshold_mb: float = SWAP_TRIPWIRE_MB, hold_s: float = SWAP_TRIPWIRE_HOLD_S):
        self.threshold_mb = threshold_mb
        self.hold_s = hold_s
        self.baseline_mb: float | None = None
        self.elevated_since: float | None = None
        self.fired = False

    def observe(self, now: float, swap_used_mb: float) -> bool:
        """Feed one sample; returns True on the sample that first trips it.

        `now` measures elapsed time, so it must come from a monotonic clock: an
        NTP correction or a DST shift in the middle of a run would otherwise
        lengthen or shorten the hold window by that jump.
        """
        if self.baseline_mb is None:
            self.baseline_mb = swap_used_mb
            return False
        if swap_used_mb - self.baseline_mb <= self.threshold_mb:
            self.elevated_since = None
            return False
        if self.elevated_since is None:
            self.elevated_since = now
            return False
        if self.fired or now - self.elevated_since <= self.hold_s:
            return False
        self.fired = True
        return True


def sample(now: float, run: Callable[[list[str]], str] = run_command, status_url: str = STATUS_URL) -> dict:
    swap_used_mb = parse_swap_used_mb(run(["sysctl", "vm.swapusage"]))
    pages_free, page_size_bytes = parse_vm_stat(run(["vm_stat"]))
    return {
        "ts": now,
        "api_status": fetch_status(status_url),
        "swap_used_mb": swap_used_mb,
        "pages_free": pages_free,
        "page_size_bytes": page_size_bytes,
    }


def write_sentinel(path: Path, now: float, baseline_mb: float, swap_used_mb: float, hold_s: float) -> None:
    path.write_text(
        "omlx swap tripwire fired\n"
        f"fired_at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))} (epoch {now:.0f})\n"
        f"swap_baseline_mb: {baseline_mb:.2f}\n"
        f"swap_used_mb: {swap_used_mb:.2f}\n"
        f"delta_mb: {swap_used_mb - baseline_mb:.2f}\n"
        f"sustained_for_s: >{hold_s:.0f}\n"
        "\n"
        "Host swap stayed far above its run-start level. Nothing was changed\n"
        "automatically -- an operator decides whether to intervene.\n"
    )


def poll(
    run_dir: Path,
    interval: float,
    status_url: str = STATUS_URL,
    run: Callable[[list[str]], str] = run_command,
    clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> int:
    """Sample until signalled. Two clocks, because they answer different questions.

    `clock` is wall-clock and only ever gets recorded: the JSONL `ts` has to line
    up with server.log's wall-clock stamps for the report to correlate them, and
    the sentinel states a date an operator can read. Everything measuring elapsed
    time -- the tripwire's hold window, the gap between samples -- runs off
    `monotonic_clock` so a clock adjustment mid-run cannot stretch or collapse it.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / METRICS_FILENAME
    sentinel_path = run_dir / SENTINEL_FILENAME
    tripwire = SwapTripwire()

    stopping = False

    def _stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"polling {status_url} every {interval:g}s -> {metrics_path}", flush=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        while not stopping:
            now = clock()
            elapsed = monotonic_clock()
            try:
                row = sample(now, run=run, status_url=status_url)
            except Exception as exc:  # a bad sample must not end the run's monitoring
                print(f"sample failed: {exc}", flush=True)
                _sleep_until(elapsed + interval, lambda: stopping, monotonic_clock)
                continue

            handle.write(json.dumps(row) + "\n")
            handle.flush()

            if tripwire.observe(elapsed, row["swap_used_mb"]):
                print(
                    f"SWAP TRIPWIRE: swap {row['swap_used_mb']:.0f}MB is "
                    f"{row['swap_used_mb'] - tripwire.baseline_mb:.0f}MB above the "
                    f"{tripwire.baseline_mb:.0f}MB baseline and has been for over "
                    f"{tripwire.hold_s:g}s",
                    flush=True,
                )
                if sentinel_path.exists():
                    print(f"sentinel already present at {sentinel_path}", flush=True)
                else:
                    write_sentinel(sentinel_path, now, tripwire.baseline_mb, row["swap_used_mb"], tripwire.hold_s)
                    print(f"wrote {sentinel_path}", flush=True)

            _sleep_until(elapsed + interval, lambda: stopping, monotonic_clock)

    print("stopped", flush=True)
    return 0


def _sleep_until(
    deadline: float, stopping: Callable[[], bool], monotonic_clock: Callable[[], float] = time.monotonic
) -> None:
    """Nap in short slices so a signal is acted on promptly, not `interval` later."""
    while not stopping():
        remaining = deadline - monotonic_clock()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 1.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path, help="Harbor run directory to write metrics into")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between samples (default: 30)")
    parser.add_argument("--status-url", default=STATUS_URL, help=f"omlx status endpoint (default: {STATUS_URL})")
    args = parser.parse_args(argv)
    # Zero or negative leaves _sleep_until with nothing to wait for, turning the
    # loop into an unthrottled hammering of /api/status and the subprocesses.
    if args.interval <= 0:
        parser.error("--interval must be a positive number of seconds")
    return poll(args.run_dir, args.interval, status_url=args.status_url)


if __name__ == "__main__":
    sys.exit(main())
