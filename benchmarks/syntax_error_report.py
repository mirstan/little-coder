#!/usr/bin/env python3
"""Syntax-error frequency across Harbor trials -- a before/after metric for
the syntax-check extension proposed in the harness-improvement plan.

Counts real interpreter/compiler syntax-error occurrences in each trial's
own little_coder.live.log (the model's own tool-call/tool-result trace, not
the omlx server log -- this measures the harness/model side, not the model
server). Point it at one or more run directories (each holding many trial
subdirectories) or trial directories directly; it tells the two apart by
whether agent/little_coder.live.log exists one level down or two:

    benchmarks/syntax_error_report.py benchmarks/harbor_runs/<job> [<job2> ...]
    benchmarks/syntax_error_report.py benchmarks/harbor_runs/<job>/<task>__<id>

Writes syntax_errors.csv into the current directory (or --out) with one row
per trial, tagged by which argument it came from -- pass a pre-extension run
directory and a post-extension one in the same invocation and the CSV is
already a matched before/after table.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

CSV_FILENAME = "syntax_errors.csv"
CSV_COLUMNS = [
    "run_label",
    "trial",
    "shell_calls",
    "python_errors",
    "perl_errors",
    "gcc_errors",
    "bash_errors",
    "total_errors",
    "error_rate_pct",
]

# Every pattern below was checked against real trajectories before being
# picked -- in particular gcc_error is anchored to the compiler's own
# "file:line:col: error: " shape (MULTILINE, so `^` matches every line, not
# just the first) specifically to exclude "env.exec error: Command timed out
# after Ns", a real false positive an unanchored `error: ` search picks up.
_SHELL_CALL_RE = re.compile(r">> ShellSession\(")
PATTERNS = {
    "python_errors": re.compile(r"SyntaxError|IndentationError"),
    "perl_errors": re.compile(r'syntax error at .* line \d+|Global symbol "[^"]*" requires'),
    "gcc_errors": re.compile(r"^[\w./]+\.[ch]:\d+:\d+: error: ", re.MULTILINE),
    "bash_errors": re.compile(r"syntax error near unexpected token|unexpected EOF while looking for"),
}


def is_trial_dir(path: Path) -> bool:
    return (path / "agent" / "little_coder.live.log").exists()


def find_trial_dirs(path: Path) -> list[Path]:
    """`path` itself if it's a trial dir, else its trial-shaped subdirectories."""
    if is_trial_dir(path):
        return [path]
    return sorted(p for p in path.iterdir() if p.is_dir() and is_trial_dir(p))


def count_errors(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()}


def trial_row(trial_dir: Path, run_label: str) -> dict:
    log = trial_dir / "agent" / "little_coder.live.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    shell_calls = len(_SHELL_CALL_RE.findall(text))
    errors = count_errors(text)
    total = sum(errors.values())
    row = {
        "run_label": run_label,
        "trial": trial_dir.name,
        "shell_calls": shell_calls,
        "total_errors": total,
        "error_rate_pct": round(total / shell_calls * 100, 2) if shell_calls else 0.0,
    }
    row.update(errors)
    return row


def collect_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        run_label = path.name
        for trial_dir in find_trial_dirs(path):
            rows.append(trial_row(trial_dir, run_label))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("no trial directories found (wrong path, or no little_coder.live.log yet)")
        return
    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run_label"], []).append(row)

    for run_label, run_rows in by_run.items():
        total_calls = sum(r["shell_calls"] for r in run_rows)
        total_errors = sum(r["total_errors"] for r in run_rows)
        affected = sum(1 for r in run_rows if r["total_errors"] > 0)
        rate = round(total_errors / total_calls * 100, 2) if total_calls else 0.0
        print(f"=== {run_label} ===")
        print(f"trials: {len(run_rows)}  shell_calls: {total_calls}  syntax_errors: {total_errors}")
        print(f"trials with >=1 syntax error: {affected}/{len(run_rows)}  overall rate: {rate}%")
        worst = sorted(run_rows, key=lambda r: r["total_errors"], reverse=True)[:3]
        for r in worst:
            if r["total_errors"] > 0:
                print(f"  {r['trial']}: {r['total_errors']} errors in {r['shell_calls']} calls ({r['error_rate_pct']}%)")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="+", type=Path, help="one or more Harbor run or trial directories"
    )
    parser.add_argument("--out", type=Path, default=Path(CSV_FILENAME), help=f"CSV output path (default: {CSV_FILENAME})")
    args = parser.parse_args(argv)

    for path in args.paths:
        if not path.is_dir():
            parser.error(f"not a directory: {path}")

    rows = collect_rows(args.paths)
    write_csv(args.out, rows)
    print(f"wrote {args.out} ({len(rows)} trials)\n")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
