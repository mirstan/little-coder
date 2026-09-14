#!/usr/bin/env python3
"""Syntax-error frequency across Harbor trials -- a before/after metric for
the syntax-check extension proposed in the harness-improvement plan.

Counts real interpreter/compiler syntax-error occurrences in each trial's
own little_coder.live.log (the model's own tool-call/tool-result trace, not
the omlx server log -- this measures the harness/model side, not the model
server). Only scans each tool result's own OUTPUT text (the `<< ...` portion
up to its `[exit=...]` footer), never the `>> Tool({'command': ...})` line
that precedes it -- the command can itself contain a heredoc'd script whose
source legitimately says `except SyntaxError:` or a syntax-check wrapper
like `perl -c f.pl && echo OK`, and counting those would move the metric
backwards for a run whose agent got better at self-checking, not worse.
Truncated (`... [+N chars truncated]`) or deduplicated (`[... N duplicate
line(s) collapsed ...]`) tool output means an error past that point, or a
repeated one, can be under-counted -- this is a real trajectory-log limit,
not fixable here, so treat this as a lower bound, not an exact count.

Point it at one or more run directories (each holding many trial
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

# Real tool output starts with "<< " on its own line and, for shell tools,
# ends at a "[exit=N cwd=... timed_out=... backend=...]" footer line -- DOTALL
# so "." spans the (often many) output lines in between, MULTILINE so "^"/"$"
# anchor to each line rather than the whole file. Not every tool produces
# that footer: across the real corpus, non-shell tools ("write", "webfetch",
# "websearch") routinely appear with no footer at all -- in every real trial
# checked, `>> ` calls consistently outnumber `[exit=` footers. A naive
# non-greedy `(.*?)` would keep expanding right past one of those footerless
# results looking for the NEXT footer anywhere later in the file, swallowing
# an intervening ShellSession call's own command text (and any further
# results before it) as if all of it were the first result's output -- the
# exact command-text-scanning bug this file exists to avoid, reintroduced by
# a different mechanism. The `(?!^>> |^<< )` guard on every consumed
# character stops the match dead at the next call/result marker, so a
# footerless result (no footer before the next marker) simply produces no
# match at all instead of over-consuming.
_RESULT_RE = re.compile(r"^<< ((?:(?!^>> |^<< ).)*?)^\[exit=.*?\]$", re.MULTILINE | re.DOTALL)

# Harbor/TB2.1 trials call the shell tool "ShellSession"; GAIA trials (see
# benchmarks/gaia.py's ALLOWED_TOOLS) call it "bash"/"Bash" instead -- match
# both so the denominator isn't silently 0 for a GAIA-shaped log
# (is_trial_dir/find_trial_dirs don't restrict to Harbor-only directories, so
# nothing stops this script from being pointed at one).
_SHELL_CALL_RE = re.compile(r">> (?:ShellSession|[Bb]ash)\(")

# Checked against all 27 real trial logs available at the time these were
# picked (benchmarks/harbor_runs/*/*/agent/little_coder.live.log). gcc_errors
# is intentionally NOT anchored to line-start: real tool output can put the
# diagnostic on the very first output line (right after the "<< " prefix),
# which a `^`-anchored pattern would silently miss -- the ":line:col: error:"
# shape alone is what excludes the one real false positive found in this
# corpus, "env.exec error: Command timed out after Ns" (no digits either side
# of a ':'), so the anchor was never doing that job. Filename charset
# includes '-' (hyphenated C filenames are common) and a few C++ extensions.
PATTERNS = {
    "python_errors": re.compile(r"SyntaxError|IndentationError"),
    "perl_errors": re.compile(r'syntax error at .* line \d+|Global symbol "[^"]*" requires'),
    "gcc_errors": re.compile(r"[-\w./]+\.(?:[ch]|cc|cpp|hpp|cxx):\d+:\d+: error: "),
    # GnuCOBOL's shape omits the column number gcc has: "file:line: error: msg",
    # not "file:line:col: error:". Confirmed against a real trial
    # (cobol-modernization__ZadiHUd): "src/program.cbl:3: error: syntax error,
    # unexpected ENVIRONMENT, expecting FUNCTION-ID or PROGRAM-ID" -- a
    # genuine syntax error (missing PROGRAM-ID.) that the gcc pattern alone
    # would never match.
    "cobol_errors": re.compile(r"[-\w./]+\.cbl:\d+: error: "),
    "bash_errors": re.compile(r"syntax error near unexpected token|unexpected EOF while looking for"),
}

CSV_COLUMNS = [
    "run_label",
    "trial",
    "shell_calls",
    *PATTERNS,
    "total_errors",
    "calls_with_errors",
    "error_rate_pct",
]


def is_trial_dir(path: Path) -> bool:
    return (path / "agent" / "little_coder.live.log").exists()


def has_agent_dir_but_no_log(path: Path) -> bool:
    return (path / "agent").is_dir() and not is_trial_dir(path)


def find_trial_dirs(path: Path) -> tuple[list[Path], list[Path]]:
    """(trial dirs, skipped dirs that look like a trial but have no log yet).

    A crashed-before-first-write trial and a directory that was never a
    trial both lack agent/little_coder.live.log -- reporting the second list
    lets a before/after comparison notice a different crash rate between two
    runs instead of the two runs silently having different denominators.
    """
    if is_trial_dir(path):
        return [path], []
    if has_agent_dir_but_no_log(path):
        # `path` itself is a trial dir, just not ready yet -- e.g. passed
        # directly (not via its run-dir parent) before the trial has written
        # its first line, or after it crashed during setup. Without this
        # check it would fall through to iterdir() and get scanned as if it
        # were a RUN directory containing sibling trials, none of which
        # `path`'s own children (agent/, result.json, ...) actually are --
        # silently returning ([], []) and dropping this trial from both
        # lists instead of reporting it as skipped.
        return [], [path]
    candidates = sorted(p for p in path.iterdir() if p.is_dir())
    trials = [p for p in candidates if is_trial_dir(p)]
    skipped = [p for p in candidates if has_agent_dir_but_no_log(p)]
    return trials, skipped


def result_text(text: str) -> str:
    """Every tool result's own output, command text and narration excluded."""
    return "\n".join(_RESULT_RE.findall(text))


def count_errors(text: str) -> tuple[dict[str, int], int]:
    """Per-pattern occurrence counts and the count of *results* touched by >=1.

    `total_errors` is occurrence-weighted (Perl's "Global symbol" fires once
    per bad variable, so one broken script can contribute many), which is
    useful for "how bad is this" but not for "how often does this happen" --
    `calls_with_errors` answers the second question and is the more
    robust before/after signal when the language mix (and so per-error
    verbosity) can differ between two runs.
    """
    results = _RESULT_RE.findall(text)
    counts = {name: 0 for name in PATTERNS}
    calls_with_errors = 0
    for result in results:
        hit = False
        for name, pattern in PATTERNS.items():
            n = len(pattern.findall(result))
            counts[name] += n
            hit = hit or n > 0
        if hit:
            calls_with_errors += 1
    return counts, calls_with_errors


def trial_row(trial_dir: Path, run_label: str) -> dict:
    log = trial_dir / "agent" / "little_coder.live.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    shell_calls = len(_SHELL_CALL_RE.findall(text))
    errors, calls_with_errors = count_errors(text)
    total = sum(errors.values())
    row = {
        "run_label": run_label,
        "trial": trial_dir.name,
        "shell_calls": shell_calls,
        "total_errors": total,
        "calls_with_errors": calls_with_errors,
        "error_rate_pct": round(calls_with_errors / shell_calls * 100, 2) if shell_calls else 0.0,
    }
    row.update(errors)
    return row


def collect_rows(paths: list[Path]) -> tuple[list[dict], list[Path]]:
    rows = []
    all_skipped: list[Path] = []
    for path in paths:
        # The full resolved path, not just its basename -- two arguments that
        # share a basename (before/job-7 and after/job-7, the exact shape a
        # before/after comparison invites) would otherwise collide into one
        # summary group and one indistinguishable set of CSV rows. Also fixes
        # "." resolving to an empty label.
        run_label = str(path.resolve())
        trial_dirs, skipped = find_trial_dirs(path)
        all_skipped.extend(skipped)
        for trial_dir in trial_dirs:
            rows.append(trial_row(trial_dir, run_label))
    return rows, all_skipped


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict], skipped: list[Path]) -> None:
    if not rows:
        print("no trial directories found (wrong path, or no little_coder.live.log yet)")
    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run_label"], []).append(row)

    for run_label, run_rows in by_run.items():
        total_calls = sum(r["shell_calls"] for r in run_rows)
        total_errors = sum(r["total_errors"] for r in run_rows)
        total_calls_with_errors = sum(r["calls_with_errors"] for r in run_rows)
        affected = sum(1 for r in run_rows if r["total_errors"] > 0)
        rate = round(total_calls_with_errors / total_calls * 100, 2) if total_calls else 0.0
        print(f"=== {run_label} ===")
        print(f"trials: {len(run_rows)}  shell_calls: {total_calls}  syntax_errors: {total_errors}")
        print(f"trials with >=1 syntax error: {affected}/{len(run_rows)}  calls-with-error rate: {rate}%")
        worst = sorted(run_rows, key=lambda r: r["total_errors"], reverse=True)[:3]
        for r in worst:
            if r["total_errors"] > 0:
                print(f"  {r['trial']}: {r['total_errors']} errors in {r['shell_calls']} calls ({r['error_rate_pct']}%)")
        print()

    if skipped:
        print(f"skipped {len(skipped)} trial-shaped dir(s) with no live log yet (crashed early, or still running):")
        for path in skipped:
            print(f"  {path}")


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

    rows, skipped = collect_rows(args.paths)
    if rows:
        write_csv(args.out, rows)
        print(f"wrote {args.out} ({len(rows)} trials)\n")
    else:
        # Don't clobber a previous good report at the same --out path with an
        # empty one just because this invocation found nothing (wrong path
        # depth is an easy mistake -- trials live two levels under a run dir).
        print(f"nothing written to {args.out} -- found 0 trials\n")
    print_summary(rows, skipped)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
