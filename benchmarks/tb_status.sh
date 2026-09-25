#!/usr/bin/env bash
# One-liner status for an in-flight Terminal-Bench run.
#
# Usage:
#   benchmarks/tb_status.sh                             # auto-detect newest leaderboard-* dir
#   benchmarks/tb_status.sh leaderboard-2026-04-22__22-50-08
#   RUN_ID=foo benchmarks/tb_status.sh
#
# Prints: process health, docker in-flight, completed/remaining, accuracy,
# per-minute rate, ETA, pass/fail list, and (v0.1.4+) extension-activity
# metrics — tool-call averages, turn-cap hits, skill/knowledge-injection
# counts, thinking-budget fires, quality-monitor corrections.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TB_RUNS="$REPO_ROOT/benchmarks/tb_runs"

RUN_ID="${1:-${RUN_ID:-}}"
if [ -z "$RUN_ID" ]; then
  # Portable, no find/printf: -regextype and -printf are GNU-only. Names are
  # ISO-date-prefixed (leaderboard-YYYY-MM-DD...), so lexicographic string
  # comparison is correct here -- no mtime lookup needed.
  NEWEST=""
  for d in "$TB_RUNS"/leaderboard-[0-9]* "$TB_RUNS"/full-[0-9]*; do
    [ -d "$d" ] || continue
    base="${d##*/}"
    if [ -z "$NEWEST" ] || [[ "$base" > "$NEWEST" ]]; then
      NEWEST="$base"
    fi
  done
  RUN_ID="$NEWEST"
fi
if [ -z "$RUN_ID" ] || [ ! -d "$TB_RUNS/$RUN_ID" ]; then
  echo "No run dir found (looked in $TB_RUNS)." >&2
  exit 1
fi

DIR="$TB_RUNS/$RUN_ID"
META="$DIR/run_metadata.json"
RES="$DIR/results.json"

python3 - "$RUN_ID" "$DIR" "$META" "$RES" <<'PY'
import json, os, sys, time, subprocess, re, statistics
run_id, dir_, meta_p, res_p = sys.argv[1:5]

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

def humanize(s):
    if s <= 0: return "-"
    h, rem = divmod(int(s), 3600); m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"

# ── Metadata ─────────────────────────────────────────────────────────────
total = 0; dataset = ""
if os.path.exists(meta_p):
    m = json.load(open(meta_p))
    total = len(m.get("task_ids", []))
    dataset = f"{m.get('dataset_name')}@{m.get('dataset_version')}"

# ── Results.json aggregate ───────────────────────────────────────────────
n_resolved = n_unresolved = 0
resolved = unresolved = []
if os.path.exists(res_p):
    r = json.load(open(res_p))
    n_resolved = r.get("n_resolved", 0)
    n_unresolved = r.get("n_unresolved", 0)
    resolved = r.get("resolved_ids", []) or []
    unresolved = r.get("unresolved_ids", []) or []
done = n_resolved + n_unresolved
remaining = max(total - done, 0)
acc = (n_resolved / done * 100.0) if done else 0.0

# ── Trial subdirs
trial_dirs = [d for d in os.listdir(dir_) if os.path.isdir(os.path.join(dir_, d))]

# ── Process
run_pid = sh(f"pgrep -af 'tb run.*{run_id}' | awk '{{print $1}}' | head -1")
elapsed = sh(f"ps -p {run_pid} -o etime= 2>/dev/null").strip() if run_pid else "(not running)"
elapsed_s = 0
if elapsed and ":" in elapsed:
    parts = elapsed.split("-"); days = int(parts[0]) if len(parts) == 2 else 0
    hms = parts[-1].split(":")
    if len(hms) == 3:
        elapsed_s = days*86400 + int(hms[0])*3600 + int(hms[1])*60 + int(hms[2])
    elif len(hms) == 2:
        elapsed_s = days*86400 + int(hms[0])*60 + int(hms[1])

in_flight = sh(f"sg docker -c \"docker ps --filter 'name={run_id}' --format '{{{{.Names}}}}  up {{{{.Status}}}}'\"")

rate_s_per_task = (elapsed_s / done) if (done and elapsed_s) else 0
eta_s = rate_s_per_task * remaining

# ── Walk every completed trial's adapter log to extract tool-call + notify
# metrics ────────────────────────────────────────────────────────────────
tool_calls_per_task = {}   # task -> int
tool_name_counts = {}      # tool -> int
notes_per_task = {}        # task -> [note strings]
turn_cap_env = 25          # TB default

for d in trial_dirs:
    log_glob = subprocess.check_output(
        f"find {dir_}/{d} -name 'little_coder*.log' 2>/dev/null | head -1",
        shell=True, text=True,
    ).strip()
    if not log_glob or not os.path.exists(log_glob):
        continue
    calls = 0
    names = []
    notes = []
    in_notes = False
    with open(log_glob, errors="replace") as f:
        for line in f:
            if line.startswith(">> "):
                calls += 1
                m = re.match(r">> (\w+)\(", line)
                if m:
                    names.append(m.group(1))
            elif "=== pi notifications" in line:
                in_notes = True
                continue
            elif line.startswith("=== "):
                in_notes = False
            elif in_notes and line.strip():
                notes.append(line.strip())
    tool_calls_per_task[d] = calls
    for n in names:
        tool_name_counts[n] = tool_name_counts.get(n, 0) + 1
    notes_per_task[d] = notes

# ── Aggregate metrics over COMPLETED tasks only
completed_tasks = set(resolved) | set(unresolved)
completed_calls = [tool_calls_per_task[t] for t in trial_dirs if t in completed_tasks and t in tool_calls_per_task]

def count_notes_matching(pattern):
    tasks_hit = 0; events = 0
    for t, ns in notes_per_task.items():
        if t not in completed_tasks: continue
        k = sum(1 for n in ns if pattern in n)
        if k > 0: tasks_hit += 1
        events += k
    return tasks_hit, events

skill_tasks, skill_events = count_notes_matching("skill-inject: +")
knowledge_tasks, knowledge_events = count_notes_matching("knowledge-inject: +")
thinking_tasks, thinking_events = count_notes_matching("thinking-budget:")
quality_tasks, quality_events = count_notes_matching("quality-monitor:")
turn_cap_tasks, _ = count_notes_matching("turn-cap:")
evidence_tasks, evidence_events = count_notes_matching("evidence-compact:")
parser_tasks, parser_events = count_notes_matching("text-embedded tool")

turn_cap_hits_by_calls = sum(1 for c in completed_calls if c >= turn_cap_env)

any_notes_captured = sum(len(v) for v in notes_per_task.values()) > 0

# ── Print
print(f"run_id     : {run_id}")
print(f"dataset    : {dataset}")
print(f"process    : pid={run_pid or 'DEAD'}  elapsed={elapsed}")
print(f"progress   : {done}/{total} done  ({remaining} remaining)  in-progress-dirs={max(len(trial_dirs) - done, 0)}")
print(f"accuracy   : {n_resolved}/{done} = {acc:.1f} %  {'⚠ early-sample' if done < 10 else ''}")
print(f"rate       : {humanize(rate_s_per_task)} / task  →  ETA +{humanize(eta_s)} remaining")
print(f"in-flight  : {in_flight or '(between tasks)'}")

if resolved:
    print(f"\npassed ({len(resolved)}):")
    for t in resolved[-10:]:
        print(f"  ✓ {t}")
    if len(resolved) > 10: print(f"  ... and {len(resolved) - 10} more above")
if unresolved:
    print(f"\nfailed ({len(unresolved)}):")
    for t in unresolved[-10:]:
        print(f"  ✗ {t}")
    if len(unresolved) > 10: print(f"  ... and {len(unresolved) - 10} more above")

# Current / last-touched trial dir
try:
    trial_paths = [(os.path.getmtime(os.path.join(dir_, d)), d) for d in trial_dirs]
    trial_paths.sort(reverse=True)
    if trial_paths and trial_paths[0][1] not in completed_tasks:
        age = int(time.time() - trial_paths[0][0])
        print(f"\ncurrent    : {trial_paths[0][1]} (last-modified {age}s ago)")
except Exception:
    pass

# ── Metrics section
print(f"\n── metrics ──")
if completed_calls:
    avg = statistics.mean(completed_calls)
    med = statistics.median(completed_calls)
    mn = min(completed_calls); mx = max(completed_calls)
    print(f"tool calls/task  : avg={avg:.1f}  median={med:.0f}  min={mn}  max={mx}")
    print(f"at turn-cap ({turn_cap_env})  : {turn_cap_hits_by_calls}/{len(completed_calls)} tasks hit the cap")
if tool_name_counts:
    breakdown = ", ".join(f"{n}={c}" for n, c in sorted(tool_name_counts.items(), key=lambda x: -x[1]))
    print(f"tool breakdown   : {breakdown}")

if any_notes_captured:
    print(f"skill-inject     : {skill_events} fires across {skill_tasks}/{len(completed_tasks)} tasks")
    print(f"knowledge-inject : {knowledge_events} fires across {knowledge_tasks}/{len(completed_tasks)} tasks")
    print(f"thinking-budget  : {thinking_events} fires across {thinking_tasks}/{len(completed_tasks)} tasks")
    print(f"quality-monitor  : {quality_events} corrections across {quality_tasks}/{len(completed_tasks)} tasks")
    print(f"output-parser    : {parser_events} text-tool rescues across {parser_tasks}/{len(completed_tasks)} tasks")
    print(f"evidence-compact : {evidence_events} bridges across {evidence_tasks}/{len(completed_tasks)} tasks")
    print(f"turn-cap aborts  : {turn_cap_tasks}/{len(completed_tasks)} tasks (from notify; may lag calls-based cap hits)")
else:
    print(f"extension events : N/A — run launched before v0.1.4 (notify capture)")

PY
