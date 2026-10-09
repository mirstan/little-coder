#!/bin/bash
# Compaction A/B live driver (plan §2.6, §5). One harbor job per trial, in the
# pre-registered order from schedule.json. Launch it with nohup from the main
# session, never from a fork:
#   nohup benchmarks/compaction_ab/driver.sh SCHEDULE.json OUTDIR FROZEN_SHA [first_index] > OUTDIR/driver.log 2>&1 &
# DRY_RUN=1 prints each trial's env and command and launches nothing.
set -u
SCHED=$1; OUT=$2; SHA=$3; START=${4:-0}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=$ROOT/benchmarks/harbor_runs
LOCK=/private/tmp/claude-501/-Users-stanmiroshnikov/6316e8ea-50c4-4839-8de2-3613b65eef65/scratchpad/omlx-gpu.lock
mkdir -p "$OUT"
n=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$SCHED")
log() { echo "$(date '+%F %T') $*"; }

for ((i=START; i<n; i++)); do
  read -r TASK ARM <<<"$(python3 -c "import json,sys;r=json.load(open(sys.argv[1]))[int(sys.argv[2])];print(r['task'],r['arm'])" "$SCHED" "$i")"
  [ -f "$OUT/STOP" ] && { log "STOP file present; halting before trial $i"; exit 0; }
  # Frozen code: base 26773a3 is an ancestor and HEAD is the frozen SHA.
  [ "$(git -C "$ROOT" rev-parse HEAD)" = "$SHA" ] || { log "HEAD != frozen $SHA"; exit 2; }
  git -C "$ROOT" merge-base --is-ancestor 26773a3 HEAD || { log "base 26773a3 not an ancestor"; exit 2; }
  [ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] || { log "worktree dirty"; exit 2; }
  (cd "$ROOT/vendor/compaction-ab" && shasum -a 256 -c SHA256SUMS >/dev/null) || { log "vendored checksum mismatch"; exit 2; }
  pgrep -f "harbor run" >/dev/null && { log "another harbor run is active"; exit 2; }
  act=$(curl -s -m 5 http://127.0.0.1:8000/api/status | python3 -c "import json,sys;print(json.load(sys.stdin)['active_requests'])" 2>/dev/null)
  [ "$act" = "0" ] || { log "omlx not idle (active_requests=$act)"; exit 2; }

  ENVV=(env -u LC_COMPACTION_ARM -u LITTLE_CODER_CACHE_REUSE_COMPACTION)
  for v in $(env | sed -n 's/^\(PI_BLACKHOLE_[A-Z_]*\)=.*/\1/p'); do ENVV+=(-u "$v"); done
  ENVV+=(LITTLE_CODER_PI_SESSION_DIR_IN_LOGS=1 LITTLE_CODER_AB_OBSERVER=1)
  BH_AGENT=$ROOT/.cache/pi-bench-agent/pi-blackhole
  case $ARM in
    native) ;;
    reuse84) ENVV+=(LITTLE_CODER_CACHE_REUSE_COMPACTION=1) ;;
    blackhole) ENVV+=(LC_COMPACTION_ARM=blackhole)
      mkdir -p "$BH_AGENT" && cp "$ROOT/vendor/compaction-ab/config/pi-blackhole-config.json" "$BH_AGENT/pi-blackhole-config.json"
      cmp -s "$ROOT/vendor/compaction-ab/config/pi-blackhole-config.json" "$BH_AGENT/pi-blackhole-config.json" || { log "blackhole config copy failed"; exit 2; } ;;
    prefix-cache) ENVV+=(LC_COMPACTION_ARM=prefix-cache) ;;
    *) log "unknown arm $ARM"; exit 2 ;;
  esac
  CMD=("${ENVV[@]}" "$ROOT/benchmarks/harbor_pilot.sh" "terminal-bench/$TASK")
  log "trial $i: task=$TASK arm=$ARM"
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '  %q' "${CMD[@]}"; echo; continue; fi

  echo "compaction-ab live trial $i $TASK $ARM $(date -u +%FT%TZ)" > "$LOCK"
  before=$(ls "$RUNS")
  off=$(stat -f %z /opt/homebrew/var/log/omlx.log)
  "${CMD[@]}" > "$OUT/trial-$i.harbor.log" 2>&1 &
  hpid=$!
  job=""
  for _ in $(seq 1 120); do job=$(comm -13 <(echo "$before") <(ls "$RUNS") | tail -1); [ -n "$job" ] && break; sleep 5; done
  if [ -n "$job" ]; then
    python3 "$ROOT/benchmarks/run_watch.py" "$RUNS/$job" --server-log /opt/homebrew/var/log/omlx.log >> "$OUT/run_watch.log" 2>&1 &
    wpid=$!
  fi
  wait $hpid; rc=$?
  [ -n "${wpid:-}" ] && kill "$wpid" 2>/dev/null
  trial=$(ls -d "$RUNS/$job"/"$TASK"__* 2>/dev/null | head -1)
  log "trial $i harbor exit=$rc job=$job trial=$trial"
  python3 "$ROOT/benchmarks/compaction_ab/trial_check.py" "$trial" "$ARM" "$SHA" "$off" "$OUT/ledger.jsonl"
  log "trial $i check exit=$?"
  rm -f "$LOCK"
done
log "schedule complete"
