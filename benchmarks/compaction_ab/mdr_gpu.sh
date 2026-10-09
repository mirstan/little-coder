#!/bin/bash
# Multi-depth replay on omlx: the GPU steps, in the foreground, with preflight and lock.
#   benchmarks/compaction_ab/mdr_gpu.sh STAGE_DIR [compaction|probe|all]
# STAGE_DIR holds points/, template.json and source/ (from mdr_capture.py, mdr_build.mjs).
# It refuses to start unless no harbor run is active, omlx is idle, and no other lock holder exists.
set -euo pipefail
D=$1; WHAT=${2:-all}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOCK=/private/tmp/claude-501/-Users-stanmiroshnikov/6316e8ea-50c4-4839-8de2-3613b65eef65/scratchpad/omlx-gpu.lock
pgrep -f "harbor run" >/dev/null && { echo "harbor run active: refusing"; exit 2; }
act=$(curl -s -m 5 http://127.0.0.1:8000/api/status | python3 -c "import json,sys;print(json.load(sys.stdin)['active_requests'])")
[ "$act" = "0" ] || { echo "omlx busy (active_requests=$act): refusing"; exit 2; }
[ -e "$LOCK" ] && { echo "lock held: $(cat "$LOCK")"; exit 2; }
echo "compaction-ab mdr $WHAT $(date -u +%FT%TZ)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
S=$(ls "$D"/source/agent/pi-session/2026*.jsonl)
if [ "$WHAT" = compaction ] || [ "$WHAT" = all ]; then
  perl -e 'alarm shift; exec @ARGV' 14400 node "$ROOT/benchmarks/compaction_ab/mdr_run.mjs" --points "$D/points" --out "$D/run.jsonl" --k 2
fi
if [ "$WHAT" = probe ] || [ "$WHAT" = all ]; then
  perl -e 'alarm shift; exec @ARGV' 14400 node "$ROOT/benchmarks/compaction_ab/mdr_quality.mjs" --points "$D/points" --run "$D/run.jsonl" \
    --session "$S" --turns "$D/source/agent/turns.jsonl" --template "$D/template.json" --probe "$D/probe.jsonl" --k 3 --packets "$D/packets"
fi
