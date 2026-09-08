#!/usr/bin/env bash
# Pilot runner for Terminal-Bench 2.0 via harbor.
#
# Usage:
#   benchmarks/harbor_pilot.sh hello-world
#   benchmarks/harbor_pilot.sh task-a task-b
#
# Env:
#   TB_LITTLE_CODER_MODEL  — model override (default: llamacpp/qwen3.6-35b-a3b)
#
# Requires:
#   - harbor installed (uv tool install harbor)
#   - docker access (user in docker group or sg-wrapped below)
#   - llama.cpp or Ollama running at the extension-registered baseUrl
#
# Output:
#   benchmarks/harbor_runs/<timestamp>/...
set -euo pipefail

MODEL="${TB_LITTLE_CODER_MODEL:-llamacpp/qwen3.6-35b-a3b}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/benchmarks/harbor_runs"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <task-id> [<task-id> ...]" >&2
  exit 1
fi

TASK_FLAGS=()
for t in "$@"; do
  TASK_FLAGS+=(--include-task-name "$t")
done

export TB_LITTLE_CODER_MODEL="$MODEL"
export LLAMACPP_API_KEY="${LLAMACPP_API_KEY:-noop}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-noop}"

echo "model:   $MODEL"
echo "dataset: terminal-bench@2.0"
echo "tasks:   $*"
echo "output:  $OUT"
echo

HB_CMD=(harbor run
  --dataset terminal-bench@2.0
  "${TASK_FLAGS[@]}"
  --agent-import-path benchmarks.harbor_adapter.little_coder_agent:LittleCoderAgent
  --model "$MODEL"
  --jobs-dir "$OUT"
  --n-concurrent 1
  # TB2.0 tasks pin an amd64-only prebuilt image per task; on arm64 Docker
  # hosts this silently falls back to Rosetta/QEMU emulation (confirmed:
  # overfull-hbox's perl/pdflatex search loop ran under rosetta, contributing
  # to it hitting its deadline). --force-build makes harbor build each task's
  # own environment/Dockerfile locally instead of pulling the pinned image,
  # targeting the local Docker daemon's native platform -- content-addressed
  # and cached, so it's a one-time build per task, not per-trial.
  --force-build)

if groups | grep -q '\bdocker\b'; then
  cd "$REPO_ROOT" && "${HB_CMD[@]}"
else
  printf -v CMD_STR '%q ' "${HB_CMD[@]}"
  sg docker -c "cd '$REPO_ROOT' && $CMD_STR"
fi
