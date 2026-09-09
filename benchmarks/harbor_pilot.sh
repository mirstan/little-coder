#!/usr/bin/env bash
# Pilot runner for Terminal-Bench 2.1 via harbor.
#
# Usage:
#   benchmarks/harbor_pilot.sh hello-world
#   benchmarks/harbor_pilot.sh task-a task-b
#
# Env:
#   TB_LITTLE_CODER_MODEL   — model override (default: llamacpp/qwen3.6-35b-a3b)
#   TB_DATASET              — dataset override (default:
#                             terminal-bench/terminal-bench-2-1, resolved via
#                             harbor's newer org/name package registry, NOT
#                             the legacy name@version one -- confirmed 2.1
#                             fixes configure-git-webserver's SSH username/
#                             auth-method mismatch against the instructions,
#                             which 2.0 never did)
#   TB_TIMEOUT_MULTIPLIER   — per-task timeout multiplier (default: 15, raised
#                             5x from the original 3 -- Qwen's own published
#                             Terminal-Bench 2.0 methodology uses a flat 3h
#                             per-task timeout; at task.toml's typical
#                             ~750s base, 15x lands close to that, vs. the
#                             prior 3x which gave the model only ~20-35% of
#                             the vendor's reference wall-clock budget)
#   TB_OVERRIDE_CPUS        — per-task CPU override (default: 4). Confirmed via
#                             cgroup cpu.stat on a live run that TB2.1's own
#                             per-task default (1 CPU for 83/89 tasks) causes
#                             real throttling (39% of scheduling periods on
#                             llm-inference-batching-scheduler); 4 matches the
#                             highest CPU count any task.toml itself requests
#                             (only 3/89 tasks ask for more than 1), so it's
#                             not an arbitrary bump. Host has 10 physical
#                             CPUs and --n-concurrent 1 means only one
#                             container runs at a time, so this doesn't
#                             compete with itself.
#   TB_OVERRIDE_MEMORY_MB   — per-task memory override (default: 4096). Docker
#                             Desktop's VM has only ~7.75GB total (`docker
#                             info`) -- below what the 8 tasks requesting
#                             8192MB in their own task.toml could ever
#                             actually get anyway, so 4096 isn't a regression
#                             for them, while it's a real 2x increase for the
#                             68/89 tasks that only ask for 2048MB.
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
DATASET="${TB_DATASET:-terminal-bench/terminal-bench-2-1}"
TIMEOUT_MULTIPLIER="${TB_TIMEOUT_MULTIPLIER:-15}"
OVERRIDE_CPUS="${TB_OVERRIDE_CPUS:-4}"
OVERRIDE_MEMORY_MB="${TB_OVERRIDE_MEMORY_MB:-4096}"
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
# --agent-import-path is deprecated and fails with "ModuleNotFoundError: No
# module named 'benchmarks'" under harbor's isolated uv-tool Python env,
# which doesn't inherit this repo's cwd on sys.path automatically -- use
# --agent plus an explicit PYTHONPATH instead (found working this session).
export PYTHONPATH="$REPO_ROOT"

echo "model:   $MODEL"
echo "dataset: $DATASET"
echo "tasks:   $*"
echo "output:  $OUT"
echo

HB_CMD=(harbor run
  --dataset "$DATASET"
  "${TASK_FLAGS[@]}"
  --agent benchmarks.harbor_adapter.little_coder_agent:LittleCoderAgent
  --model "$MODEL"
  --jobs-dir "$OUT"
  --n-concurrent 1
  --timeout-multiplier "$TIMEOUT_MULTIPLIER"
  --override-cpus "$OVERRIDE_CPUS"
  --override-memory-mb "$OVERRIDE_MEMORY_MB"
  -y
  # TB2.0 tasks pin an amd64-only prebuilt image per task; on arm64 Docker
  # hosts this silently falls back to Rosetta/QEMU emulation (confirmed:
  # overfull-hbox's perl/pdflatex search loop ran under rosetta, contributing
  # to it hitting its deadline). --force-build makes harbor build each task's
  # own environment/Dockerfile locally instead of pulling the pinned image,
  # targeting the local Docker daemon's native platform -- content-addressed
  # and cached, so it's a one-time build per task, not per-trial.
  --force-build)

# `sg` (group-switch) is a Linux-only workaround for a user who isn't in the
# `docker` group; it doesn't exist on macOS, and Docker Desktop grants socket
# access without that unix-group mechanism at all, so `groups` never contains
# "docker" there either -- unconditionally falling into the sg branch on
# macOS failed outright with "sg: command not found". Only use it where it
# can actually exist.
if ! command -v sg >/dev/null 2>&1 || groups | grep -q '\bdocker\b'; then
  cd "$REPO_ROOT" && "${HB_CMD[@]}"
else
  # %q for the path too: hand-quoting it as '$REPO_ROOT' breaks (and would
  # let the path inject shell) if any parent directory contains a quote.
  printf -v CMD_STR '%q ' "${HB_CMD[@]}"
  printf -v ROOT_Q '%q' "$REPO_ROOT"
  sg docker -c "cd $ROOT_Q && $CMD_STR"
fi
