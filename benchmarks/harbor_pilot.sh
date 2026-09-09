#!/usr/bin/env bash
# Pilot runner for Terminal-Bench 2.1 via harbor.
#
# Usage:
#   benchmarks/harbor_pilot.sh hello-world
#   benchmarks/harbor_pilot.sh task-a task-b
#
# Runbook note (Plan 5 / stale-code incident, 2026-09-08 22:26): a running
# Harbor job never reloads code -- it imports LittleCoderAgent once at job
# start and keeps running that frozen module for the job's entire lifetime,
# however many hours that is. After pulling a fix on this branch (or any
# branch a job is running from), the fix does NOT take effect until you kill
# the running job and relaunch this script; there is no live-reload path
# (see little_coder_agent.py's module docstring / Plan 5's "Rejected
# alternatives"). The banner this script prints below (code: <sha>) and each
# trial's own logged code_sha (little_coder_agent.py, at trial start) are the
# retroactive cross-check: compare either against `result.json`'s
# started_at timestamp and the commit history to tell whether a given run
# actually used the code you think it did.
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
#                             an increase-or-equal for every task, not an
#                             arbitrary bump. Host has 10 physical CPUs and
#                             --n-concurrent 1 means only one container runs
#                             at a time, so this doesn't compete with itself.
#   TB_OVERRIDE_MEMORY_MB   — OPT-IN per-task memory override, unset by
#                             default (Plan 5 / Codex finding [high]: this
#                             used to default to 4096 and was ALWAYS applied,
#                             but Harbor's --override-memory-mb REPLACES a
#                             task's own request rather than raising a floor
#                             -- verified via `harbor run --help`, no
#                             per-task max()/floor mode exists. Verified
#                             against the TB2.1 package cache: 68/89 tasks
#                             request 2048MB, 13 request 4096MB, and 8
#                             request 8192MB (including mteb-leaderboard,
#                             gpt2-codegolf, caffe-cifar-10). Forcing 4096
#                             unconditionally therefore LOWERED those 8 tasks'
#                             memory ceiling -- under --n-concurrent 1 (one
#                             container at a time) a single container can
#                             actually get most of Docker Desktop's ~7.75GB
#                             VM, so the old comment claiming "4096 isn't a
#                             regression for them" was wrong; it assumed
#                             concurrent-container contention that
#                             --n-concurrent 1 rules out. Set this env var
#                             only for a deliberate, informed override -- it
#                             will still replace, not floor, every task's own
#                             request, including lowering the 8 that ask for
#                             more than whatever you set.
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
# No default -- see TB_OVERRIDE_MEMORY_MB above. Empty means "don't pass
# --override-memory-mb at all", not "pass a memory override of empty".
OVERRIDE_MEMORY_MB="${TB_OVERRIDE_MEMORY_MB:-}"
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

# Launch-time code provenance (Plan 5 / Codex finding [medium], stale-code
# incident): a job launched right now runs whatever this worktree's HEAD is
# at this instant, frozen for the job's whole lifetime -- print it so a
# retroactive look at a job's output can be matched against commit history.
CODE_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
CODE_DIRTY=""
if git -C "$REPO_ROOT" diff --quiet --ignore-submodules HEAD 2>/dev/null; then
  :
else
  CODE_DIRTY="-dirty"
fi

echo "model:   $MODEL"
echo "dataset: $DATASET"
echo "tasks:   $*"
echo "output:  $OUT"
echo "code:    ${CODE_SHA}${CODE_DIRTY}"
if [[ -n "$CODE_DIRTY" ]]; then
  echo "WARNING: worktree has uncommitted changes -- this job's code does NOT" >&2
  echo "         match any commit; the printed sha is the nearest ancestor," >&2
  echo "         not what actually ran." >&2
fi
echo

HB_CMD=(harbor run
  --dataset "$DATASET"
  "${TASK_FLAGS[@]}"
  --agent benchmarks.harbor_adapter.little_coder_agent:LittleCoderAgent
  --model "$MODEL"
  --jobs-dir "$OUT"
  --n-concurrent 1
  --timeout-multiplier "$TIMEOUT_MULTIPLIER"
  --override-cpus "$OVERRIDE_CPUS")
if [[ -n "$OVERRIDE_MEMORY_MB" ]]; then
  HB_CMD+=(--override-memory-mb "$OVERRIDE_MEMORY_MB")
fi
HB_CMD+=(
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
