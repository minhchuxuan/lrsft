#!/usr/bin/env bash
# Run the experiment unattended. Re-running the same command resumes where it stopped.
#   ./run_all.sh                                   # stages A,B,extras with configs/default.yaml
#   STAGES=A ./run_all.sh                          # only part A (layer ranking, SFT, diagnostics)
#   CONFIGS="configs/default.yaml configs/fast.yaml" ./run_all.sh
#   ./run_all.sh --set rl.rounds=40 --dry-run      # extra args go to lrsft.pipeline
# In the background:  tmux new -d -s lrsft ./run_all.sh     or     nohup ./run_all.sh >/dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

export HF_HOME=${HF_HOME:-$PWD/.hf_cache}      # model + dataset cache
# compile caches (vLLM / torch.compile, Triton kernels, Inductor, matplotlib) stay inside the repo
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PWD/.cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$XDG_CACHE_HOME/flashinfer_base}  # ignores XDG
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# export LRSFT_WORK=/big/disk/lrsft            # optional: put models/checkpoints/rollouts elsewhere (~45 GB)

# shellcheck disable=SC1091
source "${VENV:-.venv}/bin/activate"
mkdir -p logs
LOG=logs/run_all_$(date +%Y%m%d_%H%M%S).log

CONFIG_ARGS=()
for c in ${CONFIGS:-configs/default.yaml}; do CONFIG_ARGS+=(--config "$c"); done

{
  echo "== $(date '+%F %T') run_all: configs=${CONFIGS:-configs/default.yaml} stages=${STAGES:-A,B,extras} args=$*"
  nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv 2>/dev/null || true
  python -m lrsft.pipeline "${CONFIG_ARGS[@]}" --stages "${STAGES:-A,B,extras}" "$@"
  echo "== $(date '+%F %T') run_all finished (exit $?)"
} 2>&1 | tee -a "$LOG"
