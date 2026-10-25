#!/usr/bin/env bash
# One-time environment setup on a Linux + NVIDIA machine (tested pins, see README "Stack").
#   ./setup.sh                      # picks the CUDA flavour from the driver
#   CUDA_FLAVOR=cu129 ./setup.sh    # force CUDA 12.9 wheels
#   VENV=~/venvs/lrsft ./setup.sh   # venv elsewhere (run_all.sh reads the same VENV)
# uv goes to $UV_INSTALL_DIR (default ~/.local/bin) and never edits shell rc files; UV_CACHE_DIR and
# UV_PYTHON_INSTALL_DIR are honoured, so everything can be kept inside one directory.
set -euo pipefail
cd "$(dirname "$0")"

VLLM_VERSION=0.31.0          # hard-pins torch==2.13.0
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
VENV=${VENV:-.venv}
UV_VERSION=0.12.21
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}   # torch + CUDA wheels are several GB

UV_DIR=${UV_INSTALL_DIR:-$HOME/.local/bin}
export PATH="$UV_DIR:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  # "unmanaged" install: just the binaries in $UV_DIR, no shell-rc edits, no updater receipt
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_UNMANAGED_INSTALL="$UV_DIR" sh
fi
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found: this script is for NVIDIA GPUs"; exit 1; }

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
MAJOR=${DRIVER%%.*}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ -z "${CUDA_FLAVOR:-}" ]; then
  if (( MAJOR >= 580 )); then CUDA_FLAVOR=cu130      # PyPI default wheels of torch 2.13 / vllm 0.31 are CUDA 13.0
  elif (( MAJOR >= 525 )); then CUDA_FLAVOR=cu129    # CUDA 12.x wheels run on any >=525 driver (minor-version compat)
  else echo "driver $DRIVER is too old for CUDA 12.9 / 13.0 wheels"; exit 1; fi
fi
echo "driver $DRIVER -> $CUDA_FLAVOR"

[ -d "$VENV" ] || uv venv --python "$PYTHON_VERSION" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# One resolver call so torch / vllm / transformers / trl constraints are solved together.
# --torch-backend sends torch, torchvision, torchaudio to the matching PyTorch index.
if [ "$CUDA_FLAVOR" = cu130 ]; then
  uv pip install "vllm==${VLLM_VERSION}" --torch-backend=cu130 -e . pytest
else
  uv pip install "vllm==${VLLM_VERSION}+${CUDA_FLAVOR}" \
    --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${CUDA_FLAVOR}" \
    --torch-backend="${CUDA_FLAVOR}" -e . pytest
fi

python - <<'EOF'
import torch, transformers, trl, vllm, datasets, math_verify  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__,
      "| transformers", transformers.__version__, "| trl", trl.__version__)
assert torch.cuda.is_available(), "torch does not see the GPU"
cc = torch.cuda.get_device_capability(0)
print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB",
      f"| compute capability {cc[0]}.{cc[1]}")
if cc < (8, 0):
    print("note: pre-Ampere GPU (no bf16): use configs/small_gpu.yaml for debugging only")
EOF
echo "setup ok (venv: $VENV). Next: ./run_all.sh (see README)"
