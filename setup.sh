#!/usr/bin/env bash
# One-time environment setup on a Linux + NVIDIA machine (tested pins, see README "Stack").
#   ./setup.sh                      # picks the CUDA flavour from the driver
#   CUDA_FLAVOR=cu129 ./setup.sh    # force CUDA 12.9 wheels
set -euo pipefail
cd "$(dirname "$0")"

VLLM_VERSION=0.31.0          # hard-pins torch==2.13.0
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
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

[ -d .venv ] || uv venv --python "$PYTHON_VERSION" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

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
print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB",
      "| bf16:", torch.cuda.is_bf16_supported())
EOF
echo "setup ok. Next: ./run_all.sh (see README)"
