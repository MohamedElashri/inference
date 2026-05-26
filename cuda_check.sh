#!/usr/bin/env bash
set -euo pipefail

echo "=== Host ==="
hostname
date

echo
echo "=== NVIDIA driver / GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total --format=csv || nvidia-smi
else
    echo "nvidia-smi not found"
fi

echo
echo "=== CUDA toolkit ==="
echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "PATH nvcc: $(command -v nvcc || true)"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc not found"
fi

echo
echo "=== Modules ==="
module list 2>&1 || true

echo
echo "=== Python / PyTorch ==="
python - <<'PY' || true
import sys
print("Python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("No usable torch check:", repr(e))
PY
