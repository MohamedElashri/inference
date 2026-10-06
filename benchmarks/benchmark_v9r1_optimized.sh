#!/usr/bin/env bash
# The validated RTX 3090 BF16 full-PV shadow operating point.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH="${CUDNN_ROOT:-$HOME/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export PVF_SEQUENCES=${PVF_SEQUENCES:-"hlt1_pp_default hlt1_pp_pvs_pvfinder_unet_benchmark"}
exec bash "$SCRIPT_DIR/benchmark_pvfinder_batch.sh" \
    --label v9r1_optimized -B buildv9r1gpu12 -d 2 \
    --model unet16_lc4_scnone_asym5_best_bf16 --use-bf16 true \
    --gpu-work-list true --fc-grid-fraction 0.0625 --unet-grid-fraction 0.125 \
    -n 500 -m 500 -r 1000 -t 16 --repeats 5 "$@"
