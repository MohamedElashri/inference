#!/usr/bin/env bash
# Retest the selected BF16/grid/work-list settings on the current Allen import.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/benchmark_v9r1_optimized.sh" \
    --label allen_master_optimized -B buildupstreamgpu12 "$@"
