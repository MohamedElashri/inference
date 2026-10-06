#!/usr/bin/env bash
# Build the currently imported Allen revision with the validated CUDA toolchain.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/Allen/buildupstreamgpu12}
exec bash "$REPO_ROOT/benchmarks/build_v9r1.sh" "$@"
