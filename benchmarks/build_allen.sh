#!/usr/bin/env bash
# Configure and build standalone Allen and the PVFinder GPU unit tests.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_ROOT/benchmarks/defaults.sh"
SOURCE_DIR=$(realpath "${SOURCE_DIR:-$REPO_ROOT/Allen}")
BUILD_DIR=$(realpath -m "${BUILD_DIR:-$PVF_BUILD_DIR}")
JOBS=${JOBS:-12}
DEPENDENCY_CACHE=${DEPENDENCY_CACHE:-$REPO_ROOT/.cache/allen-dependencies}
CUDNN_ROOT=${CUDNN_ROOT:-$HOME/local/cuda}
CUDA_COMPILER=${CUDA_COMPILER:-/usr/local/cuda-12.8/bin/nvcc}
TOOLCHAIN_FILE=${TOOLCHAIN_FILE:-/cvmfs/lhcb.cern.ch/lib/lhcb/lcg-toolchains/LCG_108c/x86_64_v3-el9-gcc13+cuda12_4-opt+g.cmake}
DRY_RUN=${DRY_RUN:-0}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-d) DRY_RUN=1 ;;
        --help|-h)
            cat <<'HELP'
Usage: bash benchmarks/build_allen.sh [--dry-run]
Build Allen and GPU unit tests in Allen/build.
Environment overrides: SOURCE_DIR, BUILD_DIR (or ALLEN_BUILD_DIR), JOBS,
DEPENDENCY_CACHE (LHCb/Gaudi/ParamFiles sources), CUDNN_ROOT, CUDA_ARCH,
CUDA_COMPILER, TOOLCHAIN_FILE, BUILD_TESTING, WITH_CUDNN,
PVFINDER_UNET_N_FEAT, PVFINDER_UNET_N_BATCH_CHANNELS.
HELP
            exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
LHCB_SOURCE=$(realpath -m "${LHCBROOT:-$DEPENDENCY_CACHE/LHCb}")
GAUDI_SOURCE=$(realpath -m "${GAUDIROOT:-$DEPENDENCY_CACHE/Gaudi}")
PARAMFILES_SOURCE=$(realpath -m "${PARAMFILESROOT:-$DEPENDENCY_CACHE/ParamFiles}")

if [[ $DRY_RUN != 1 ]]; then
    for source_root in "$LHCB_SOURCE" "$GAUDI_SOURCE" "$PARAMFILES_SOURCE"; do
        [[ -d $source_root ]] || { echo "Missing dependency source: $source_root; set DEPENDENCY_CACHE or LHCBROOT/GAUDIROOT/PARAMFILESROOT" >&2; exit 1; }
    done
    set +u
    source /cvmfs/lhcb.cern.ch/lib/LbEnv
    set -u
fi
# Reuse source repositories directly; never symlink build/external to the cache.
export LHCBROOT="$LHCB_SOURCE" GAUDIROOT="$GAUDI_SOURCE" PARAMFILESROOT="$PARAMFILES_SOURCE"
args=(
    -S "$SOURCE_DIR" -B "$BUILD_DIR"
    -DSTANDALONE=ON -DTARGET_DEVICE=CUDA -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE" "-DCMAKE_CUDA_COMPILER=$CUDA_COMPILER"
    "-DCUDA_ARCH=${CUDA_ARCH:-86}" -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
    -DCMAKE_SHARED_LINKER_FLAGS=
    '-DCMAKE_CUDA_FLAGS=--device-entity-has-hidden-visibility=true --static-global-template-stub=true'
    "-DWITH_CUDNN=${WITH_CUDNN:-ON}" "-DCUDNN_LIBRARY=$CUDNN_ROOT/lib64/libcudnn.so"
    "-DCUDNN_INCLUDE_DIR=$CUDNN_ROOT/include" "-DBUILD_TESTING=${BUILD_TESTING:-ON}"
    "-DPVFINDER_UNET_N_FEAT=${PVFINDER_UNET_N_FEAT:-16}"
    "-DPVFINDER_UNET_N_BATCH_CHANNELS=${PVFINDER_UNET_N_BATCH_CHANNELS:-4}"
    "-DLHCBROOT=$LHCBROOT" "-DGAUDIROOT=$GAUDIROOT" "-DPARAMFILESROOT=$PARAMFILESROOT"
)
targets=(Allen)
[[ ${BUILD_TESTING:-ON} == ON ]] && targets+=(unit_tests)
if [[ $DRY_RUN == 1 ]]; then
    printf '%q ' cmake "${args[@]}"; printf '\n'
    printf '%q ' cmake --build "$BUILD_DIR" --target "${targets[@]}" -j "$JOBS"; printf '\n'
else
    cmake "${args[@]}"
    cmake --build "$BUILD_DIR" --target "${targets[@]}" -j "$JOBS"
fi
