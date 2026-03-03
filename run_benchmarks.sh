#!/bin/bash
# run_benchmarks.sh
#
# Three-way throughput benchmark on the two RTX 2080 Ti GPUs:
#   Run 1 (device 0): HLT1 baseline         — hlt1_pp_default
#   Run 2 (device 1): HLT1 + PVFinder FC    — hlt1_pp_pvfinder_benchmark
#   Run 3 (device 0): HLT1 + PVFinder FC+UNet — hlt1_pp_pvfinder_unet_benchmark
#
# Runs 1 and 2 are executed in parallel (one GPU each).
# Run 3 runs afterwards on device 0 (same GPU as baseline for a fair comparison).
#
# Usage:
#   ./run_benchmarks.sh [--device0 N] [--device1 M] [--threads T]
#                          [--events N] [--slices M] [--repetitions R]
#
# Defaults:
#   devices 0 and 1, -t 16, -n 500, -m 500, -r 1000
#
# Output:
#   Allen/buildgpu/bench_baseline.log
#   Allen/buildgpu/bench_fc.log
#   Allen/buildgpu/bench_unet.log
#   Allen/buildgpu/bench_results_<timestamp>.txt   (summary)
###############################################################################
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEVICE0=0
DEVICE1=1
THREADS=16
EVENTS=500
SLICES=500
REPS=1000

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device0)    DEVICE0="$2";    shift 2 ;;
        --device1)    DEVICE1="$2";    shift 2 ;;
        --threads|-t) THREADS="$2";    shift 2 ;;
        --events|-n)  EVENTS="$2";     shift 2 ;;
        --slices|-m)  SLICES="$2";     shift 2 ;;
        --repetitions|-r) REPS="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths  (all relative to repo root; script can be run from anywhere)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/Allen/buildgpu"
ALLEN="${BUILD_DIR}/toolchain/wrapper ${BUILD_DIR}/Allen"
MDF="${SCRIPT_DIR}/Allen/input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf"
GEO="${SCRIPT_DIR}/Allen/input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry"
WEIGHT_FILE="${SCRIPT_DIR}/cnn_weights.bin"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_BASE="${BUILD_DIR}/bench_baseline.log"
LOG_FC="${BUILD_DIR}/bench_fc.log"
LOG_UNET="${BUILD_DIR}/bench_unet.log"
SUMMARY="${BUILD_DIR}/bench_results_${TIMESTAMP}.txt"

COMMON_ARGS="--mdf ${MDF} -g ${GEO} -n ${EVENTS} -m ${SLICES} -r ${REPS} -t ${THREADS}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
extract_rate() {
    # Extracts the final "X events/s" line from a log file
    grep -oP '[0-9]+\.[0-9]+(?=\s+events/s)' "$1" | tail -1
}

print_header() {
    echo "================================================================"
    echo " PVFinder Benchmark Suite"
    echo " $(date)"
    echo " MDF    : ${MDF}"
    echo " Geometry: ${GEO}"
    echo " Params : -n ${EVENTS} -m ${SLICES} -r ${REPS} -t ${THREADS}"
    echo " Device0: ${DEVICE0}   Device1: ${DEVICE1}"
    echo "================================================================"
}

# ---------------------------------------------------------------------------
# Phase 1: Baseline (device 0) and FC (device 1) — run in parallel
# ---------------------------------------------------------------------------
print_header

echo ""
echo "[Phase 1] Running baseline (device ${DEVICE0}) and FC (device ${DEVICE1}) in parallel..."

${ALLEN} --sequence hlt1_pp_default \
    ${COMMON_ARGS} --device ${DEVICE0} > "${LOG_BASE}" 2>&1 &
PID_BASE=$!

${ALLEN} --sequence hlt1_pp_pvfinder_benchmark \
    ${COMMON_ARGS} --device ${DEVICE1} > "${LOG_FC}" 2>&1 &
PID_FC=$!

wait ${PID_BASE}
RC_BASE=$?
wait ${PID_FC}
RC_FC=$?

[[ ${RC_BASE} -ne 0 ]] && { echo "ERROR: baseline run failed (exit ${RC_BASE}). See ${LOG_BASE}"; exit 1; }
[[ ${RC_FC}   -ne 0 ]] && { echo "ERROR: FC run failed (exit ${RC_FC}). See ${LOG_FC}";       exit 1; }

echo "  Phase 1 complete."

# ---------------------------------------------------------------------------
# Phase 2: FC+UNet (device 0) — sequential so it gets the same GPU as baseline
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 2] Running FC+UNet on device ${DEVICE0} (same GPU as baseline)..."

${ALLEN} --sequence hlt1_pp_pvfinder_unet_benchmark \
    ${COMMON_ARGS} --device ${DEVICE0} > "${LOG_UNET}" 2>&1

echo "  Phase 2 complete."

# ---------------------------------------------------------------------------
# Extract rates and compute deltas
# ---------------------------------------------------------------------------
RATE_BASE=$(extract_rate "${LOG_BASE}")
RATE_FC=$(extract_rate "${LOG_FC}")
RATE_UNET=$(extract_rate "${LOG_UNET}")

if [[ -z "${RATE_BASE}" || -z "${RATE_FC}" || -z "${RATE_UNET}" ]]; then
    echo ""
    echo "ERROR: Could not extract one or more throughput values."
    echo "  baseline : '${RATE_BASE}'"
    echo "  fc       : '${RATE_FC}'"
    echo "  unet     : '${RATE_UNET}'"
    echo "Check logs: ${LOG_BASE}  ${LOG_FC}  ${LOG_UNET}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
awk -v base="${RATE_BASE}" -v fc="${RATE_FC}" -v unet="${RATE_UNET}" \
    -v ts="${TIMESTAMP}" -v t="${THREADS}" -v n="${EVENTS}" \
    -v m="${SLICES}" -v r="${REPS}" \
    -v d0="${DEVICE0}" -v d1="${DEVICE1}" \
    -v log_base="${LOG_BASE}" -v log_fc="${LOG_FC}" -v log_unet="${LOG_UNET}" \
'BEGIN {
    fc_diff    = base - fc
    unet_diff  = base - unet
    fc_pct     = (base > 0) ? (fc_diff  / base) * 100 : 0
    unet_pct   = (base > 0) ? (unet_diff / base) * 100 : 0

    printf "\n"
    printf "================================================================\n"
    printf " PVFinder Benchmark Results — %s\n", ts
    printf " -n %s  -m %s  -r %s  -t %s\n", n, m, r, t
    printf "================================================================\n"
    printf "  %-40s  %12s  %10s  %8s\n", "Sequence", "events/s", "delta", "overhead"
    printf "  %-40s  %12s  %10s  %8s\n", "--------", "--------", "-----", "-------"
    printf "  %-40s  %12.2f  %10s  %8s\n",  "hlt1_pp_default (baseline)",          base, "--", "--"
    printf "  %-40s  %12.2f  %10.2f  %7.2f%%\n", "hlt1_pp_pvfinder_benchmark (FC)",     fc,   fc_diff,   fc_pct
    printf "  %-40s  %12.2f  %10.2f  %7.2f%%\n", "hlt1_pp_pvfinder_unet_benchmark (FC+UNet)", unet, unet_diff, unet_pct
    printf "================================================================\n"
    printf "\n"
    printf "  Logs:\n"
    printf "    baseline : %s\n", log_base
    printf "    FC       : %s\n", log_fc
    printf "    FC+UNet  : %s\n", log_unet
    printf "\n"
}' | tee "${SUMMARY}"

echo "Summary saved to: ${SUMMARY}"
