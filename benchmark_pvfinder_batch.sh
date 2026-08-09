#!/usr/bin/env bash
# Reproducible PVFinder benchmark batch runner.
#
# This wraps Allen benchmark runs with provenance capture, repeated measurements,
# per-run logs/configs, optional nsys profiling, and an explicit pvfinder_unet
# use_fp16 configuration toggle.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./benchmark_pvfinder_batch.sh --label LABEL [options]

Options:
  --label LABEL              Required result label, e.g. reference_A_fp32_head_d1874d8
  -B, --build-dir NAME       Allen build directory name under Allen/ (default: buildgpu16chgpu)
  -d, --device N             GPU device index (default: 2)
  -t, --threads N            Allen threads / streams (default: 16)
  -n, --events N             Events to process (default: 100)
  -m, --memory MB            Device memory per thread / stream (default: 300)
  -r, --repetitions N        Repetitions per thread / stream (default: 500)
  --repeats N                Number of repeated benchmark runs (default: 3)
  --cnn-weights PATH         Override pvfinder_unet weight_file
  --use-fp16 BOOL            Set pvfinder_unet.use_fp16 true/false (default: false)
  --skip-mode MODE           Set pvfinder_unet.skip_mode: concat|add|none (default: concat)
                             add/none are throughput-only ablations, not physics-valid
  --use-cuda-graph BOOL      Set pvfinder_unet.use_cuda_graph true/false (default: false)
                             only active when use_fp16=false and skip_mode=concat
  --use-fused-cbr BOOL       Set pvfinder_unet.use_fused_cbr true/false (default: false)
                             rcbn1 only, FP32 only; falls back automatically if
                             unsupported on the GPU (see optimization_plan.md Phase 1)
  --fwd-algo-ws-budget-mb N  Set pvfinder_unet.fwd_algo_ws_budget_bytes = N*1024*1024
                             (default: 0 = pinned IMPLICIT_GEMM, no search; see
                             optimization_plan.md Phase 2)
  --use-fused-rcbn3 BOOL     Set pvfinder_unet.use_fused_rcbn3 true/false (default: false)
                             rcbn3 only, eager FP32 path only; see
                             optimization_plan.md Phase 3
  --l6a-m N                  Override pvfinder_fc_aggregation.l6a_m GEMM row count
                             (default: 800, physics-valid; any other value is
                             throughput-only tile-alignment testing, not physics-valid)
  --use-nonatomic-l6a-reduce BOOL
                             Set pvfinder_fc_aggregation.use_nonatomic_l6a_reduce
                             true/false (default: false); see optimization_plan.md
  --use-warp-parallel-reduce BOOL
                             Set pvfinder_fc_aggregation.use_warp_parallel_reduce
                             true/false (default: false); Phase 6 idea 2, see
                             optimization_plan.md
  --fc-chunk-size N          Set pvfinder_fc_aggregation.fc_chunk_size
                             (default: 20); Phase 6 idea 3, see optimization_plan.md
  --use-fused-bias-relu-reduce BOOL
                             Set pvfinder_fc_aggregation.use_fused_bias_relu_reduce
                             true/false (default: false); Phase 7 idea 1, see
                             optimization_plan.md
  --skip-redundant-memset BOOL
                             Set pvfinder_fc_aggregation.skip_redundant_memset
                             true/false (default: false); Phase 8 idea 1, see
                             optimization_plan.md
  --use-grid-stride-reduce BOOL
                             Set pvfinder_fc_aggregation.use_grid_stride_reduce
                             true/false (default: false); Phase 15, see
                             optimization_plan.md
  --profile                  Run each sequence under nsys
  --result-root DIR          Directory for batches (default: benchmark_results)
  -h, --help                 Show this help

Example:
  ./benchmark_pvfinder_batch.sh \
    --label reference_A_fp32_head_d1874d8 \
    -B buildgpu16chgpu --cnn-weights cnn_weights_16ch.bin \
    -d 2 -t 16 -n 100 -m 300 -r 500 --repeats 3 --use-fp16 false
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_ARGS=("$@")

LABEL=""
BUILD_NAME="buildgpu16chgpu"
DEVICE=2
THREADS=16
EVENTS=100
MEMORY=300
REPS=500
REPEATS=3
CNN_WEIGHTS_OVERRIDE=""
USE_FP16=false
SKIP_MODE=concat
USE_CUDA_GRAPH=false
USE_FUSED_CBR=false
FWD_ALGO_WS_BUDGET_MB=0
USE_FUSED_RCBN3=false
L6A_M=800
USE_NONATOMIC_L6A_REDUCE=false
USE_WARP_PARALLEL_REDUCE=false
FC_CHUNK_SIZE=20
USE_FUSED_BIAS_RELU_REDUCE=false
SKIP_REDUNDANT_MEMSET=false
USE_GRID_STRIDE_REDUCE=false
PROFILE=0
RESULT_ROOT="${SCRIPT_DIR}/benchmark_results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --build-dir|-B) BUILD_NAME="$2"; shift 2 ;;
        --device|-d) DEVICE="$2"; shift 2 ;;
        --threads|-t) THREADS="$2"; shift 2 ;;
        --events|-n) EVENTS="$2"; shift 2 ;;
        --memory|-m) MEMORY="$2"; shift 2 ;;
        --repetitions|-r) REPS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --cnn-weights) CNN_WEIGHTS_OVERRIDE="$2"; shift 2 ;;
        --use-fp16) USE_FP16="$2"; shift 2 ;;
        --skip-mode) SKIP_MODE="$2"; shift 2 ;;
        --use-cuda-graph) USE_CUDA_GRAPH="$2"; shift 2 ;;
        --use-fused-cbr) USE_FUSED_CBR="$2"; shift 2 ;;
        --fwd-algo-ws-budget-mb) FWD_ALGO_WS_BUDGET_MB="$2"; shift 2 ;;
        --use-fused-rcbn3) USE_FUSED_RCBN3="$2"; shift 2 ;;
        --l6a-m) L6A_M="$2"; shift 2 ;;
        --use-nonatomic-l6a-reduce) USE_NONATOMIC_L6A_REDUCE="$2"; shift 2 ;;
        --use-warp-parallel-reduce) USE_WARP_PARALLEL_REDUCE="$2"; shift 2 ;;
        --fc-chunk-size) FC_CHUNK_SIZE="$2"; shift 2 ;;
        --use-fused-bias-relu-reduce) USE_FUSED_BIAS_RELU_REDUCE="$2"; shift 2 ;;
        --skip-redundant-memset) SKIP_REDUNDANT_MEMSET="$2"; shift 2 ;;
        --use-grid-stride-reduce) USE_GRID_STRIDE_REDUCE="$2"; shift 2 ;;
        --profile) PROFILE=1; shift 1 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${LABEL}" ]]; then
    echo "ERROR: --label is required" >&2
    usage >&2
    exit 1
fi

case "${USE_FP16}" in
    true|false) ;;
    *) echo "ERROR: --use-fp16 must be true or false" >&2; exit 1 ;;
esac

case "${SKIP_MODE}" in
    concat|add|none) ;;
    *) echo "ERROR: --skip-mode must be concat, add, or none" >&2; exit 1 ;;
esac

case "${USE_CUDA_GRAPH}" in
    true|false) ;;
    *) echo "ERROR: --use-cuda-graph must be true or false" >&2; exit 1 ;;
esac

case "${USE_FUSED_CBR}" in
    true|false) ;;
    *) echo "ERROR: --use-fused-cbr must be true or false" >&2; exit 1 ;;
esac

if ! [[ "${FWD_ALGO_WS_BUDGET_MB}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --fwd-algo-ws-budget-mb must be a non-negative integer" >&2
    exit 1
fi

case "${USE_FUSED_RCBN3}" in
    true|false) ;;
    *) echo "ERROR: --use-fused-rcbn3 must be true or false" >&2; exit 1 ;;
esac

case "${USE_NONATOMIC_L6A_REDUCE}" in
    true|false) ;;
    *) echo "ERROR: --use-nonatomic-l6a-reduce must be true or false" >&2; exit 1 ;;
esac

case "${USE_WARP_PARALLEL_REDUCE}" in
    true|false) ;;
    *) echo "ERROR: --use-warp-parallel-reduce must be true or false" >&2; exit 1 ;;
esac

if ! [[ "${FC_CHUNK_SIZE}" =~ ^[0-9]+$ ]] || [[ "${FC_CHUNK_SIZE}" -lt 1 ]]; then
    echo "ERROR: --fc-chunk-size must be a positive integer" >&2
    exit 1
fi

case "${USE_FUSED_BIAS_RELU_REDUCE}" in
    true|false) ;;
    *) echo "ERROR: --use-fused-bias-relu-reduce must be true or false" >&2; exit 1 ;;
esac

case "${SKIP_REDUNDANT_MEMSET}" in
    true|false) ;;
    *) echo "ERROR: --skip-redundant-memset must be true or false" >&2; exit 1 ;;
esac

case "${USE_GRID_STRIDE_REDUCE}" in
    true|false) ;;
    *) echo "ERROR: --use-grid-stride-reduce must be true or false" >&2; exit 1 ;;
esac

if ! [[ "${L6A_M}" =~ ^[0-9]+$ ]] || [[ "${L6A_M}" -lt 1 ]] || [[ "${L6A_M}" -gt 800 ]]; then
    echo "ERROR: --l6a-m must be an integer in [1, 800] (800 = physics-valid default)" >&2
    exit 1
fi

BUILD_DIR="${SCRIPT_DIR}/Allen/${BUILD_NAME}"
ALLEN_WRAPPER="${BUILD_DIR}/toolchain/wrapper"
ALLEN_BIN="${BUILD_DIR}/Allen"
MDF="${SCRIPT_DIR}/Allen/input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf"
GEO="${SCRIPT_DIR}/Allen/input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry"

if [[ ! -x "${ALLEN_WRAPPER}" || ! -x "${ALLEN_BIN}" ]]; then
    echo "ERROR: build does not look runnable: ${BUILD_DIR}" >&2
    exit 1
fi

if [[ -n "${CNN_WEIGHTS_OVERRIDE}" ]]; then
    if [[ "${CNN_WEIGHTS_OVERRIDE}" = /* ]]; then
        CNN_WEIGHTS_ABS="${CNN_WEIGHTS_OVERRIDE}"
    else
        CNN_WEIGHTS_ABS="${SCRIPT_DIR}/${CNN_WEIGHTS_OVERRIDE}"
    fi
    if [[ ! -f "${CNN_WEIGHTS_ABS}" ]]; then
        echo "ERROR: --cnn-weights file not found: ${CNN_WEIGHTS_ABS}" >&2
        exit 1
    fi
else
    CNN_WEIGHTS_ABS="${SCRIPT_DIR}/cnn_weights.bin"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_LABEL="$(printf '%s' "${LABEL}" | tr -c 'A-Za-z0-9_.=-' '_')"
BATCH_DIR="${RESULT_ROOT}/${TIMESTAMP}_${SAFE_LABEL}"
mkdir -p "${BATCH_DIR}"

COMMON_ARGS=(
    --mdf "${MDF}"
    -g "${GEO}"
    -n "${EVENTS}"
    -m "${MEMORY}"
    -r "${REPS}"
    -t "${THREADS}"
)

SEQUENCES=(
    "hlt1_pp_default"
    "hlt1_pp_pvfinder_benchmark"
    "hlt1_pp_pvfinder_unet_benchmark"
)

sequence_label() {
    case "$1" in
        hlt1_pp_default) echo "baseline" ;;
        hlt1_pp_pvfinder_benchmark) echo "fc" ;;
        hlt1_pp_pvfinder_unet_benchmark) echo "unet" ;;
        *) echo "$1" ;;
    esac
}

extract_rate() {
    grep -oP '[0-9]+\.[0-9]+(?=\s+events/s)' "$1" | tail -1
}

write_command() {
    local out="$1"; shift
    printf '%q ' "$@" > "${out}"
    printf '\n' >> "${out}"
}

patch_unet_config() {
    local config="$1"
    python3 - "$config" "$CNN_WEIGHTS_ABS" "$USE_FP16" "$SKIP_MODE" "$USE_CUDA_GRAPH" "$USE_FUSED_CBR" "$FWD_ALGO_WS_BUDGET_MB" "$USE_FUSED_RCBN3" <<'PY'
import json
import sys

path, weights, use_fp16_raw, skip_mode, use_cuda_graph_raw, use_fused_cbr_raw, fwd_ws_budget_mb_raw, use_fused_rcbn3_raw = sys.argv[1:]
use_fp16 = use_fp16_raw == "true"
use_cuda_graph = use_cuda_graph_raw == "true"
use_fused_cbr = use_fused_cbr_raw == "true"
fwd_ws_budget_bytes = int(fwd_ws_budget_mb_raw) * 1024 * 1024
use_fused_rcbn3 = use_fused_rcbn3_raw == "true"

with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

pvfinder_unet = data.setdefault("pvfinder_unet", {})
pvfinder_unet["weight_file"] = weights
pvfinder_unet["use_fp16"] = use_fp16
pvfinder_unet["skip_mode"] = skip_mode
pvfinder_unet["use_cuda_graph"] = use_cuda_graph
pvfinder_unet["use_fused_cbr"] = use_fused_cbr
pvfinder_unet["fwd_algo_ws_budget_bytes"] = fwd_ws_budget_bytes
pvfinder_unet["use_fused_rcbn3"] = use_fused_rcbn3

with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

patch_fc_config() {
    local config="$1"
    python3 - "$config" "$L6A_M" "$USE_NONATOMIC_L6A_REDUCE" \
        "$USE_WARP_PARALLEL_REDUCE" "$FC_CHUNK_SIZE" \
        "$USE_FUSED_BIAS_RELU_REDUCE" "$SKIP_REDUNDANT_MEMSET" \
        "$USE_GRID_STRIDE_REDUCE" <<'PY'
import json
import sys

(path, l6a_m_raw, use_nonatomic_raw, use_warp_parallel_raw, fc_chunk_size_raw,
 use_fused_bias_relu_raw, skip_memset_raw, use_grid_stride_reduce_raw) = sys.argv[1:]

with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

fc_agg = data.setdefault("pvfinder_fc_aggregation", {})
fc_agg["l6a_m"] = int(l6a_m_raw)
fc_agg["use_nonatomic_l6a_reduce"] = use_nonatomic_raw == "true"
fc_agg["use_warp_parallel_reduce"] = use_warp_parallel_raw == "true"
fc_agg["fc_chunk_size"] = int(fc_chunk_size_raw)
fc_agg["use_fused_bias_relu_reduce"] = use_fused_bias_relu_raw == "true"
fc_agg["skip_redundant_memset"] = skip_memset_raw == "true"
fc_agg["use_grid_stride_reduce"] = use_grid_stride_reduce_raw == "true"

with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

generate_config() {
    local seq="$1"
    local out_config="$2"
    local log="$3"
    local seq_py="${BUILD_DIR}/code_generation/sequences/AllenSequences/${seq}.py"
    local tmp

    if [[ ! -f "${seq_py}" ]]; then
        echo "ERROR: sequence Python file not found: ${seq_py}" >&2
        exit 1
    fi

    tmp="$(mktemp -d)"
    (
        cd "${tmp}"
        "${ALLEN_WRAPPER}" bash -c '
            export ALLEN_BUILD_DIR="$1"
            export PYTHONPATH="$1/code_generation/sequences:${PYTHONPATH:-}"
            python3 "$1/code_generation/sequences/AllenCore/gen_allen_json.py" \
                --no-register-keys --seqpath "$2"
        ' bash "${BUILD_DIR}" "${seq_py}"
    ) > "${log}" 2>&1

    if [[ ! -f "${tmp}/Sequence.json" ]]; then
        echo "ERROR: failed to generate config for ${seq}; see ${log}" >&2
        rm -rf "${tmp}"
        exit 1
    fi

    cp "${tmp}/Sequence.json" "${out_config}"
    rm -rf "${tmp}"

    if [[ "${seq}" == "hlt1_pp_pvfinder_unet_benchmark" ]]; then
        patch_unet_config "${out_config}"
    fi
    if [[ "${seq}" == "hlt1_pp_pvfinder_benchmark" || "${seq}" == "hlt1_pp_pvfinder_unet_benchmark" ]]; then
        patch_fc_config "${out_config}"
    fi
}

run_sequence() {
    local seq="$1"
    local run_dir="$2"
    local short
    short="$(sequence_label "${seq}")"

    local config="${run_dir}/${short}_effective_config.json"
    local gen_log="${run_dir}/${short}_generate_config.log"
    local log="${run_dir}/bench_${short}.log"
    local cmd_file="${run_dir}/bench_${short}.cmd"

    generate_config "${seq}" "${config}" "${gen_log}"

    if [[ "${PROFILE}" -eq 1 ]]; then
        local profile_out="${run_dir}/pvfinder_profile_${short}"
        write_command "${cmd_file}" nsys profile -f true --stats=true -o "${profile_out}" -t cuda \
            "${ALLEN_WRAPPER}" "${ALLEN_BIN}" --sequence "${config}" \
            "${COMMON_ARGS[@]}" --device "${DEVICE}"
        (
            cd "${run_dir}"
            nsys profile -f true --stats=true -o "${profile_out}" -t cuda \
                "${ALLEN_WRAPPER}" "${ALLEN_BIN}" --sequence "${config}" \
                "${COMMON_ARGS[@]}" --device "${DEVICE}"
        ) > "${log}" 2>&1
    else
        write_command "${cmd_file}" "${ALLEN_WRAPPER}" "${ALLEN_BIN}" --sequence "${config}" \
            "${COMMON_ARGS[@]}" --device "${DEVICE}"
        (
            cd "${run_dir}"
            "${ALLEN_WRAPPER}" "${ALLEN_BIN}" --sequence "${config}" \
                "${COMMON_ARGS[@]}" --device "${DEVICE}"
        ) > "${log}" 2>&1
    fi

    local rate
    rate="$(extract_rate "${log}")"
    if [[ -z "${rate}" ]]; then
        echo "ERROR: could not extract rate for ${seq}; see ${log}" >&2
        exit 1
    fi
    printf '%s\t%s\n' "${short}" "${rate}"
}

{
    echo "# PVFinder benchmark batch"
    echo "label=${LABEL}"
    echo "timestamp=${TIMESTAMP}"
    echo "build_name=${BUILD_NAME}"
    echo "build_dir=${BUILD_DIR}"
    echo "device=${DEVICE}"
    echo "threads=${THREADS}"
    echo "events=${EVENTS}"
    echo "memory=${MEMORY}"
    echo "repetitions=${REPS}"
    echo "repeats=${REPEATS}"
    echo "profile=${PROFILE}"
    echo "cnn_weights=${CNN_WEIGHTS_ABS}"
    echo "use_fp16=${USE_FP16}"
    echo "skip_mode=${SKIP_MODE}"
    echo "use_cuda_graph=${USE_CUDA_GRAPH}"
    echo "use_fused_cbr=${USE_FUSED_CBR}"
    echo "fwd_algo_ws_budget_mb=${FWD_ALGO_WS_BUDGET_MB}"
    echo "use_fused_rcbn3=${USE_FUSED_RCBN3}"
    echo "l6a_m=${L6A_M}"
    echo "use_nonatomic_l6a_reduce=${USE_NONATOMIC_L6A_REDUCE}"
    echo "use_warp_parallel_reduce=${USE_WARP_PARALLEL_REDUCE}"
    echo "fc_chunk_size=${FC_CHUNK_SIZE}"
    echo "use_fused_bias_relu_reduce=${USE_FUSED_BIAS_RELU_REDUCE}"
    echo "skip_redundant_memset=${SKIP_REDUNDANT_MEMSET}"
    echo "use_grid_stride_reduce=${USE_GRID_STRIDE_REDUCE}"
    echo "mdf=${MDF}"
    echo "geometry=${GEO}"
} > "${BATCH_DIR}/metadata.env"

git -C "${SCRIPT_DIR}" rev-parse HEAD > "${BATCH_DIR}/git_head.txt"
git -C "${SCRIPT_DIR}" status --short > "${BATCH_DIR}/git_status_short.txt"
git -C "${SCRIPT_DIR}" log --oneline -8 --decorate > "${BATCH_DIR}/git_log_oneline.txt"

sha256sum "${CNN_WEIGHTS_ABS}" > "${BATCH_DIR}/weights.sha256"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${BATCH_DIR}/nvidia_smi.txt" 2>&1 || true
    nvidia-smi pmon -c 5 > "${BATCH_DIR}/nvidia_smi_pmon.txt" 2>&1 || true
fi

write_command "${BATCH_DIR}/batch_command.cmd" "$0" "${ORIGINAL_ARGS[@]}"

printf 'run\tbaseline\tfc\tunet\tfc_overhead_pct\tunet_overhead_pct\tunet_retention_pct\n' \
    > "${BATCH_DIR}/summary.tsv"

for run_idx in $(seq 1 "${REPEATS}"); do
    run_dir="${BATCH_DIR}/run_$(printf '%02d' "${run_idx}")"
    mkdir -p "${run_dir}"
    printf 'Running batch %s, repeat %s/%s...\n' "${LABEL}" "${run_idx}" "${REPEATS}"

    rates_file="${run_dir}/rates.tsv"
    : > "${rates_file}"
    for seq in "${SEQUENCES[@]}"; do
        run_sequence "${seq}" "${run_dir}" | tee -a "${rates_file}"
    done

    baseline="$(awk '$1=="baseline"{print $2}' "${rates_file}")"
    fc="$(awk '$1=="fc"{print $2}' "${rates_file}")"
    unet="$(awk '$1=="unet"{print $2}' "${rates_file}")"

    awk -v run="${run_idx}" -v base="${baseline}" -v fc="${fc}" -v unet="${unet}" '
      BEGIN {
        fc_over = (base > 0) ? (base - fc) / base * 100.0 : 0.0;
        unet_over = (base > 0) ? (base - unet) / base * 100.0 : 0.0;
        retain = (base > 0) ? unet / base * 100.0 : 0.0;
        printf "%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",
          run, base, fc, unet, fc_over, unet_over, retain;
      }' >> "${BATCH_DIR}/summary.tsv"
done

awk '
  NR == 1 { next }
  {
    b[NR-1] = $2; f[NR-1] = $3; u[NR-1] = $4;
    fo[NR-1] = $5; uo[NR-1] = $6; r[NR-1] = $7;
    n = NR-1;
  }
  function sort(a, n, i, j, t) {
    for (i = 1; i <= n; ++i) for (j = i + 1; j <= n; ++j) if (a[j] < a[i]) {
      t = a[i]; a[i] = a[j]; a[j] = t;
    }
  }
  function median(a, n) {
    sort(a, n);
    return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2.0;
  }
  END {
    if (n == 0) exit;
    mb = median(b, n); mf = median(f, n); mu = median(u, n);
    mfo = median(fo, n); muo = median(uo, n); mr = median(r, n);
    minb = b[1]; maxb = b[n];
    spread = (mb > 0) ? (maxb - minb) / mb * 100.0 : 0.0;
    printf "# PVFinder benchmark summary\n\n";
    printf "- repeats: %d\n", n;
    printf "- median baseline events/s: %.2f\n", mb;
    printf "- median FC events/s: %.2f\n", mf;
    printf "- median FC+UNet events/s: %.2f\n", mu;
    printf "- median FC overhead: %.2f%%\n", mfo;
    printf "- median FC+UNet overhead: %.2f%%\n", muo;
    printf "- median FC+UNet retention: %.2f%%\n", mr;
    printf "- baseline spread: %.2f%%\n", spread;
    if (spread > 5.0) {
      printf "- contention status: contended, repeat later\n";
    } else {
      printf "- contention status: acceptable\n";
    }
  }
' "${BATCH_DIR}/summary.tsv" > "${BATCH_DIR}/summary.md"

printf '\nBatch complete: %s\n' "${BATCH_DIR}"
cat "${BATCH_DIR}/summary.md"
