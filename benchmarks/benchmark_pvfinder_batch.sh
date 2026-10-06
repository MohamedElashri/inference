#!/usr/bin/env bash
# Reproducible PVFinder benchmark batch runner.
#
# This wraps Allen benchmark runs with provenance capture, repeated measurements,
# per-run logs/configs, optional nsys profiling, and the PVFinder precision
# (--use-bf16).
#
# Every batch, including failed and interrupted ones, leaves a JSON run record
# in results/runs/ (tracked in git; see results/README.md and
# benchmarks/runs.py). Raw logs, configs and nsys reports stay in the ignored
# batch directory under benchmark_results/.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  benchmarks/benchmark_pvfinder_batch.sh --label LABEL [options]

Options:
  --label LABEL              Required result label, e.g. reference_A_fp32_head_d1874d8
  -B, --build-dir NAME       Build name under Allen/ or absolute path (default: build)
  -d, --device N             GPU device index (default: 2)
  -t, --threads N            Allen threads / streams (default: 16)
  -n, --events N             Events to process (default: 500)
  -m, --memory MB            Device memory per thread / stream (default: 500)
  -r, --repetitions N        Repetitions per thread / stream (default: 1000)
  --repeats N                Number of repeated benchmark runs (default: 5)
  --alternate-order          Reverse sequence order on even repeats
  --telemetry                Log GPU clocks/power/throttling and host scheduling
  --cpu-affinity LIST        Run Allen with taskset CPU affinity (default: unbound)
  --model NAME               Weights from the weights/ pipeline:
                             weights/out/NAME/pvfinder_model.json
                             (default: unet16_lc4_scnone_asym5_best_bf16; see make -C weights list)
  --model-file PATH          The model file for both algorithms' "model" property
                             (default: weights/out/<--model>/pvfinder_model.json)
  --use-bf16 BOOL            precision = bfloat16 for pvfinder_fc_aggregation and
                             pvfinder_unet (default: true)
  --gpu-work-list BOOL      Build FC work lists on the GPU (BF16 only; defaults to --use-bf16)
  --unet-batch-events N      Set pvfinder_unet.unet_batch_events and
                             pvfinder_fc_aggregation.unet_batch_events together
                             (default: 20); float32 cuDNN batch size in events
  --fc-grid-fraction F       Set pvfinder_fc_aggregation.fused_grid_fraction (default 0.0625)
  --unet-grid-fraction F     Set pvfinder_unet.fused_grid_fraction (default 0.125)
  --profile                  Run each sequence under nsys; the record gets
                             the per-sequence kernel summary
  --result-root DIR          Directory for batches (default: benchmark_results)
  --no-record                Do not write a results/runs/ record (smoke tests)
  -h, --help                 Show this help

Default sequences: plain HLT1 and HLT1 with the full PVFinder shadow chain.
Set PVF_SEQUENCES="sequence_a sequence_b ..." to compare other stages.
Set MDF_FILE and GEOMETRY_DIR together to select an input and matching geometry.

Example:
  bash benchmarks/benchmark_pvfinder_batch.sh --label production --repeats 1
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The script lives in benchmarks/; Allen, weights and results are resolved
# from the repository root.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "$SCRIPT_DIR/defaults.sh"
ORIGINAL_ARGS=("$@")

LABEL=""
BUILD_NAME="$PVF_BUILD_DIR"
DEVICE=$PVF_DEVICE
THREADS=$PVF_THREADS
EVENTS=$PVF_EVENTS
MEMORY=$PVF_MEMORY
REPS=$PVF_REPETITIONS
REPEATS=$PVF_REPEATS
MODEL=$PVF_MODEL
MODEL_FILE_OVERRIDE=""
USE_BF16=true
GPU_WORK_LIST=""
UNET_BATCH_EVENTS=20
FC_GRID_FRACTION=$PVF_FC_GRID_FRACTION
UNET_GRID_FRACTION=$PVF_UNET_GRID_FRACTION
PROFILE=0
RESULT_ROOT="${REPO_ROOT}/benchmark_results"
RECORD=1
ALTERNATE_ORDER=0
TELEMETRY=0
CPU_AFFINITY=""

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
        --alternate-order) ALTERNATE_ORDER=1; shift ;;
        --telemetry) TELEMETRY=1; shift ;;
        --cpu-affinity) CPU_AFFINITY="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model-file) MODEL_FILE_OVERRIDE="$2"; shift 2 ;;
        --use-bf16) USE_BF16="$2"; shift 2 ;;
        --gpu-work-list) GPU_WORK_LIST="$2"; shift 2 ;;
        --unet-batch-events) UNET_BATCH_EVENTS="$2"; shift 2 ;;
        --fc-grid-fraction) FC_GRID_FRACTION="$2"; shift 2 ;;
        --unet-grid-fraction) UNET_GRID_FRACTION="$2"; shift 2 ;;
        --profile) PROFILE=1; shift 1 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --no-record) RECORD=0; shift 1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${LABEL}" ]]; then
    echo "ERROR: --label is required" >&2
    usage >&2
    exit 1
fi

case "${USE_BF16}" in
    true) PRECISION=bfloat16 ;;
    false) PRECISION=float32 ;;
    *) echo "ERROR: --use-bf16 must be true or false" >&2; exit 1 ;;
esac

GPU_WORK_LIST=${GPU_WORK_LIST:-$USE_BF16}
case "${GPU_WORK_LIST}" in
    true|false) ;;
    *) echo "ERROR: --gpu-work-list must be true or false" >&2; exit 1 ;;
esac
if [[ $GPU_WORK_LIST == true && $USE_BF16 != true ]]; then
    echo "ERROR: --gpu-work-list true requires --use-bf16 true" >&2; exit 1
fi

if ! [[ "${UNET_BATCH_EVENTS}" =~ ^[0-9]+$ ]] || [[ "${UNET_BATCH_EVENTS}" -lt 1 ]]; then
    echo "ERROR: --unet-batch-events must be a positive integer" >&2
    exit 1
fi

if [[ $BUILD_NAME == /* ]]; then
    BUILD_DIR="$BUILD_NAME"
else
    BUILD_DIR="${REPO_ROOT}/Allen/${BUILD_NAME}"
fi
ALLEN_WRAPPER="${BUILD_DIR}/toolchain/wrapper"
ALLEN_BIN="${BUILD_DIR}/Allen"
RUN_PREFIX=()
if [[ -n $CPU_AFFINITY ]]; then
    taskset -c "$CPU_AFFINITY" true
    RUN_PREFIX+=(taskset -c "$CPU_AFFINITY")
fi
[[ $TELEMETRY == 0 ]] || RUN_PREFIX+=(stdbuf -oL -eL)
MDF="$PVF_MDF"
GEO="$PVF_GEOMETRY"

if [[ ! -x "${ALLEN_WRAPPER}" || ! -x "${ALLEN_BIN}" ]]; then
    echo "ERROR: build does not look runnable: ${BUILD_DIR}" >&2
    exit 1
fi

# The model file comes from the weights/ pipeline (make -C weights verify
# MODEL=<name>); both PVFinder algorithms read it through their "model" property.
if [[ -n "${MODEL_FILE_OVERRIDE}" ]]; then
    [[ "${MODEL_FILE_OVERRIDE}" = /* ]] && MODEL_FILE="${MODEL_FILE_OVERRIDE}" || MODEL_FILE="${REPO_ROOT}/${MODEL_FILE_OVERRIDE}"
else
    MODEL_FILE="${REPO_ROOT}/weights/out/${MODEL}/pvfinder_model.json"
fi
if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "ERROR: ${MODEL_FILE} not found; run: make -C weights verify MODEL=${MODEL}" >&2
    exit 1
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

# The default comparison includes every PVFinder stage and retains the baseline PVs.
# Override PVF_SEQUENCES for FC-only, UNet-only, replacement or hybrid studies.
if [[ -n "${PVF_SEQUENCES:-}" ]]; then
    read -r -a SEQUENCES <<< "${PVF_SEQUENCES}"
else
    SEQUENCES=("hlt1_pp_default" "hlt1_pp_pvs_pvfinder_unet_benchmark")
fi

sequence_label() {
    case "$1" in
        *_pvs_pvfinder_unet_benchmark) echo "pvs" ;;
        *_pvfinder_replace_benchmark) echo "replace" ;;
        *_pvfinder_hybrid_benchmark) echo "hybrid" ;;
        *_pvfinder_unet_benchmark) echo "unet" ;;
        *_pvfinder_benchmark) echo "fc" ;;
        *) echo "baseline" ;;
    esac
}

# Run records are written with the repository venv when it exists; runs.py
# needs only the standard library, so any python3 works.
if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
    RECORD_PY="${REPO_ROOT}/.venv/bin/python3"
else
    RECORD_PY="python3"
fi
RUNS_PY="${REPO_ROOT}/benchmarks/runs.py"

extract_rate() {
    grep -oP '[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?(?=\s+events/s)' "$1" | tail -1
}

write_command() {
    local out="$1"; shift
    printf '%q ' "$@" > "${out}"
    printf '\n' >> "${out}"
}

patch_unet_config() {
    local config="$1"
    python3 - "$config" "$MODEL_FILE" "$PRECISION" "$UNET_BATCH_EVENTS" "$UNET_GRID_FRACTION" <<'PY'
import json
import sys

path, weights, precision, unet_batch_events, unet_grid_fraction = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
unet = data.setdefault("pvfinder_unet", {})
unet["model"] = weights
unet["precision"] = precision
unet["unet_batch_events"] = int(unet_batch_events)
unet["fused_grid_fraction"] = float(unet_grid_fraction)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

patch_fc_config() {
    local config="$1"
    python3 - "$config" "$MODEL_FILE" "$PRECISION" "$UNET_BATCH_EVENTS" "$FC_GRID_FRACTION" "$GPU_WORK_LIST" <<'PY'
import json
import sys

path, weights, precision, unet_batch_events, fc_grid_fraction, gpu_work_list = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
fc = data.setdefault("pvfinder_fc_aggregation", {})
fc["model"] = weights
fc["precision"] = precision
fc["unet_batch_events"] = int(unet_batch_events)
fc["fused_grid_fraction"] = float(fc_grid_fraction)
fc["gpu_work_list"] = gpu_work_list == "true"
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

    # *_pvs_pvfinder_unet_benchmark is also a *_pvfinder_unet_benchmark.
    if [[ "${seq}" == *_pvfinder_unet_benchmark || "${seq}" == *_pvfinder_replace_benchmark \
          || "${seq}" == *_pvfinder_hybrid_benchmark ]]; then
        patch_unet_config "${out_config}"
    fi
    if [[ "${seq}" == *_pvfinder_benchmark || "${seq}" == *_pvfinder_unet_benchmark \
          || "${seq}" == *_pvfinder_replace_benchmark || "${seq}" == *_pvfinder_hybrid_benchmark ]]; then
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
    if [[ $TELEMETRY == 1 ]]; then
        "$RECORD_PY" "$SCRIPT_DIR/benchmark_telemetry.py" mark "$BATCH_DIR" start "$run_idx" "$short" "$log"
    fi

    if [[ "${PROFILE}" -eq 1 ]]; then
        local profile_out="${run_dir}/pvfinder_profile_${short}"
        write_command "${cmd_file}" nsys profile -f true --stats=true -o "${profile_out}" -t cuda \
            "${RUN_PREFIX[@]}" "${ALLEN_WRAPPER}" "${ALLEN_BIN}" --sequence "${config}" \
            "${COMMON_ARGS[@]}" --device "${DEVICE}"
        (
            cd "${run_dir}"
            nsys profile -f true --stats=true -o "${profile_out}" -t cuda \
                "${RUN_PREFIX[@]}" "${ALLEN_WRAPPER}" "${ALLEN_BIN}" --sequence "${config}" \
                "${COMMON_ARGS[@]}" --device "${DEVICE}"
        ) > "${log}" 2>&1
        # Kernel summary as CSV for the run record.
        nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
            --output "${profile_out}" "${profile_out}.nsys-rep" \
            > "${run_dir}/${short}_nsys_stats.log" 2>&1 || \
            echo "WARNING: nsys stats failed for ${seq}; see ${run_dir}/${short}_nsys_stats.log" >&2
    else
        write_command "${cmd_file}" "${RUN_PREFIX[@]}" "${ALLEN_WRAPPER}" "${ALLEN_BIN}" --sequence "${config}" \
            "${COMMON_ARGS[@]}" --device "${DEVICE}"
        (
            cd "${run_dir}"
            "${RUN_PREFIX[@]}" "${ALLEN_WRAPPER}" "${ALLEN_BIN}" --sequence "${config}" \
                "${COMMON_ARGS[@]}" --device "${DEVICE}"
        ) > "${log}" 2>&1
    fi
    if [[ $TELEMETRY == 1 ]]; then
        "$RECORD_PY" "$SCRIPT_DIR/benchmark_telemetry.py" mark "$BATCH_DIR" end "$run_idx" "$short" "$log"
    fi

    local rate
    if ! rate="$(extract_rate "${log}")" || [[ -z "${rate}" ]]; then
        echo "ERROR: could not extract rate for ${seq}; see ${log}" >&2
        exit 1
    fi
    # Allen halves a slice and retries when it exceeds -m; the rate is then
    # measured on smaller slices than requested, so flag it loudly.
    local n_splits
    n_splits="$(grep -c "Insufficient memory to process slice" "${log}" || true)"
    if [[ "${n_splits}" -gt 0 ]]; then
        echo "WARNING: ${seq} hit ${n_splits} slice split(s) from insufficient -m; rate is not at the requested slice size" >&2
        printf '%s\t%s\n' "${short}" "${n_splits}" >> "${run_dir}/slice_splits.tsv"
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
    echo "alternate_order=${ALTERNATE_ORDER}"
    echo "telemetry=${TELEMETRY}"
    echo "cpu_affinity=${CPU_AFFINITY}"
    echo "profile=${PROFILE}"
    echo "model=${MODEL}"
    echo "model_file=${MODEL_FILE}"
    echo "use_bf16=${USE_BF16}"
    echo "gpu_work_list=${GPU_WORK_LIST}"
    echo "precision=${PRECISION}"
    echo "unet_batch_events=${UNET_BATCH_EVENTS}"
    echo "fc_grid_fraction=${FC_GRID_FRACTION}"
    echo "unet_grid_fraction=${UNET_GRID_FRACTION}"
    echo "mdf=${MDF}"
    echo "geometry=${GEO}"
    echo "sequences=${SEQUENCES[*]}"
} > "${BATCH_DIR}/metadata.env"

git -C "${REPO_ROOT}" rev-parse HEAD > "${BATCH_DIR}/git_head.txt"
git -C "${REPO_ROOT}" status --short > "${BATCH_DIR}/git_status_short.txt"
git -C "${REPO_ROOT}" log --oneline -8 --decorate > "${BATCH_DIR}/git_log_oneline.txt"

sha256sum "${MODEL_FILE}" > "${BATCH_DIR}/weights.sha256"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${BATCH_DIR}/nvidia_smi.txt" 2>&1 || true
    nvidia-smi pmon -c 5 > "${BATCH_DIR}/nvidia_smi_pmon.txt" 2>&1 || true
fi

write_command "${BATCH_DIR}/batch_command.cmd" "$0" "${ORIGINAL_ARGS[@]}"

# Environment at the start of the batch (GPU and its other processes, git
# state, build flags, model and weight hashes) for the run record.
RECORDED=0
TELEMETRY_PID=""
if [[ "${RECORD}" -eq 1 ]]; then
    "${RECORD_PY}" "${RUNS_PY}" snapshot "${BATCH_DIR}" --device "${DEVICE}" \
        --build-dir "${BUILD_DIR}" --model "${MODEL}" \
        --model-file "${MODEL_FILE}"
fi
if [[ $TELEMETRY == 1 ]]; then
    "$RECORD_PY" "$SCRIPT_DIR/benchmark_telemetry.py" monitor "$BATCH_DIR" --device "$DEVICE" --parent "$$" &
    TELEMETRY_PID=$!
fi
# Stop the collector and preserve failed/interrupted batches too.
on_exit() {
    local rc=$?
    if [[ -n $TELEMETRY_PID ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" || true
    fi
    if [[ $RECORD == 1 && "${RECORDED}" -eq 0 ]]; then
        local status=failed
        [[ ${rc} -eq 130 || ${rc} -eq 143 ]] && status=interrupted
        "${RECORD_PY}" "${RUNS_PY}" record "${BATCH_DIR}" --status "${status}" || true
    fi
    return "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for run_idx in $(seq 1 "${REPEATS}"); do
    run_dir="${BATCH_DIR}/run_$(printf '%02d' "${run_idx}")"
    mkdir -p "${run_dir}"
    printf 'Running batch %s, repeat %s/%s...\n' "${LABEL}" "${run_idx}" "${REPEATS}"

    rates_file="${run_dir}/rates.tsv"
    : > "${rates_file}"
    ORDER=("${SEQUENCES[@]}")
    if [[ $ALTERNATE_ORDER == 1 && $((run_idx % 2)) == 0 ]]; then
        ORDER=()
        for ((i=${#SEQUENCES[@]}-1; i>=0; i--)); do ORDER+=("${SEQUENCES[i]}"); done
    fi
    printf '%s\n' "${ORDER[@]}" > "$run_dir/sequence_order.txt"
    for seq in "${ORDER[@]}"; do
        run_sequence "${seq}" "${run_dir}" | tee -a "${rates_file}"
    done

done

"${RECORD_PY}" "${REPO_ROOT}/benchmarks/summarize_batch.py" "${BATCH_DIR}"

printf '\nBatch complete: %s\n' "${BATCH_DIR}"
cat "${BATCH_DIR}/summary.md"
if [[ "${RECORD}" -eq 1 ]]; then
    "${RECORD_PY}" "${RUNS_PY}" record "${BATCH_DIR}" --status ok
    RECORDED=1
fi
