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
  -B, --build-dir NAME       Allen build directory name under Allen/ (default: buildgpu16chL4gpu)
  -d, --device N             GPU device index (default: 2)
  -t, --threads N            Allen threads / streams (default: 16)
  -n, --events N             Events to process (default: 100)
  -m, --memory MB            Device memory per thread / stream (default: 300)
  -r, --repetitions N        Repetitions per thread / stream (default: 500)
  --repeats N                Number of repeated benchmark runs (default: 3)
  --model NAME               Weights from the weights/ pipeline:
                             weights/out/NAME/{cnn,fc}_weights.bin
                             (default: unet16_lc4_scnone_asym5_final; see make -C weights list)
  --model-file PATH          The model file for both algorithms' "model" property
                             (default: weights/out/<--model>/pvfinder_model.json)
  --use-bf16 BOOL            precision = bfloat16 for pvfinder_fc_aggregation and
                             pvfinder_unet (default: false, i.e. float32)
  --gpu-work-list BOOL      Build FC work lists on the GPU (BF16 only; default false)
  --unet-batch-events N      Set pvfinder_unet.unet_batch_events and
                             pvfinder_fc_aggregation.unet_batch_events together
                             (default: 20); float32 cuDNN batch size in events
  --fc-grid-fraction F       Set pvfinder_fc_aggregation.fused_grid_fraction (default 0.125)
  --unet-grid-fraction F     Set pvfinder_unet.fused_grid_fraction (default 0.25)
  --profile                  Run each sequence under nsys; the record gets
                             the per-sequence kernel summary
  --result-root DIR          Directory for batches (default: benchmark_results)
  --no-record                Do not write a results/runs/ record (smoke tests)
  -h, --help                 Show this help

Example:
  benchmarks/benchmark_pvfinder_batch.sh \
    --label reference_A_fp32_head_d1874d8 \
    -B buildgpu16chL4gpu --model unet16_lc4_scnone_asym5_final \
    -d 2 -t 16 -n 500 -m 500 -r 1000 --repeats 3 --use-bf16 true
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The script lives in benchmarks/; Allen, weights and results are resolved
# from the repository root.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")

LABEL=""
BUILD_NAME="buildgpu16chL4gpu"
DEVICE=2
THREADS=16
EVENTS=100
MEMORY=300
REPS=500
REPEATS=3
MODEL=unet16_lc4_scnone_asym5_final
MODEL_FILE_OVERRIDE=""
USE_BF16=false
GPU_WORK_LIST=false
UNET_BATCH_EVENTS=20
FC_GRID_FRACTION=0.125
UNET_GRID_FRACTION=0.25
PROFILE=0
RESULT_ROOT="${REPO_ROOT}/benchmark_results"
RECORD=1

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
        --model) MODEL="$2"; shift 2 ;;
        --model-file) MODEL_FILE_OVERRIDE="$2"; shift 2 ;;
        --cnn-weights|--fc-weights)
            echo "ERROR: $1 was replaced by --model-file (one pvfinder_model.json for both algorithms)" >&2; exit 1 ;;
        --use-bf16) USE_BF16="$2"; shift 2 ;;
        --gpu-work-list) GPU_WORK_LIST="$2"; shift 2 ;;
        --unet-batch-events) UNET_BATCH_EVENTS="$2"; shift 2 ;;
        --fc-grid-fraction) FC_GRID_FRACTION="$2"; shift 2 ;;
        --unet-grid-fraction) UNET_GRID_FRACTION="$2"; shift 2 ;;
        --use-fp16|--use-cuda-graph|--use-fused-cbr|--fwd-algo-ws-budget-mb|--use-fused-rcbn3|\
        --use-fused-bias-relu-pool|--use-merged-up1|--l6a-m|--use-nonatomic-l6a-reduce|\
        --use-warp-parallel-reduce|--fc-chunk-size|--use-fused-bias-relu-reduce|--skip-redundant-memset|\
        --use-grid-stride-reduce|--fc-single-hidden-layer|--l1-l5-hidden-width|--l6a-active-channels|\
        --use-precomputed-csr-offset|--safe-avg-entries-per-event|--skip-empty-intervals|--fc-fused|\
        --canonical-track-order|--fc-fused-per-warp|--fc-hidden-dtype|--l6a-dtype|--min-interval-tracks|\
        --unet-input-dtype|--bf16-layout|--unet-fused-kernel|--unet-input-layout)
            echo "ERROR: $1 was removed with the experimental PVFinder paths (2026-09-27)" >&2; exit 1 ;;
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

BUILD_DIR="${REPO_ROOT}/Allen/${BUILD_NAME}"
ALLEN_WRAPPER="${BUILD_DIR}/toolchain/wrapper"
ALLEN_BIN="${BUILD_DIR}/Allen"
MDF="${REPO_ROOT}/Allen/input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf"
GEO="${REPO_ROOT}/Allen/input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry"

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

# Default triple: HLT1 alone | HLT1+FC | HLT1+FC+UNet. Two more roles are
# recognised by name: *_pvs_pvfinder_unet_benchmark ("pvs": + peak finding and
# the PV fit) and *_pvfinder_replace_benchmark ("replace": PVFinder's vertices
# replace the beamline PV finder's for all of HLT1), and
# *_pvfinder_hybrid_benchmark ("hybrid": as replace, the beamline PV finder's
# seeds kept outside PVFinder's z range).
# Override with PVF_SEQUENCES="<baseline_seq> <fc_seq> <unet_seq>" to benchmark
# against a different HLT1 sequence (e.g. a reduced-work HLT1). Names must end
# in the same _pvfinder_benchmark / _pvfinder_unet_benchmark suffixes so the
# config patchers and the summary labels still recognise the FC and UNet rows.
if [[ -n "${PVF_SEQUENCES:-}" ]]; then
    read -r -a SEQUENCES <<< "${PVF_SEQUENCES}"
else
    SEQUENCES=(
        "hlt1_pp_default"
        "hlt1_pp_pvfinder_benchmark"
        "hlt1_pp_pvfinder_unet_benchmark"
    )
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
    grep -oP '[0-9]+\.[0-9]+(?=\s+events/s)' "$1" | tail -1
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
        # Kernel summary as CSV for the run record.
        nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
            --output "${profile_out}" "${profile_out}.nsys-rep" \
            > "${run_dir}/${short}_nsys_stats.log" 2>&1 || \
            echo "WARNING: nsys stats failed for ${seq}; see ${run_dir}/${short}_nsys_stats.log" >&2
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
if [[ "${RECORD}" -eq 1 ]]; then
    "${RECORD_PY}" "${RUNS_PY}" snapshot "${BATCH_DIR}" --device "${DEVICE}" \
        --build-dir "${BUILD_DIR}" --model "${MODEL}" \
        --model-file "${MODEL_FILE}"
    # A batch that stops early still gets a record, with whatever repeats finished.
    on_exit() {
        local rc=$?
        if [[ "${RECORDED}" -eq 0 ]]; then
            local status=failed
            [[ ${rc} -eq 130 || ${rc} -eq 143 ]] && status=interrupted
            "${RECORD_PY}" "${RUNS_PY}" record "${BATCH_DIR}" --status "${status}" || true
        fi
    }
    trap on_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
fi

for run_idx in $(seq 1 "${REPEATS}"); do
    run_dir="${BATCH_DIR}/run_$(printf '%02d' "${run_idx}")"
    mkdir -p "${run_dir}"
    printf 'Running batch %s, repeat %s/%s...\n' "${LABEL}" "${run_idx}" "${REPEATS}"

    rates_file="${run_dir}/rates.tsv"
    : > "${rates_file}"
    for seq in "${SEQUENCES[@]}"; do
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
