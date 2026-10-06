#!/usr/bin/env bash
# Run Allen on one fixed slice with PVFinder's validation dumps enabled.
#
# Generates the sequence configuration, points pvfinder_fc_aggregation and
# pvfinder_unet at the model file (their "model" property), switches on
# dump_validation for every PVFinder algorithm in the sequence (FC, UNet, and
# pvfinder_peak when the sequence finds PVs), and runs one stream for two
# repetitions of the same slice. The dumps are read by the validators in this
# directory (validate_fc, validate_unet, validate_model, validate_features,
# validate_peaks).
#
# --set ALG.PROPERTY=VALUE (repeatable) overrides one algorithm property in the
# generated configuration, e.g. --set pvfinder_unet.precision=bfloat16; VALUE is
# read as JSON when it parses, as a string otherwise.
#
# Usage:
#   allen_dump.sh --build ALLEN_BUILD_DIR --model-file FILE --sequence NAME --dump-dir DIR
#                 [--events N] [--memory MB] [--device N] [--set ALG.PROP=VALUE]...
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/benchmarks/defaults.sh"
PY="${PY:-${REPO}/.venv/bin/python3}"
MDF="${REPO}/Allen/input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf"
GEO="${REPO}/Allen/input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry"

BUILD="$PVF_BUILD_DIR" MODEL_FILE="" SEQ="" DUMP="" EVENTS=$PVF_EVENTS MEMORY=1000 DEVICE=$PVF_DEVICE
SETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD="$2"; shift 2 ;;
        --model-file) MODEL_FILE="$2"; shift 2 ;;
        --sequence) SEQ="$2"; shift 2 ;;
        --dump-dir) DUMP="$2"; shift 2 ;;
        --events) EVENTS="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --set) SETS+=("$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for v in BUILD MODEL_FILE SEQ DUMP; do
    [[ -n "${!v}" ]] || { echo "missing --${v,,} (see the header of $0)" >&2; exit 2; }
done
[[ -x "${BUILD}/Allen" ]] || { echo "no Allen binary in ${BUILD}: run 'make build'" >&2; exit 1; }
[[ -f "${MODEL_FILE}" ]] || { echo "missing ${MODEL_FILE}: run 'make convert'" >&2; exit 1; }
MODEL_FILE="$(cd "$(dirname "${MODEL_FILE}")" && pwd)/$(basename "${MODEL_FILE}")"

rm -rf "${DUMP}"
mkdir -p "${DUMP}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

if ! (cd "${tmp}" && "${BUILD}/toolchain/wrapper" bash -c '
        export ALLEN_BUILD_DIR="$1"
        export PYTHONPATH="$1/code_generation/sequences:${PYTHONPATH:-}"
        python3 "$1/code_generation/sequences/AllenCore/gen_allen_json.py" \
            --no-register-keys --seqpath "$1/code_generation/sequences/AllenSequences/$2.py"
    ' bash "${BUILD}" "${SEQ}") > "${DUMP}/generate_config.log" 2>&1; then
    echo "sequence generation failed, see ${DUMP}/generate_config.log" >&2
    tail -5 "${DUMP}/generate_config.log" >&2
    exit 1
fi

"${PY}" - "${tmp}/Sequence.json" "${DUMP}/config.json" "${DUMP}" "${MODEL_FILE}" "${SETS[@]}" <<'EOF'
import json, sys
src, dst, dump, model, *sets = sys.argv[1:]
cfg = json.load(open(src))
for alg in cfg:
    if alg in ("pvfinder_fc_aggregation", "pvfinder_unet"):
        cfg[alg]["dump_validation"] = dump
        cfg[alg]["model"] = model
    elif alg.startswith("pvfinder_peak"):
        cfg[alg]["dump_validation"] = dump
for item in sets:
    key, _, raw = item.partition("=")
    alg, _, prop = key.partition(".")
    if not prop or alg not in cfg:
        sys.exit(f"--set {item}: expected ALG.PROPERTY=VALUE with ALG in the sequence")
    try:
        cfg[alg][prop] = json.loads(raw)
    except json.JSONDecodeError:
        cfg[alg][prop] = raw
    print(f"override: {alg}.{prop} = {cfg[alg][prop]!r}")
json.dump(cfg, open(dst, "w"), indent=2, sort_keys=True)
EOF

if ! (cd "${DUMP}" && "${BUILD}/toolchain/wrapper" "${BUILD}/Allen" --sequence "${DUMP}/config.json" \
        --mdf "${MDF}" -g "${GEO}" -n "${EVENTS}" -m "${MEMORY}" -r 2 -t 1 --device "${DEVICE}") \
        > "${DUMP}/allen.log" 2>&1; then
    echo "Allen failed, see ${DUMP}/allen.log" >&2
    tail -20 "${DUMP}/allen.log" >&2
    exit 1
fi
grep -h "validation dump written\|Validation dump written" "${DUMP}/allen.log" || {
    echo "Allen ran but wrote no validation dump, see ${DUMP}/allen.log" >&2; exit 1; }
