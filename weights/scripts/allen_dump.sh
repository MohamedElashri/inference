#!/usr/bin/env bash
# Run Allen on one fixed slice with PVFinder's validation dumps enabled.
#
# Generates the sequence configuration with PVFINDER_WEIGHTS_DIR pointing at the
# model's weight files, switches on pvfinder_fc_aggregation.dump_validation (and
# pvfinder_unet.dump_validation when the sequence has the UNet), and runs one
# stream for two repetitions of the same slice. The dumps are read by
# validate_fc.py and validate_unet.py.
#
# Usage:
#   allen_dump.sh --build ALLEN_BUILD_DIR --weights-dir DIR --sequence NAME --dump-dir DIR
#                 [--events N] [--memory MB] [--device N]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-${REPO}/.venv/bin/python3}"
MDF="${REPO}/Allen/input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf"
GEO="${REPO}/Allen/input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry"

BUILD="" WDIR="" SEQ="" DUMP="" EVENTS=500 MEMORY=1000 DEVICE=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD="$2"; shift 2 ;;
        --weights-dir) WDIR="$2"; shift 2 ;;
        --sequence) SEQ="$2"; shift 2 ;;
        --dump-dir) DUMP="$2"; shift 2 ;;
        --events) EVENTS="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for v in BUILD WDIR SEQ DUMP; do
    [[ -n "${!v}" ]] || { echo "missing --${v,,} (see the header of $0)" >&2; exit 2; }
done
[[ -x "${BUILD}/Allen" ]] || { echo "no Allen binary in ${BUILD}: run 'make build'" >&2; exit 1; }
for f in cnn_weights.bin fc_weights.bin; do
    [[ -f "${WDIR}/${f}" ]] || { echo "missing ${WDIR}/${f}: run 'make convert'" >&2; exit 1; }
done

rm -rf "${DUMP}"
mkdir -p "${DUMP}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

if ! (cd "${tmp}" && PVFINDER_WEIGHTS_DIR="$(cd "${WDIR}" && pwd)" "${BUILD}/toolchain/wrapper" bash -c '
        export ALLEN_BUILD_DIR="$1"
        export PYTHONPATH="$1/code_generation/sequences:${PYTHONPATH:-}"
        python3 "$1/code_generation/sequences/AllenCore/gen_allen_json.py" \
            --no-register-keys --seqpath "$1/code_generation/sequences/AllenSequences/$2.py"
    ' bash "${BUILD}" "${SEQ}") > "${DUMP}/generate_config.log" 2>&1; then
    echo "sequence generation failed, see ${DUMP}/generate_config.log" >&2
    tail -5 "${DUMP}/generate_config.log" >&2
    exit 1
fi

"${PY}" - "${tmp}/Sequence.json" "${DUMP}/config.json" "${DUMP}" <<'EOF'
import json, sys
src, dst, dump = sys.argv[1:]
cfg = json.load(open(src))
for alg in ("pvfinder_fc_aggregation", "pvfinder_unet"):
    if alg in cfg:
        cfg[alg]["dump_validation"] = dump
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
