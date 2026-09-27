#!/usr/bin/env bash
# Primary vertices from PVFinder against Allen's beamline PV finder, on MC.
#
# Runs the Allen sequence pvfinder_pv_validation (VELO tracks, the beamline PV
# finder, and PVFinder's KDE -> pvfinder_peak -> the same beamline association
# and fit) over the input file, single stream, then
#   weights/scripts/compare_pvs.py    efficiency / false rate / resolution of
#                                     both, all z and in PVFinder's z range,
#                                     and the event-by-event comparison;
#   weights/scripts/validate_peaks.py pvfinder_peak against pv-finder's peak
#                                     finder on the first slice's KDE,
# and writes one run record (kind "physics", results/runs/) with every point.
# Raw outputs stay in benchmark_results/<stamp>_<label>/<point>/.
#
# Usage:
#   benchmarks/pv_comparison.sh --label LABEL [--model NAME] [--bf16]
#       [--set ALG.PROP=VALUE]... [--scan ALG.PROP=V1,V2,...]...
#       [-B BUILD_DIR] [-d DEVICE] [-n EVENTS] [--no-record]
#
#   --bf16          the BF16 path (precision = bfloat16 for the FC and the UNet),
#                   as benchmark_pvfinder_batch.sh --use-bf16
#   --set           one property for every point (VALUE read as JSON if it parses)
#   --scan          one point per value; several --scan give their product
#   -n EVENTS       0 (default) = the whole input file
#
# Example, the peak finder's thresholds:
#   benchmarks/pv_comparison.sh --label peak_scan \
#       --scan pvfinder_peak_pvfinder.threshold=0.035,0.07 \
#       --scan pvfinder_peak_pvfinder.integral_threshold=0.5,0.7,1.0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${REPO}/.venv/bin/python3}"
MDF="${REPO}/Allen/input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf"
GEO="${REPO}/Allen/input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry"
SEQ=pvfinder_pv_validation

LABEL="" MODEL=unet16_lc4_scnone_asym5_final BUILD=build16chL4finalgpu DEVICE=2 EVENTS=0 RECORD=1
SETS=() SCANS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --bf16) SETS+=(pvfinder_unet.precision=bfloat16 pvfinder_fc_aggregation.precision=bfloat16); shift ;;
        --set) SETS+=("$2"); shift 2 ;;
        --scan) SCANS+=("$2"); shift 2 ;;
        -B|--build-dir) BUILD="$2"; shift 2 ;;
        -d|--device) DEVICE="$2"; shift 2 ;;
        -n|--events) EVENTS="$2"; shift 2 ;;
        --no-record) RECORD=0; shift ;;
        -h|--help) sed -n '2,31s/^# \{0,1\}//p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
[[ -n "${LABEL}" ]] || { echo "--label is required" >&2; exit 2; }
B="${REPO}/Allen/${BUILD}"
[[ -x "${B}/Allen" ]] || { echo "no Allen binary in ${B}" >&2; exit 1; }
MODEL_FILE="${REPO}/weights/out/${MODEL}/pvfinder_model.json"
[[ -f "${MODEL_FILE}" ]] || { echo "missing ${MODEL_FILE}: make -C weights verify MODEL=${MODEL}" >&2; exit 1; }

BATCH="${REPO}/benchmark_results/$(date +%Y%m%d_%H%M%S)_${LABEL}"
mkdir -p "${BATCH}"

# One configuration: the sequence, once per batch.
(cd "${BATCH}" && "${B}/toolchain/wrapper" bash -c '
    export PYTHONPATH="$1/code_generation/sequences:${PYTHONPATH:-}"
    python3 "$1/code_generation/sequences/AllenCore/gen_allen_json.py" --no-register-keys \
        --seqpath "$1/code_generation/sequences/AllenSequences/$2.py"' bash "${B}" "${SEQ}") \
    > "${BATCH}/generate_config.log" 2>&1 || {
    echo "sequence generation failed, see ${BATCH}/generate_config.log" >&2; exit 1; }

# Points: the product of the --scan lists, as "name<TAB>ALG.PROP=V ..." lines.
mapfile -t POINTS < <("${PY}" - "${SCANS[@]}" <<'EOF'
import itertools, sys
axes = []
for scan in sys.argv[1:]:
    key, _, values = scan.partition("=")
    axes.append([(key, v) for v in values.split(",")])
for combo in itertools.product(*axes):
    name = "_".join(f"{k.rpartition('.')[2]}{v}" for k, v in combo) or "default"
    print(name + "\t" + " ".join(f"{k}={v}" for k, v in combo))
EOF
)

RUN_DIRS=()
for point in "${POINTS[@]}"; do
    name="${point%%$'\t'*}"; read -r -a point_sets <<< "${point#*$'\t'}"
    d="${BATCH}/${name}"; mkdir -p "${d}"
    "${PY}" - "${BATCH}/Sequence.json" "${d}/Sequence.json" "${d}" "${MODEL_FILE}" "${SETS[@]}" "${point_sets[@]}" <<'EOF'
import json, sys
src, dst, dump, model, *sets = sys.argv[1:]
cfg = json.load(open(src))
cfg["pvfinder_fc_aggregation"]["model"] = model
cfg["pvfinder_unet"]["model"] = model
cfg["pvfinder_unet"]["dump_validation"] = dump          # first slice's KDE and seeds,
cfg["pvfinder_peak_pvfinder"]["dump_validation"] = dump  # for validate_peaks.py
for item in sets:
    key, _, raw = item.partition("=")
    alg, _, prop = key.partition(".")
    if not prop or alg not in cfg:
        sys.exit(f"{item}: expected ALG.PROPERTY=VALUE with ALG in the sequence")
    try:
        cfg[alg][prop] = json.loads(raw)
    except json.JSONDecodeError:
        cfg[alg][prop] = raw
json.dump(cfg, open(dst, "w"), indent=2, sort_keys=True)
EOF
    echo "point ${name}: ${SETS[*]} ${point_sets[*]}"
    (cd "${d}" && "${B}/toolchain/wrapper" "${B}/Allen" --sequence Sequence.json --mdf "${MDF}" -g "${GEO}" \
        -n "${EVENTS}" -m 2000 -t 1 -r 1 --device "${DEVICE}") > "${d}/allen.log" 2>&1 || {
        echo "Allen failed, see ${d}/allen.log" >&2; tail -20 "${d}/allen.log" >&2; exit 1; }
    "${PY}" "${REPO}/weights/scripts/validate_peaks.py" --dump-dir "${d}" --report "${d}/validate_peaks.json"
    "${PY}" "${REPO}/weights/scripts/compare_pvs.py" --dir "${d}" --report "${d}/compare.json" | tee "${d}/compare.txt"
    RUN_DIRS+=(--run-dir "${d}")
done

if [[ "${RECORD}" -eq 1 ]]; then
    "${PY}" "${REPO}/benchmarks/runs.py" physics --model "${MODEL}" --build-dir "${B}" --device "${DEVICE}" \
        --label "${LABEL}" "${RUN_DIRS[@]}"
fi
