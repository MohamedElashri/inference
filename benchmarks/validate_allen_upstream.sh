#!/usr/bin/env bash
# Validate a prepared upstream fork and write an importable verification record.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 2 ]] || { echo "Usage: $0 SOURCE_DIR RESULT_DIR (inside results/experiments/)" >&2; exit 2; }
source_dir=$(realpath "$1")
result_dir=$(realpath -m "$2")
case "$result_dir" in
    "$REPO_ROOT"/results/experiments/*) ;;
    *) echo "RESULT_DIR must be inside results/experiments/" >&2; exit 2 ;;
esac
[[ ! -e $result_dir ]] || { echo "Refusing to overwrite $result_dir" >&2; exit 2; }
source "$REPO_ROOT/benchmarks/defaults.sh"
build_dir=${BUILD_DIR:-$source_dir/build}
[[ -x $build_dir/Allen && -x $build_dir/test/unit_tests/unit_tests ]] || {
    echo "Build Allen and its unit tests first: $build_dir" >&2; exit 2;
}
device=$PVF_DEVICE
gpu_uuid=${GPU_UUID:-$(nvidia-smi -i "$device" --query-gpu=uuid --format=csv,noheader)}
model=$PVF_MODEL
mkdir -p "$REPO_ROOT/benchmark_results"
raw=$(mktemp -d "$REPO_ROOT/benchmark_results/allen-upstream-validation-XXXXXX")
mkdir -p "$result_dir/numerical"
cd "$REPO_ROOT"
export LD_LIBRARY_PATH="${CUDNN_ROOT:-$HOME/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
CUDA_VISIBLE_DEVICES="$gpu_uuid" "$build_dir/toolchain/wrapper" \
    "$build_dir/test/unit_tests/unit_tests" '[PVFinder],[TensorModel],[AllenCuDNN]' > "$result_dir/unit_tests.txt" 2>&1
weights/scripts/allen_dump.sh --build "$build_dir" \
    --model-file "$REPO_ROOT/weights/out/$model/pvfinder_model.json" --sequence pvfinder_pv_validation \
    --dump-dir "$raw/numerical" --events 500 --device "$device" \
    --set pvfinder_fc_aggregation.precision=bfloat16 --set pvfinder_unet.precision=bfloat16 \
    --set pvfinder_fc_aggregation.gpu_work_list=true \
    --set "pvfinder_fc_aggregation.fused_grid_fraction=$PVF_FC_GRID_FRACTION" --set "pvfinder_unet.fused_grid_fraction=$PVF_UNET_GRID_FRACTION"
"$REPO_ROOT/.venv/bin/python3" benchmarks/runs.py snapshot "$raw/numerical" --device "$device" \
    --build-dir "$build_dir" --model "$model"
for validator in fc unet model features peaks; do
    args=(--dump-dir "$raw/numerical" --report "$result_dir/numerical/validate_$validator.json")
    if [[ $validator == fc || $validator == unet || $validator == model ]]; then
        args+=(--weights "$REPO_ROOT/weights/checkpoints/$model.pyt")
    fi
    "$REPO_ROOT/.venv/bin/python3" "weights/scripts/validate_$validator.py" "${args[@]}" > "$raw/validate_$validator.txt" 2>&1
done
label="upstream_$(basename "$raw")"
bash benchmarks/pv_comparison.sh --label "$label" --model "$model" --bf16 -B "$build_dir" -d "$device" -n 10000 \
    --set pvfinder_fc_aggregation.gpu_work_list=true \
    --set "pvfinder_fc_aggregation.fused_grid_fraction=$PVF_FC_GRID_FRACTION" --set "pvfinder_unet.fused_grid_fraction=$PVF_UNET_GRID_FRACTION" \
    --scan pvfinder_peak_pvfinder.threshold=0.07,0.1
"$REPO_ROOT/.venv/bin/python3" - "$source_dir" "$result_dir" "$raw" "$label" "$build_dir" <<'PY'
import glob, json, pathlib, re, shutil, subprocess, sys
source, result, raw, label, build = sys.argv[1:]
root = pathlib.Path.cwd(); result = pathlib.Path(result)
sys.path.insert(0, str(root / 'benchmarks'))
from runs import build_info
build_record = build_info(build)
assert pathlib.Path(build_record['cmake']['CMAKE_HOME_DIRECTORY']).resolve() == pathlib.Path(source).resolve()
assert build_record['source_git']['dirty'] is False
assert not build_record['sources_newer_than_build']
batch, = glob.glob(str(root / 'benchmark_results' / ('*_' + label)))
physics = {}
for threshold in ('0.07', '0.1'):
    point = pathlib.Path(batch) / ('threshold' + threshold)
    target = result / 'physics' / ('threshold' + threshold); target.mkdir(parents=True)
    for name in ('compare.json', 'validate_peaks.json'):
        shutil.copy2(point / name, target / name)
    compare = json.loads((point / 'compare.json').read_text())
    # The benchmark MDF is the fixed 10,000-event 2024 MC sample.
    assert compare['events'] == 10000
    for algorithm in ('beamline', 'pvfinder'):
        assert compare['all_z'][algorithm]['mc'] == 53447
        assert compare['in_range'][algorithm]['mc'] == 51012
    physics[threshold] = str(target.relative_to(result))
validators = {name: json.loads((result / 'numerical' / ('validate_' + name + '.json')).read_text())['status']
              for name in ('fc', 'unet', 'model', 'features', 'peaks')}
assert set(validators.values()) == {'PASS'}
match = re.search(r'All tests passed \((\d+) assertions in (\d+) test cases\)', (result / 'unit_tests.txt').read_text())
assert match
proof = {'schema': 'allen-upstream-validation/1', 'status': 'PASS',
         'fork_commit': subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip(),
         'build': build_record, 'unit_tests': {'assertions': int(match[1]), 'cases': int(match[2])},
         'validators': validators, 'physics': physics, 'raw_artifacts': str(pathlib.Path(raw).relative_to(root))}
(result / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')
print('Validated import record:', result / 'verification.json')
PY
