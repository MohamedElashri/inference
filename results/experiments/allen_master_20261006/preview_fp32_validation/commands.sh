#!/usr/bin/env bash
set -euo pipefail
cd /data/home/melashri/iris/inference
root=$PWD
build="/tmp/pvfinder-current-master20261006/buildgpu12"
artifacts="$root/benchmark_results/allen_master_rebase20261006"
model=unet16_lc4_scnone_asym5_best
export LD_LIBRARY_PATH="$HOME/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
weights/scripts/allen_dump.sh --build "$build" \
  --model-file "$root/weights/out/$model/pvfinder_model.json" \
  --sequence pvfinder_pv_validation --dump-dir "$artifacts/validation_fp32" --events 500 --device 2 \
  --set pvfinder_fc_aggregation.precision=float32 --set pvfinder_unet.precision=float32 \
  --set pvfinder_fc_aggregation.gpu_work_list=false \
  --set pvfinder_fc_aggregation.fused_grid_fraction=0.0625 \
  --set pvfinder_unet.fused_grid_fraction=0.125
.venv/bin/python3 benchmarks/runs.py snapshot "$artifacts/validation_fp32" --device 2 --build-dir "$build" --model "$model"
for validator in fc unet model features peaks; do
  args=(--dump-dir "$artifacts/validation_fp32" --report "$artifacts/validation_fp32/validate_$validator.json")
  if [[ $validator == fc || $validator == unet || $validator == model ]]; then
    args+=(--weights "$root/weights/checkpoints/$model.pyt")
  fi
  .venv/bin/python3 "weights/scripts/validate_$validator.py" "${args[@]}" \
    > "$artifacts/validation_fp32/validate_$validator.txt" 2>&1
  tail -3 "$artifacts/validation_fp32/validate_$validator.txt"
done
.venv/bin/python3 benchmarks/runs.py validation --model "$model" --build-dir "$build" \
  --dump-dir "$artifacts/validation_fp32" --device 2 --events 500 --sequence pvfinder_pv_validation \
  --label allen-master-preview-fp32-validation \
  --fc-report "$artifacts/validation_fp32/validate_fc.json" --unet-report "$artifacts/validation_fp32/validate_unet.json" \
  --model-report "$artifacts/validation_fp32/validate_model.json" --features-report "$artifacts/validation_fp32/validate_features.json" \
  --peaks-report "$artifacts/validation_fp32/validate_peaks.json"
