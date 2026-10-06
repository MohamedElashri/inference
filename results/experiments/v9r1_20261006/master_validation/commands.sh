#!/usr/bin/env bash
set -euo pipefail
cd /data/home/melashri/iris/inference
root=$PWD
build="$root/Allen/buildv9r1gpu12"
artifacts="$root/benchmark_results/v9r1_master_transfer20261006"
model=unet16_lc4_scnone_asym5_best_bf16
export LD_LIBRARY_PATH="$HOME/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=GPU-45d31422-143d-4539-334e-69eac0bc7b97 "$build/toolchain/wrapper" \
  "$build/test/unit_tests/unit_tests" '[PVFinder],[TensorModel],[AllenCuDNN]' \
  > "$artifacts/unit_tests.txt" 2>&1
cat "$artifacts/unit_tests.txt"
weights/scripts/allen_dump.sh --build "$build" \
  --model-file "$root/weights/out/$model/pvfinder_model.json" \
  --sequence pvfinder_pv_validation --dump-dir "$artifacts/validation" --events 500 --device 2 \
  --set pvfinder_fc_aggregation.precision=bfloat16 --set pvfinder_unet.precision=bfloat16 \
  --set pvfinder_fc_aggregation.gpu_work_list=true \
  --set pvfinder_fc_aggregation.fused_grid_fraction=0.0625 \
  --set pvfinder_unet.fused_grid_fraction=0.125
.venv/bin/python3 benchmarks/runs.py snapshot "$artifacts/validation" --device 2 --build-dir "$build" --model "$model"
for validator in fc unet model features peaks; do
  args=(--dump-dir "$artifacts/validation" --report "$artifacts/validation/validate_$validator.json")
  if [[ $validator == fc || $validator == unet || $validator == model ]]; then
    args+=(--weights "$root/weights/checkpoints/$model.pyt")
  fi
  .venv/bin/python3 "weights/scripts/validate_$validator.py" "${args[@]}" \
    > "$artifacts/validation/validate_$validator.txt" 2>&1
  tail -3 "$artifacts/validation/validate_$validator.txt"
done
.venv/bin/python3 benchmarks/runs.py validation --model "$model" --build-dir "$build" \
  --dump-dir "$artifacts/validation" --device 2 --events 500 --sequence pvfinder_pv_validation \
  --label v9r1-master-gpu-worklist-validation \
  --fc-report "$artifacts/validation/validate_fc.json" --unet-report "$artifacts/validation/validate_unet.json" \
  --model-report "$artifacts/validation/validate_model.json" --features-report "$artifacts/validation/validate_features.json" \
  --peaks-report "$artifacts/validation/validate_peaks.json"
bash benchmarks/pv_comparison.sh --label v9r1_master_physics --model "$model" --bf16 \
  -B buildv9r1gpu12 -d 2 -n 10000 \
  --set pvfinder_fc_aggregation.gpu_work_list=true \
  --set pvfinder_fc_aggregation.fused_grid_fraction=0.0625 --set pvfinder_unet.fused_grid_fraction=0.125 \
  --scan pvfinder_peak_pvfinder.threshold=0.07,0.1
bash benchmarks/benchmark_v9r1_optimized.sh --label v9r1_master_smoke -r 100 --repeats 1
