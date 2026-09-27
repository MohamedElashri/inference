# Run records


Every Allen run made through the repository's tooling leaves one JSON file in
`results/runs/`, and these files are tracked in git. A record says what ran
(model, weights, build flags, algorithm configuration, git state), where it
ran (host, GPU, driver, other processes on that GPU) and what came out
(throughput per repeat and medians, the nsys kernel summary, validation
metrics). Bulky raw output (logs, effective configs, `.nsys-rep` files,
dumps) stays in the ignored `benchmark_results/<id>/` or
`weights/out/<model>/dump*/` directory. The record names that directory
under `artifacts`.

| Producer | `kind` |
|---|---|
| `benchmarks/benchmark_pvfinder_batch.sh` | `benchmark` (or `profile` with `--profile`) |
| `make -C weights dump validate` | `validation` |
| `benchmarks/pv_comparison.sh` | `physics` (PVFinder vs the beamline PV finder on MC, one point per configuration) |
| `benchmarks/runs.py import <batch_dir>` | an older batch, marked `"imported": true` |

Failed and interrupted benchmark batches are recorded too (`status` is
`failed` or `interrupted`), with whatever repeats finished. Pass
`--no-record` to the benchmark script for throwaway smoke tests.

## File names

`runs/<YYYYmmdd_HHMMSS>_<gpu>_<model>_<label>.json`, for example
`20260918_073701_rtx3090_unet16-lc4-scnone-asym5-final_production.json`. The
timestamp is the batch's start time, and the `id` inside the file is the
batch directory's name.

## Querying

```bash
benchmarks/runs.py list                              # all successful runs, one line each
benchmarks/runs.py list --model asym5 --gpu 3090 --kind benchmark
benchmarks/runs.py show 20260918_073701              # id, unique prefix or substring
benchmarks/runs.py show 20260918_073701 --json
benchmarks/runs.py compare <run_a> <run_b>           # throughput ratio + every setting that differs
```

## Schema (`pvfinder-run/1`)

| Field | Content |
|---|---|
| `kind`, `id`, `label`, `status`, `imported` | what the record is |
| `started_at`, `finished_at`, `duration_s`, `command` | when it ran, and how it was invoked |
| `host` | hostname, OS, CPU |
| `gpu` | name, UUID, driver, compute capability, memory, max SM clock, power limit, `processes_at_start` |
| `git` | `head`, `branch`, `dirty`, `dirty_files`, `diff_sha256` (hash of the uncommitted diff, excluding `results/`) |
| `build` | build directory, CMake cache (`PVFINDER_UNET_N_FEAT`, `PVFINDER_UNET_N_BATCH_CHANNELS`, cuDNN/cuBLAS, CUDA and GCC versions, `CUDA_ARCH`), library mtime, `sources_newer_than_build` |
| `model` | catalog row from `weights/models.tsv`, SHA-256 of the checkpoint and both weight files, `verified` |
| `workload` | input MDF, geometry, `events` (-n), `memory_mb` (-m), `repetitions` (-r), `threads` (-t), `repeats`, `device`, sequences |
| `options` | every benchmark-script option (precision, fusions, batch sizes, ...) |
| `config` | the `pvfinder_*` algorithm blocks of each sequence's effective Allen configuration, the ground truth for what Allen ran |
| `results` | benchmark: per-repeat events/s for `baseline`, `fc`, `unet`, overheads, slice splits, and medians with baseline spread. Validation: the validators' JSON reports: `validate_fc.py`, `validate_unet.py`, `validate_model.py`, `validate_features.py` and `validate_peaks.py` (`fc`, `unet`, `model`, `features`, `peaks`) |
| `profile` | with `--profile`: per sequence, the nsys `cuda_gpu_kern_sum` merged over repeats (median), top 25 kernels plus every PVFinder/cuDNN/cuBLAS kernel |
| `artifacts` | where the raw output lives on the machine that ran it |

`build.sources_newer_than_build` lists tracked Allen sources modified after
`libAllenLib.so` was linked. It is based on file mtimes, so a checkout can put
files on the list that did not change. An empty list together with a clean
tree means the binary was built from the recorded commit.

## Conventions

- Commit records together with the change they measure, or on their own right after the run. Do not edit a record by hand; rerun instead.
- Numbers quoted in notes, slides or reviews should cite the record `id`.
- The production operating point is `-n 500 -m 500 -r 1000 -t 16`, on GPU 2 (RTX 3090) of this host.
