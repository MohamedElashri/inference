# PVFinder inference in Allen

[Allen](https://gitlab.cern.ch/lhcb/Allen) is LHCb's GPU trigger (HLT1).
This repository integrates PVFinder into its event reconstruction. The FC
network converts VELO tracks into features in 10 mm intervals along the beam;
the UNet predicts a density whose peaks seed primary vertices.

| Directory | Contents |
|---|---|
| `Allen/` | Allen source, including `device/pvfinder/` |
| `weights/` | Checkpoint conversion, model catalog and numerical validators |
| `pvfinder_pytorch/` | PyTorch reference model |
| `benchmarks/` | Build, throughput, physics and update tools |
| `results/runs/` | Tracked run records with configurations and provenance |
| `results/experiments/` | Experiment decisions and supporting evidence |

## Build

Requirements: NVIDIA GPU with compute capability at least 8.0, CVMFS
(`/cvmfs/lhcb.cern.ch`), cuDNN under `~/local/cuda`, and Python with NumPy and
PyTorch in `.venv/`. The default throughput input and geometry are read from
CVMFS. The MC physics reference belongs in `Allen/input/` and stays outside Git.

```bash
bash benchmarks/build_allen.sh
```

The build and GPU unit tests use `Allen/build/`, CUDA 12.8, LCG 108c, cuDNN,
SM 8.6, 16 feature maps and 4 latent channels. `./ballen` calls this same script.
Set `CUDA_ARCH` for a different GPU. `--help` lists overrides; `--dry-run`
prints the commands.

Dependency sources are read from `.cache/allen-dependencies/{LHCb,Gaudi,ParamFiles}`.
On a new checkout, set `DEPENDENCY_CACHE` to a source cache or set `LHCBROOT`,
`GAUDIROOT` and `PARAMFILESROOT` individually. `BUILD_DIR` (or
`ALLEN_BUILD_DIR`), `JOBS` and `CUDNN_ROOT` override the local defaults.

## Weights and numerical validation

```bash
make -C weights verify
make -C weights dump validate
```

The default model is the BF16 export of the best asymmetry-5 checkpoint.
The weights pipeline uses the same build directory and build script. The
five validators check track features, FC arithmetic, UNet output, the full
network and peak finding against independent references. A failed check
returns a non-zero exit and is retained in the run record.

For FP32, select the corresponding checkpoint:

```bash
make -C weights dump validate MODEL=unet16_lc4_scnone_asym5_best PRECISION=float32
```

See [weights/README.md](weights/README.md) for the model format and validation
criteria. `make -C weights list` lists available checkpoints.

## Throughput

Select a new input and its matching standalone geometry together:

```bash
MDF_FILE=/path/to/sample.mdf GEOMETRY_DIR=/path/to/matching_geometry \
    bash benchmarks/benchmark_pvfinder_batch.sh --label new_sample
```

Throughput and numerical checks default to RTA's unbiased real-data sample
`hlt1_input_data_2025_run_321834_mdf`: full machine, average mu 5.26, plus
Deuterium. The matching geometry is the pair used by Allen's throughput CI.
`MDF_FILE` and `GEOMETRY_DIR` override both throughput and numerical checks.
See the [sample checks](results/experiments/rta_2025_321834_20261006/README.md).

```bash
bash benchmarks/benchmark_pvfinder_batch.sh --label production
```

Defaults: GPU 2 (RTX 3090 on this host), 500 events, 500 MB per stream,
1,000 repetitions, 16 streams and five paired repeats. The comparison is plain
HLT1 versus HLT1 with the full PVFinder shadow chain: FC, UNet, peak finding
and PV fitting, with the baseline PV reconstruction retained.

Both stages use BF16. The GPU work list is enabled; FC and UNet grid fractions
are 0.0625 and 0.125. [Shared defaults](benchmarks/defaults.sh) define the
operating point. Each run sets absolute model paths, saves effective
configurations and writes a record to `results/runs/`. Raw logs and profiles
stay in the ignored `benchmark_results/` directory.

```bash
# One paired repeat
bash benchmarks/benchmark_pvfinder_batch.sh --label quick_check --repeats 1
# FP32 comparison
bash benchmarks/benchmark_pvfinder_batch.sh --label fp32 \
    --model unet16_lc4_scnone_asym5_best --use-bf16 false
# FC and UNet stage costs
PVF_SEQUENCES="hlt1_pp_default hlt1_pp_pvfinder_benchmark hlt1_pp_pvfinder_unet_benchmark" \
    bash benchmarks/benchmark_pvfinder_batch.sh --label stages
```

Use `--help` for overrides and `--profile` for nsys. Throughput loss is
`100 × (1 − HLT1_with_PVFinder / HLT1_baseline)`, measured from paired runs.
Repeat measurements before claiming a margin below a target.

For timing diagnostics, `--alternate-order` reverses sequence order on even
repeats, and `--telemetry` records GPU clocks, power, throttling, competing
processes, host CPU load and Allen's scheduling/NUMA placement. Telemetry uses
line-buffered output to identify Allen's timed window. `--cpu-affinity LIST`
applies `taskset` to Allen for an explicit CPU-placement comparison.

## Physics validation

```bash
bash benchmarks/pv_comparison.sh --label peak_scan -n 10000 \
    --scan pvfinder_peak_pvfinder.threshold=0.07,0.1
```

This compares efficiency, false-positive rate and resolution against MC truth
for PVFinder and Allen's beamline PV finder, both over all z and within the
network's z range. It uses the same build, model and BF16 settings. `--fp32`
selects float32 arithmetic; also select the FP32 checkpoint with `--model`.
`--set ALG.PROPERTY=VALUE` changes a property; repeated `--scan` arguments
produce their Cartesian product. Physics checks retain the 2024 MC reference:
minimum bias, MagDown, nu 7.6, at 6.8 TeV beam energy. The real-data throughput
sample has no MC truth for efficiency and false-positive measurements.
Set `MC_MDF_FILE` and `MC_GEOMETRY_DIR` together to change the physics sample;
the general `MDF_FILE`/`GEOMETRY_DIR` overrides also remain supported.
Input paths are selected in [the shared defaults](benchmarks/defaults.sh).

## Inspect results

```bash
python3 benchmarks/runs.py list
python3 benchmarks/runs.py show <run_id>
python3 benchmarks/runs.py compare <run_a> <run_b>
```

[results/README.md](results/README.md) describes the records and provenance.
Past measurements remain tied to their original source revision; use a new
benchmark to characterize a changed build.

## Update Allen

The `allen/pvfinder` branch retains the integration commits and upstream
history. [`allen_upstream.json`](benchmarks/allen_upstream.json) pins the source
revisions. Prepare an update in an isolated directory:

```bash
python3 benchmarks/allen_upstream.py prepare --workdir /tmp/allen-next
SOURCE_DIR=/tmp/allen-next BUILD_DIR=/tmp/allen-next/build \
    bash benchmarks/build_allen.sh
bash benchmarks/validate_allen_upstream.sh /tmp/allen-next results/experiments/allen_next
```

Resolve conflicts there with `git rebase --continue`. Commit any further API
adaptations, rebuild and validate the resulting commit. After validation:

```bash
python3 benchmarks/allen_upstream.py import /tmp/allen-next.update.json \
    --validation results/experiments/allen_next/verification.json
bash benchmarks/build_allen.sh
git add results
git commit -m "Update Allen and preserve PVFinder integration"
```

Import stages the validated source and tracking metadata, preserves upstream
ancestry as a merge parent and leaves local input data untouched. The update
validator checks GPU unit tests, numerical agreement and the two MC working
points against the fixed 2024 MC reference counts. Its reference must be updated
before using a different sample for upstream verification. Throughput must be
measured separately. See the
[recorded integration checks](results/experiments/allen_master_20261006/README.md).
