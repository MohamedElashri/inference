# PVFinder inference in Allen

[Allen](https://gitlab.cern.ch/lhcb/Allen) is LHCb's GPU trigger (HLT1). It
reconstructs each event (tracks, vertices) and selects the interesting ones.
PVFinder is a neural network that finds the primary vertices (the proton-proton
collision points) from the VELO tracks. This repository runs the trained PVFinder
model inside Allen, based on **Allen v9r1**
(upstream `9352e3650eed34142288e3fbc6ef9154973135da`):

- **FC stage:** turns the tracks into features per 10 mm interval of the beam axis.
- **UNet:** turns those features into a density along the beam, whose peaks are
  the vertices.

| Directory | What it holds |
|---|---|
| `Allen/` | Allen, with PVFinder in `Allen/device/pvfinder/` |
| `weights/` | Pipeline from trained PyTorch checkpoints to Allen weight files, and the validators |
| `pvfinder_pytorch/` | The PyTorch model definition (used for validation) |
| `benchmarks/` | Throughput benchmark script and run-record tool |
| `results/runs/` | One JSON record per benchmark, profile or validation run |
| `results/experiments/` | Tracked optimization decisions and supporting evidence |

## Quick start

### 1. What you need

- An NVIDIA GPU. The UNet needs compute capability 8.0 or newer (e.g. RTX 3090, A100).
- CVMFS with `/cvmfs/lhcb.cern.ch`, for the compiler and CUDA toolchain.
- cuDNN in `~/local/cuda` (`lib64/libcudnn.so`, `include/`).
- Python 3 with `numpy` and `torch`, for the weights pipeline.
- The input data in `Allen/input/`, not in git:
  - an MDF file, e.g. `Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf`;
  - the matching geometry folder, e.g. `allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry`.

### 2. Build Allen

From the repository root:

```bash
bash benchmarks/build_v9r1.sh
```

This builds `Allen/buildv9r1gpu12/Allen` and its GPU unit tests, using CUDA 12.8,
LCG 108c, cuDNN, SM 8.6, 16 feature maps and 4 latent channels. The script
reuses the dependency source repositories from
`Allen/build16chL4finalgpu/external/`. On a new checkout, set `DEPENDENCY_CACHE`
to an existing cache containing `LHCb/`, `Gaudi/` and `ParamFiles/`, or set
`LHCBROOT`, `GAUDIROOT` and `PARAMFILESROOT` individually. Set `BUILD_DIR`,
`JOBS`, `CUDNN_ROOT` or `CUDA_ARCH` to override the local defaults.

`./ballen -a gpu --cudnn --tests --cuda-arch 86 -b mybuild -j 12` uses the same
build configuration with the alternative output directory `Allen/mybuildgpu/`.

### 3. Run plain HLT1

Allen runs from its build folder, through `toolchain/wrapper`, which sets up the environment:

```bash
cd Allen/buildv9r1gpu12
MDF=../input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf
GEO=../input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry
./toolchain/wrapper ./Allen --sequence hlt1_pp_default --mdf $MDF -g $GEO -n 1000 --device 2
```

`--sequence` names the algorithm chain to run. `-n` is the number of events.
`--device` is the GPU index as `nvidia-smi` numbers them. It prints the throughput in events/s at the end.

### 4. Prepare the PVFinder weights

The trained checkpoints are listed in `weights/models.tsv` (`make -C weights list`).
Convert one to Allen's format, then check the files against the checkpoint:

```bash
make -C weights verify MODEL=unet16_lc4_scnone_asym5_best_bf16
```

The weight files land in `weights/out/unet16_lc4_scnone_asym5_best_bf16/`.

### 5. Run HLT1 with PVFinder

From the repository root, run the accepted BF16 full-PV shadow configuration:

```bash
bash benchmarks/benchmark_v9r1_optimized.sh --repeats 1
```

This compares plain HLT1 with HLT1 plus PVFinder's FC stage, UNet, peak finding
and PV fit. The baseline PV reconstruction remains present in the shadow chain.
The script sets the absolute model path in both neural-network stages.

## Advanced

### The BF16 path (the fast configuration)

With the BF16 model, set `precision=bfloat16` in both neural-network stages.
The accepted configuration also enables the GPU work list and limits the
FC/UNet grids to 1/16 and 1/8 of the GPU, respectively. To set properties, generate the sequence's JSON configuration,
edit it, and pass the file to `--sequence`:

```bash
cd Allen/buildv9r1gpu12
./toolchain/wrapper bash -c 'PYTHONPATH=$PWD/code_generation/sequences:$PYTHONPATH python3 \
    code_generation/sequences/AllenCore/gen_allen_json.py --no-register-keys \
    --seqpath code_generation/sequences/AllenSequences/hlt1_pp_pvs_pvfinder_unet_benchmark.py'   # writes Sequence.json
python3 - <<'EOF'
import json
from pathlib import Path
c = json.load(open("Sequence.json"))
model = str(Path.cwd().parents[1] / "weights/out/unet16_lc4_scnone_asym5_best_bf16/pvfinder_model.json")
for alg in ("pvfinder_fc_aggregation", "pvfinder_unet"):
    c[alg]["model"] = model
c["pvfinder_unet"].update(precision="bfloat16", fused_grid_fraction=0.125)
c["pvfinder_fc_aggregation"].update(
    precision="bfloat16", gpu_work_list=True, fused_grid_fraction=0.0625)
json.dump(c, open("pvfinder_bf16.json", "w"))
EOF
./toolchain/wrapper ./Allen --sequence pvfinder_bf16.json --mdf $MDF -g $GEO -n 1000 --device 2
```

Any other property of `pvfinder_fc_aggregation` or `pvfinder_unet` is set the
same way. They are declared, with their meaning, in
`Allen/device/pvfinder/include/PVFinderFCAggregation.cuh` and `PVFinderUNet.cuh`.

### Throughput at the production point

The reference point is 16 streams:

```bash
./toolchain/wrapper ./Allen --sequence pvfinder_bf16.json --mdf $MDF -g $GEO \
    -n 500 -m 500 -r 1000 -t 16 --device 2
```

- `-t`: threads (CUDA streams).
- `-r`: repetitions per stream.
- `-m`: MB of device memory per stream.

Compare with `--sequence hlt1_pp_default` for the cost of PVFinder. On an RTX
3090, the accepted v9r1 full-PV shadow configuration measured **4.648 ± 0.570%**
throughput loss across five paired repeats (mean ± sample SD). The 95% interval
for the mean was 3.940–5.355%, so the current evidence does not establish a
reliable margin below 5%. See [the experiment report](results/experiments/v9r1_20261006/README.md).

The accepted settings are available through one command from the repository root:

```bash
bash benchmarks/benchmark_v9r1_optimized.sh
```

It compares plain HLT1 with HLT1 plus the full PVFinder shadow chain, using
16 streams, 500 events, 1,000 repetitions and five repeats on GPU 2 (RTX 3090).
It generates configurations with the current model paths and writes tracked
run records to `results/runs/`. Trailing arguments override defaults, for
example `--label my_run --repeats 3`. The underlying
`benchmarks/benchmark_pvfinder_batch.sh` also supports FC-only comparisons
and optional nsys profiles.

`--help` lists its options (one per PVFinder property). `benchmarks/runs.py
list|show|compare` reads the records; see `results/README.md`.

### Validate against the PyTorch model

`make dump` runs Allen on one slice and saves PVFinder's inputs and outputs.
`make validate` then compares them with the PyTorch model:

- the FC stage, per track and interval;
- the UNet, from Allen's FC output;
- the full model, from Allen's raw track features;
- the track features themselves, recomputed per track from Allen's track states,
  with the training reference ellipsoid code (and the rules checked against the
  training sample when `/share/lazy/sokoloff/...` is readable).

It writes a run record.

```bash
export LD_LIBRARY_PATH="$HOME/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
make -C weights ALLEN_BUILD_DIR="$PWD/Allen/buildv9r1gpu12" DEVICE=2 \
    dump validate MODEL=unet16_lc4_scnone_asym5_best_bf16 PRECISION=bfloat16 \
    DUMP_SET="pvfinder_fc_aggregation.gpu_work_list=true pvfinder_fc_aggregation.fused_grid_fraction=0.0625 pvfinder_unet.fused_grid_fraction=0.125"
```

The FP32 path must agree with PyTorch to float32 precision (max difference below
1e-3). The BF16 path differs by up to about 0.2 in the output density, so it is
judged on the peaks instead: under 1% of bins off by more than 0.01, median peak
height change under 2%, and the highest bin moving by more than one bin in under
1% of intervals (today 0.21%, 0.6% and 0.1%). See `weights/README.md`
for the whole weights pipeline.
