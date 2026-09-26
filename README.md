# PVFinder inference in Allen

[Allen](https://gitlab.cern.ch/lhcb/Allen) is LHCb's GPU trigger (HLT1). It
reconstructs each event (tracks, vertices) and selects the interesting ones.
PVFinder is a neural network that finds the primary vertices (the proton-proton
collision points) from the VELO tracks. This repository runs the trained PVFinder
model inside Allen:

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
./ballen -a gpu --cudnn --cublas --unet-feat 16 --unet-batch-channels 4 -b mybuild -j 16
```

This builds `Allen/mybuildgpu/Allen`. `--unet-feat 16 --unet-batch-channels 4`
matches the current models (16 feature maps, 4 latent channels).

### 3. Run plain HLT1

Allen runs from its build folder, through `toolchain/wrapper`, which sets up the environment:

```bash
cd Allen/mybuildgpu
MDF=../input/Beam6800GeV-expected-2024-MagDown-nu7.6_MinBiasMD.mdf
GEO=../input/allen_geometries/geometry_dddb-20231017_sim-20231017-vc-md100_new_SciFi_geometry
./toolchain/wrapper ./Allen --sequence hlt1_pp_default --mdf $MDF -g $GEO -n 1000 --device 0
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

PVFinder sequences find their weights through `PVFINDER_WEIGHTS_DIR`:

```bash
eval "$(make -s -C weights env MODEL=unet16_lc4_scnone_asym5_best_bf16)"   # sets PVFINDER_WEIGHTS_DIR
cd Allen/mybuildgpu
./toolchain/wrapper ./Allen --sequence hlt1_pp_pvfinder_unet_benchmark --mdf $MDF -g $GEO -n 1000 --device 0
```

This runs HLT1, then PVFinder's FC stage and UNet, with the default settings (FP32).

## Advanced

### The BF16 path (the fast configuration)

With the BF16 model, set three properties to run the reduced-precision path on
tensor cores. To set properties, generate the sequence's JSON configuration,
edit it, and pass the file to `--sequence`:

```bash
cd Allen/mybuildgpu
./toolchain/wrapper bash -c 'PYTHONPATH=$PWD/code_generation/sequences:$PYTHONPATH python3 \
    code_generation/sequences/AllenCore/gen_allen_json.py --no-register-keys \
    --seqpath code_generation/sequences/AllenSequences/hlt1_pp_pvfinder_unet_benchmark.py'   # writes Sequence.json
python3 - <<'EOF'
import json
c = json.load(open("Sequence.json"))
c["pvfinder_unet"]["use_bf16"] = True
c["pvfinder_fc_aggregation"].update(unet_input_dtype="bfloat16", unet_input_layout="nwc")
json.dump(c, open("pvfinder_bf16.json", "w"))
EOF
./toolchain/wrapper ./Allen --sequence pvfinder_bf16.json --mdf $MDF -g $GEO -n 1000 --device 0
```

Any other property of `pvfinder_fc_aggregation` or `pvfinder_unet` is set the
same way. They are declared, with their meaning, in
`Allen/device/pvfinder/include/PVFinderFCAggregation.cuh` and `PVFinderUNet.cuh`.

### Throughput at the production point

The reference point is 16 streams:

```bash
./toolchain/wrapper ./Allen --sequence pvfinder_bf16.json --mdf $MDF -g $GEO \
    -n 500 -m 500 -r 1000 -t 16 --device 0
```

- `-t`: threads (CUDA streams).
- `-r`: repetitions per stream.
- `-m`: MB of device memory per stream.

Compare with `--sequence hlt1_pp_default` for the cost of PVFinder. On an RTX
3090 the BF16 path costs about 4% of HLT1's throughput.

`benchmarks/benchmark_pvfinder_batch.sh` automates this. It runs HLT1 alone,
HLT1 + FC, and HLT1 + FC + UNet, with repeats and optional nsys profiles, and
writes a run record to `results/runs/`:

```bash
benchmarks/benchmark_pvfinder_batch.sh -B mybuildgpu -d 0 --model unet16_lc4_scnone_asym5_best_bf16 \
    --use-bf16 true -n 500 -m 500 -r 1000 -t 16 --repeats 3
```

`--help` lists its options (one per PVFinder property). `benchmarks/runs.py
list|show|compare` reads the records; see `results/README.md`.

### Validate against the PyTorch model

`make dump` runs Allen on one slice and saves PVFinder's inputs and outputs.
`make validate` then compares them with the PyTorch model:

- the FC stage, per track and interval;
- the UNet, from Allen's FC output;
- the full model, from Allen's raw track features;
- the track features themselves, recomputed per track from Allen's track states,
  with the training team's ellipsoid code (and the rules checked against the
  training sample when `/share/lazy/sokoloff/...` is readable).

It writes a run record.

```bash
make -C weights BUILD=mybuild DEVICE=0 dump validate MODEL=unet16_lc4_scnone_asym5_best_bf16 \
    DUMP_SET="pvfinder_unet.use_bf16=true pvfinder_fc_aggregation.unet_input_dtype=bfloat16 pvfinder_fc_aggregation.unet_input_layout=nwc"
```

The FP32 path must agree with PyTorch to float32 precision (max difference below
1e-3). The BF16 path differs by up to about 0.2 in the output density, so it is
judged on the peaks instead: under 1% of bins off by more than 0.01, median peak
height change under 2%, and the highest bin moving by more than one bin in under
1% of intervals (today 0.21%, 0.6% and 0.1%). See `weights/README.md`
for the whole weights pipeline.
