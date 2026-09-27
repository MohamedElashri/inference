# PVFinder weights pipeline

Everything needed to turn a trained PVFinder checkpoint into the model file
Allen reads, and to show that Allen reproduces that checkpoint, lives here and
is driven by `make`:

```bash
make -C weights list                                             # models in the catalog
make -C weights all MODEL=unet16_lc4_scnone_asym5_final          # fetch -> convert -> verify -> build -> dump -> validate
make -C weights verify-all                                       # fetch + convert + verify every model
make -s -C weights model-path MODEL=unet16_lc4_scnone_asym5_final  # the file to give Allen's "model" property
```

`make help` lists every target and variable (`MODEL`, `DEVICE`, `EVENTS`,
`JOBS`, `PY`, `PRECISION`, `DUMP_SET`, `DUMP_TAG`, `LABEL`).

## Stages

| Target | What it does | Output (`out/<MODEL>/`, ignored by git) |
|---|---|---|
| `fetch` | Copies the checkpoint named in `models.tsv` | `checkpoints/<MODEL>.pyt` |
| `convert` | `scripts/convert.py`: checkpoint to Allen's model file | `pvfinder_model.json` |
| `verify` | `scripts/verify.py`: re-reads the file independently of `convert.py` and compares every tensor bit for bit with the checkpoint, plus the metadata | `verify.txt` |
| `build` | `../ballen` with the model's `--unet-feat` / `--unet-batch-channels` into `Allen/<build>gpu` | Allen build |
| `dump` | `scripts/allen_dump.sh`: one 500-event slice of `hlt1_pp_pvs_pvfinder_unet_benchmark` (FC, UNet, peak finding, PV fit) with `dump_validation` on for the FC, the UNet and `pvfinder_peak`, plus a `snapshot.json` of the environment | `dump/` (`dump_<DUMP_TAG>/`) |
| `validate` | The five validators below on the dump; writes a run record to `results/runs/` | `dump*/validate_*.{txt,json}` |

The validators, each checking one step against an independent reference:

| Script | Checks |
|---|---|
| `validate_features.py` | Allen's 9 per-track input features, recomputed from each track's VELO state and the beamline with the training team's rules, and those rules against the training sample |
| `validate_fc.py` | The FC stage, recomputed in float64 from the checkpoint and Allen's track features, and Allen's track-to-interval assignment; differences measured in float32 (or bfloat16) ulps of a propagated magnitude bound |
| `validate_unet.py` | The UNet, run in PyTorch on Allen's own FC output |
| `validate_model.py` | The whole network: builds the input from Allen's raw track features exactly as the training arrays were built and runs the full PyTorch model; also prints physics-level numbers (intervals with a KDE peak, peaks per event) |
| `validate_peaks.py` | `pvfinder_peak` against pv-finder's own peak finder on Allen's KDE, with the settings Allen ran with |

`validate_fc.py` and `validate_unet.py` prove the arithmetic of one stage from
that stage's input; `validate_model.py` is the check that Allen feeds the
network what it was trained on. `validate` fails (non-zero exit) on any
mismatch, after writing its run record (see `results/README.md`).

The reduced-precision path is validated the same way:

```bash
make -C weights dump validate MODEL=unet16_lc4_scnone_asym5_best_bf16 PRECISION=bfloat16
```

`PRECISION=bfloat16` sets `precision` on both algorithms and keeps the dump in
`dump_bfloat16/`. With it, the validators judge the KDE at peak level
(`scripts/peak_agreement.py`) instead of float32 agreement. Other property
overrides go through `DUMP_SET="ALG.PROP=VALUE ..."`, with a `DUMP_TAG` to keep
that dump apart.

Primary-vertex physics (efficiency, false rate, resolution against the
beamline PV finder on MC) is not part of this pipeline:
`benchmarks/pv_comparison.sh` runs it with `scripts/compare_pvs.py` and
`scripts/validate_peaks.py`.

## The model file

One JSON file holds the FC network and the UNet. It is an Allen tensor model
file (`Allen::MVAModels::TensorModel`,
`Allen/device/utils/mva_models/include/TensorModel.h`; format described in
`Allen/doc/develop/add_mva_model.rst`) of kind `pvfinder`:

```json
{"format": "allen-tensors/1", "kind": "pvfinder", "name": "...", "source": "...", "sha256": "...",
 "metadata": {"latent_channels": 4, "unet_features": 16, "bn_eps": 1e-05},
 "tensors": {"layer1.weight": {"shape": [20, 9], "data": [...]}, ...}}
```

Tensors keep their PyTorch state-dict names and shapes, data row major. Each
value is written as the shortest decimal that reads back as the same float32,
so the file is exact; `source` and `sha256` identify the checkpoint.

## How Allen gets the model

`pvfinder_fc_aggregation` and `pvfinder_unet` read the file through their
`model` property (`PVFinder::Model`, `Allen/device/pvfinder/include/PVFinderModel.h`).
A relative path is taken in Allen's parameters directory (`--params`); the
default is `pvfinder/unet16_lc4_scnone_asym5_final.json`, for when the models
are in the ParamFiles package. Until then, pass an absolute path:

- AllenConf: `make_pvfinder_fc(velo_tracks, model="/abs/path/pvfinder_model.json")`;
  `make_pvfinder_unet` takes the same file from it.
- `benchmarks/benchmark_pvfinder_batch.sh --model <name>` uses
  `weights/out/<name>/pvfinder_model.json` (or `--model-file PATH`);
  `benchmarks/pv_comparison.sh --model <name>` likewise.
- `scripts/allen_dump.sh --model-file PATH` for any other sequence.

Allen reads the file once, before the algorithms' `init()`, and each
algorithm checks every tensor's shape against its build there. The Allen build
must match the model: `N_FEAT` (`--unet-feat`) and latentChannels
(`--unet-batch-channels`) are compile-time constants. The catalog's `build`
column names the `ballen -b` base; `make build` passes the right flags.

## Model architecture

Allen's UNet (`pvfinder_unet`) and the PyTorch reference
(`pvfinder_pytorch/utils.py`) implement the UNet **without skip connections**
(trained with `sc_mode=none`): rcbn1 -> rcbn2 -> pool -> rcbn3 -> pool -> up1
-> up2 -> out_intermediate -> outc, where up2's ConvTranspose and
out_intermediate take `N_FEAT` channels. `convert.py` and `verify.py` reject a
checkpoint trained with skip connections (`2*N_FEAT` inputs at those layers).

## Catalog (`models.tsv`)

Tab-separated: `name`, `source` checkpoint, `unet_feat`, `latent`, `build`,
`notes`, `precision` (the dtype of the checkpoint's conv and linear weights:
`fp32`, or `bf16` for the training team's reduced-precision exports; this is
not Allen's `precision` property). Add a model by adding a row. Every run
record carries the model's full row. All current models are `N_FEAT=16`,
latentChannels 4, with five 20-wide FC hidden layers and 100 bins per
interval, and build into `buildgpu16chL4gpu`. Sources are the training team's
outputs under `/share/lazy/mpeters/output/FCN6L_20-ch_UNet_16-ch_latentChannels-4_sc_none/`,
which also holds asym 7-15 sweeps and older `iter*` runs not catalogued here.

Training-side metrics recorded with the checkpoints (from the training team's
`metadata.json` and `stats.csv`, not measured here):

| Model | efficiency | fp/event | Notes |
|---|---:|---:|---|
| `unet16_lc4_scnone_asym5_final` | 0.9654 | 0.0214 | **default**; epoch 69, the last epoch |
| `unet16_lc4_scnone_asym5_best` | 0.9671 | 0.0241 | epoch 5 of 70, the lowest validation loss of that run |
| `unet16_lc4_scnone_asym5_best_bf16` | | | the training team's BF16 export of `asym5_best` (conv and linear weights rounded to bfloat16, BatchNorm float32) |
| `unet16_lc4_scnone_asym1_best` / `_final` | 0.9383 / 0.9378 | 0.0042 / 0.0042 | epochs 82 / 86 |
| `unet16_lc4_scnone_asym2.5_best` | 0.9566 | 0.0130 | from the last `stats.csv` row, approximate |
| `unet16_lc4_scnone_asym17_final` | 0.9766 | 0.0842 | epoch 131 |
| `unet16_lc4_scnone_asym19_best` | 0.9767 | 0.0807 | upstream stats have a single epoch; likely incomplete |

## History

- **2026-09-27: one model file.** Allen used to load two raw binaries
  (`fc_weights.bin`, `cnn_weights.bin`) found through `PVFINDER_WEIGHTS_DIR`.
  They are replaced by `pvfinder_model.json`, read through Allen's model
  mechanism, and the environment variable is gone.
- **2026-09-15: skip connections removed.** Earlier the catalog defaulted to
  `unet16_lc8_iter9` and held latentChannels-4 models with concatenated or
  added skip connections. Those models and all skip-connection code paths are
  gone.
- **2026-09-14: layer 6A layout bug fixed.** Allen's FC loader transposed
  layer 6A for every build, although only the plain kernel needed that; the
  cuBLAS path (since removed) read the checkpoint's own layout. The converter
  also wrote layer 6A transposed. FC results from cuBLAS builds between
  2026-03-09 and 2026-09-14 did not reproduce the trained model.
  `validate_fc.py` catches both forms.
- **`legacy/`** (ignored) keeps files whose source checkpoint no longer
  exists: `unet16_lc4_scnone_asym17_best` (the upstream `weights_best.pyt` was
  overwritten on 2026-08-28 after conversion) and the 64-channel model's files.
