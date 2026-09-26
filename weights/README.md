# PVFinder weights pipeline

Everything needed to turn a trained PVFinder checkpoint into the weight files
Allen loads, and to prove Allen reproduces that checkpoint, lives here and is
driven by `make`:

```bash
make -C weights list                                               # models in the catalog
make -C weights all MODEL=unet16_lc4_scnone_asym5_final            # fetch -> convert -> verify -> build -> dump -> validate
make -C weights verify-all                                         # fetch + convert + verify every model
eval "$(make -s -C weights env MODEL=unet16_lc4_scnone_asym5_final)" # export PVFINDER_WEIGHTS_DIR for Allen configs
```

`make help` lists every target and variable (`MODEL`, `DEVICE`, `EVENTS`, `JOBS`, `PY`,
`DUMP_SET`, `DUMP_TAG`, `LABEL`).

## Stages

| Target | What it does | Output (`out/<MODEL>/`, ignored by git) |
|---|---|---|
| `fetch` | Copies the checkpoint named in `models.tsv` | `checkpoints/<MODEL>.pyt` |
| `convert` | `scripts/convert.py`: checkpoint to Allen format | `cnn_weights.bin`, `fc_weights.bin` |
| `verify` | `scripts/verify.py`: re-reads both files in Allen's loader order and compares every tensor bit for bit with the checkpoint | `verify.txt` |
| `build` | `../ballen` with the model's `--unet-feat` / `--unet-batch-channels` into `Allen/<build>gpu` | Allen build |
| `dump` | `scripts/allen_dump.sh`: one 500-event slice with `dump_validation` on for FC and UNet, plus a `snapshot.json` of the environment | `dump/` (`dump_<DUMP_TAG>/`) |
| `validate` | `scripts/validate_fc.py` recomputes FC from the checkpoint per (track, interval) entry and checks Allen's track-to-interval assignment against the training rules; `scripts/validate_unet.py` recomputes the UNet from Allen's FC output; `scripts/validate_model.py` builds the network input from Allen's raw track features exactly like the training arrays and runs the full PyTorch model. Writes a run record to `results/runs/` | `dump*/validate_{fc,unet,model}.{txt,json}` |

To validate a non-default Allen configuration, override properties at dump
time and keep that dump separate with a tag, for example the BF16 path:

```bash
make -C weights dump validate MODEL=<name> DUMP_TAG=bf16 \
    DUMP_SET="pvfinder_unet.use_bf16=true pvfinder_fc_aggregation.unet_input_dtype=bfloat16 pvfinder_fc_aggregation.unet_input_layout=nwc"
```

`validate_fc.py` and `validate_unet.py` each check one Allen stage against its
own dumped input, so they prove the arithmetic but not that Allen feeds the
network what it was trained on; `validate_model.py` is the check for that,
and also prints physics-level numbers (fraction of intervals with a KDE peak,
peaks per event) to compare with the training sample. Allen's input
definition is documented in `docs/pvfinder/pvfinder_input_features.md`.

(The BF16 path runs channels last by default, `pvfinder_unet.bf16_layout = nwc`;
`unet_input_layout = nwc` has the FC stage write its BF16 output in that
layout. The dumps convert back to the channels-first float layout the
validators read. See `docs/pvfinder/bf16.md`.)

`validate` fails (non-zero exit) on any mismatch, after writing its run record
(see `results/README.md`). `validate_fc.py` also checks
that the `.bin` Allen loaded is the checkpoint in Allen's layout, and names a
transposed layer 6A explicitly.

## How Allen gets the weights

Allen has no built-in weight location. `pvfinder_fc_aggregation.weight_file`
and `pvfinder_unet.weight_file` default to empty, and Allen stops with an error
if either is unset. AllenConf (`pvfinder_weight_file()` in
`Allen/configuration/python/AllenConf/pvfinder_fc_reconstruction.py`) fills
them in from `$PVFINDER_WEIGHTS_DIR/{fc,cnn}_weights.bin` when a sequence
configuration is generated, and raises if the variable is unset or the files
are missing. `make env` prints the export, `make dump` sets it itself, and
`benchmarks/benchmark_pvfinder_batch.sh --model <name>` sets it from
`weights/out/<name>/`.

The Allen build must match the model: `N_FEAT` and `latentChannels` are
compile-time constants. The catalog's `build` column names the `ballen -b`
base; `make build` passes the right flags.

## Model architecture

Allen's UNet (`pvfinder_unet`) and the PyTorch reference
(`pvfinder_pytorch/utils.py`) implement the UNet **without skip connections**
(trained with `sc_mode=none`): rcbn1 -> rcbn2 -> pool -> rcbn3 -> pool -> up1
-> up2 -> out_intermediate -> outc, where up2's ConvTranspose and
out_intermediate take `N_FEAT` channels. `convert.py` and `verify.py` reject a
checkpoint trained with skip connections (`2*N_FEAT` inputs at those layers),
and Allen's loader checks every layer's shape against its build.

## Catalog (`models.tsv`)

Tab-separated: `name`, `source` checkpoint, `unet_feat`, `latent`, `build`,
`notes`, `precision` (the dtype of the checkpoint's conv and linear weights:
`fp32`, or `bf16` for the training team's reduced-precision exports). Add a
model by adding a row. Every run record carries the model's full row. All current models are `N_FEAT=16`,
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
| `unet16_lc4_scnone_asym1_best` / `_final` | 0.9383 / 0.9378 | 0.0042 / 0.0042 | epochs 82 / 86 |
| `unet16_lc4_scnone_asym2.5_best` | 0.9566 | 0.0130 | from the last `stats.csv` row, approximate |
| `unet16_lc4_scnone_asym17_final` | 0.9766 | 0.0842 | epoch 131 |
| `unet16_lc4_scnone_asym19_best` | 0.9767 | 0.0807 | upstream stats have a single epoch; likely incomplete |

## Limitations and history

- **Skip connections removed 2026-09-15.** Earlier the catalog defaulted to
  `unet16_lc8_iter9` and held latentChannels-4 models with concatenated or
  added skip connections; Allen only ran the concat architecture, so the
  no-skip models were validated on their FC stage alone. Those models and all
  skip-connection code paths (concat kernels, `skip_mode`,
  `use_merged_oint_outc`) are gone.
- **`legacy/`** (ignored) keeps files whose source checkpoint no longer exists:
  `unet16_lc4_scnone_asym17_best` (the upstream `weights_best.pyt` was
  overwritten on 2026-08-28 after conversion) and the 64-channel model's files.
- **Layer 6A layout bug, fixed 2026-09-14.** Allen's FC loader transposed
  layer 6A for every build, but the cuBLAS path reads the checkpoint's own
  `[latent*100 x 20]` row-major layout; only the non-cuBLAS kernel needs the
  transpose. In addition the converter wrote layer 6A transposed. FC results
  from cuBLAS builds before the fix (since 2026-03-09) did not reproduce the
  trained model. `validate_fc.py` catches both forms.
