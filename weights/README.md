# PVFinder weights pipeline

Everything needed to turn a trained PVFinder checkpoint into the weight files
Allen loads, and to prove Allen reproduces that checkpoint, lives here and is
driven by `make`:

```bash
make -C weights list                                   # models in the catalog
make -C weights all MODEL=unet16_lc8_iter9             # fetch -> convert -> verify -> build -> dump -> validate
make -C weights verify-all                             # fetch + convert + verify every model
eval "$(make -s -C weights env MODEL=unet16_lc8_iter9)" # export PVFINDER_WEIGHTS_DIR for Allen configs
```

`make help` lists every target and variable (`MODEL`, `DEVICE`, `EVENTS`, `JOBS`, `PY`).

## Stages

| Target | What it does | Output (`out/<MODEL>/`, ignored by git) |
|---|---|---|
| `fetch` | Copies the checkpoint named in `models.tsv` | `checkpoints/<MODEL>.pyt` |
| `convert` | `scripts/convert.py`: checkpoint to Allen format | `cnn_weights.bin`, `fc_weights.bin` |
| `verify` | `scripts/verify.py`: re-reads both files in Allen's loader order and compares every tensor bit for bit with the checkpoint | `verify.txt` |
| `build` | `../ballen` with the model's `--unet-feat` / `--unet-batch-channels` into `Allen/<build>gpu` | Allen build |
| `dump` | `scripts/allen_dump.sh`: one 500-event slice with `dump_validation` on for FC (and UNet) | `dump/` |
| `validate` | `scripts/validate_fc.py` recomputes FC from the checkpoint with Allen's own track-to-interval assignment; `scripts/validate_unet.py` does the same for the UNet | `validate_fc.txt`, `validate_unet.txt` |

`validate` fails (non-zero exit) on any mismatch. `validate_fc.py` also checks
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

## Catalog (`models.tsv`)

Tab-separated: `name`, `source` checkpoint, `unet_feat`, `latent`, `sc_mode`,
`allen_unet` (can Allen's UNet load it), `build`, `notes`. Add a model by
adding a row. All current models are `N_FEAT=16` with five 20-wide FC hidden
layers and 100 bins per interval. Sources are the training team's outputs
under `/share/lazy/mpeters/output/`.

| Model group | Allen build | Notes |
|---|---|---|
| `unet16_lc8_iter9` | `buildgpu16chgpu` | Default model; latentChannels 8 |
| `unet16_lc4_asym5_*` (8) | `buildgpu16chL4gpu` | latentChannels 4, concat skips, asym 5: `best`, `final`, and fp16/bf16 quantized variants of `best` (stored as fp32 on disk) |
| `unet16_lc4_scadd_asym7.5_*` (2) | `buildgpu16chL4gpu` | `add` skip connections; FC stage only in Allen |
| `unet16_lc4_scnone_*` (5) | `buildgpu16chL4gpu` | no skip connections (asym 1, 2.5, 17, 19); FC stage only in Allen |

Training-side metrics recorded with the checkpoints (from the training team's
metadata, not measured here):

| Model | efficiency | fp/event | Notes |
|---|---:|---:|---|
| `unet16_lc4_asym5_best` | 0.9632 | 0.0207 | epoch 36, source of the quantized variants |
| `unet16_lc4_asym5_final` | 0.9628 | 0.0194 | epoch 39 |
| `unet16_lc4_asym5_fp16_both` | 0.9632 | 0.0209 | quantized loss reported as NaN upstream |
| `unet16_lc4_asym5_fp16_fcn` | 0.9632 | 0.0209 | metrics identical to fp16_both upstream |
| `unet16_lc4_asym5_fp16_unet` | 0.9632 | 0.0207 | |
| `unet16_lc4_asym5_bf16_both` | 0.9634 | 0.0211 | |
| `unet16_lc4_asym5_bf16_fcn` | 0.9635 | 0.0210 | |
| `unet16_lc4_asym5_bf16_unet` | 0.9633 | 0.0204 | |
| `unet16_lc4_scadd_asym7.5_best` / `_final` | 0.9705 / 0.9682 | 0.0402 / 0.0301 | epochs 54 / 75 |
| `unet16_lc4_scnone_asym1_best` / `_final` | 0.9383 / 0.9378 | 0.0042 / 0.0042 | epochs 82 / 86 |
| `unet16_lc4_scnone_asym2.5_best` | 0.9566 | 0.0130 | from the last `stats.csv` row, approximate |
| `unet16_lc4_scnone_asym17_final` | 0.9766 | 0.0842 | epoch 131 |
| `unet16_lc4_scnone_asym19_best` | 0.9767 | 0.0807 | upstream stats have a single epoch; likely incomplete |

The `sc_mode` groups use different `asym` values, so they are not a matched
comparison of skip-connection choices.

## Limitations and history

- **Allen's UNet loader only supports concatenated skip connections.** For
  `add`/`none` checkpoints `out_intermediate` has `N_FEAT` inputs, which the
  loader's split does not handle; the pipeline validates their FC stage only.
- **`legacy/`** (ignored) keeps files whose source checkpoint no longer exists:
  `unet16_lc4_scnone_asym17_best` (the upstream `weights_best.pyt` was
  overwritten on 2026-08-28 after conversion) and the 64-channel model's files.
- **Layer 6A layout bug, fixed 2026-09-14.** Allen's FC loader transposed
  layer 6A for every build, but the cuBLAS path reads the checkpoint's own
  `[latent*100 x 20]` row-major layout; only the non-cuBLAS kernel needs the
  transpose. In addition the converter wrote layer 6A transposed. FC results
  from cuBLAS builds before the fix (since 2026-03-09) did not reproduce the
  trained model. `validate_fc.py` catches both forms.
