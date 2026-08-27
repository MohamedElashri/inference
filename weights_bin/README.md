# Converted PVFinder weight bins

All source `.pyt` files come from the training team (mpeters) at
`/share/lazy/mpeters/output/`. Converted with `convert_weights.py` (repo
root) into `fc_weights.bin` + `cnn_weights.bin` pairs, all FP32 on disk
regardless of source precision (see each subfolder's notes on that).
Every architecture below is `latentChannels=4`, `n_unet_channels=16`
(`N_FEAT=16`, requires the `buildgpu16chgpu` Allen build),
`l_hidden_nodes=[20,20,20,20,20]`, `n_bins_per_interval=100` — only
`sc_mode` and the asymmetric loss coefficient (`asym`, trades efficiency
against false-positive rate) differ between groups.

| Folder | `sc_mode` | `asym` value(s) available | Precision variants | Detail |
|---|---|---|---|---|
| `latentChannels-4_asym_5/` | `concat` (current default) | 5 only | fp32, fp16 (x3 targets), bf16 (x3 targets) | see that folder's `MANIFEST.md` |
| `latentChannels-4_sc_add_asymm-7.5/` | `add` | 7.5 only (only value that exists upstream) | fp32 only | below |
| `latentChannels-4_sc_none_asym-{1,2.5,17,19}/` | `none` | 1, 2.5, 17, 19 (only values that exist upstream; no 5) | fp32 only | below |

**Structural confirmation**: `add`/`none` checkpoints have `up2`'s
ConvTranspose and `out_intermediate` at `16→16` channels; `concat` has
them at `32→16` — matches the `factor=1` vs. `factor=2` distinction in
`pvfinder_pytorch/utils.py`, confirms these are genuinely different
architectures, not the same weights with an ablation flag zeroing an
input.

**No fair `sc_mode` comparison exists yet at matched `asym`.** `concat`
only has `asym=5`; `add` only has `asym=7.5`; `none` has 1/2.5/17/19.
None of these overlap — do not read the raw efficiency numbers below as
"skip removal costs/gains X" without training a matched-`asym` point
first, or at minimum interpolating carefully within `sc_none`'s own
sweep. This was flagged as an open physics question in
`docs/pvfinder_followup_plan.md` Phase 0 and remains open.

## `latentChannels-4_sc_add_asymm-7.5/`

Source: `.../FCN6L_20-ch_UNet_16-ch_latentChannels-4_sc_add/asymm/7.5/weights/`
(`sc_mode="add"`, `asym=7.5`, 75 epochs trained).

| Folder | Checkpoint | efficiency | fp/event | val_loss |
|---|---|---:|---:|---:|
| `weights_best/` | epoch 54 (best val loss) | 0.9705 | 0.0402 | 0.05364 |
| `weights_final/` | epoch 75 (last) | 0.9682 | 0.0301 | 0.05408 |

## `latentChannels-4_sc_none_asym-*/`

Source: `.../FCN6L_20-ch_UNet_16-ch_latentChannels-4_sc_none/asym_{1,2.5,17,19}/weights/`
(`sc_mode="none"` — no skip connections at all, structurally, not just an
ablation flag).

| Folder | asym | Checkpoint(s) available | efficiency | fp/event | Notes |
|---|---:|---|---:|---:|---|
| `latentChannels-4_sc_none_asym-1/` | 1 | best (epoch 82) + final (epoch 86) | 0.9383 / 0.9378 | 0.0042 / 0.0042 | fully converged run |
| `latentChannels-4_sc_none_asym-2.5/` | 2.5 | best only | 0.9566 (last logged epoch, 69) | 0.0130 | no `metadata.json` upstream; pulled from `stats.csv`'s last row instead — treat as approximate |
| `latentChannels-4_sc_none_asym-17/` | 17 | best (epoch 107) + final (epoch 131) | 0.9766 | 0.0842 | fully converged run |
| `latentChannels-4_sc_none_asym-19/` | 19 | best only | 0.9767 | 0.0807 | **caution**: upstream `stats.csv` has exactly one row (epoch 0) — this looks like an early/possibly-incomplete run, not a converged model like the others. Don't treat this one as comparable to the rest without asking the training team about it. |

## Known gaps (same as `latentChannels-4_asym_5/`)

- No `--fc-weights` override in `benchmark_pvfinder_batch.sh` today —
  swapping `fc_weights.bin` needs a manual sequence-config edit.
- None of the `sc_add`/`sc_none` sources have fp16/bf16 quantized
  variants upstream — only `concat`/`asym_5` got that treatment so far.
