# `latentChannels-4_asym_5` weight bins

Converted from the training team's (mpeters) real trained weights at:
```
/share/lazy/mpeters/output/FCN6L_20-ch_UNet_16-ch_latentChannels-4/asym_5/weights/
```
using `convert_weights.py` (repo root; fixed here to handle bf16 source
tensors — numpy has no native bfloat16 dtype, so `.numpy()` on a bf16
tensor failed before the existing `.astype(np.float32)` ever ran; now
casts via `.float()` in torch first). Every `.bin` file below is FP32 —
this format doesn't carry precision info, it's a straight snapshot of
whatever the source `.pyt` held, upcast losslessly where the source was
fp16/bf16. **Do not read the `.bin` file sizes as evidence of any
throughput/storage saving** — that only happens at whatever point Allen's
own runtime casts down to a narrower dtype, not here.

Architecture (all 8): `latentChannels=4`, `n_unet_channels=16`,
`sc_mode="concat"` (same as current `pvfinder_pytorch/utils.py`),
`l_hidden_nodes=[20,20,20,20,20]`, `n_bins_per_interval=100`,
`loss_asym_coeff=5`. Requires Allen built with `N_FEAT=16`
(`buildgpu16chgpu`) — same build already used throughout
`optimization_plan.md`.

## FP32 baseline metrics (asym_5, best checkpoint, epoch 36)

efficiency 0.9632, fp_per_event 0.0207, val_loss 0.04624
(`weights_best.pyt`'s source metadata). `weights_final.pyt` is the
end-of-training (epoch 39) checkpoint instead; metrics are close but not
identical to `best` — see the two files' own metadata if the distinction
matters for a given test.

## Per-variant source and precision comparison

Quantization comparisons below are all against `weights_best.pyt` (the
`source_weights` recorded in each `quantization_*.json`), computed by the
training team's own eval harness, not by us — treat as a strong prior,
re-check against our path before trusting for an Allen default flip.

| Folder | Source `.pyt` | Precision / target | FP32 eff / fp-per-evt | Quantized eff / fp-per-evt | Payload reduction | Notes |
|---|---|---|---:|---:|---:|---|
| `weights_final/` | `weights_final.pyt` | fp32 (unquantized, last epoch) | 0.9628 / 0.0194 | — | — | end-of-training checkpoint |
| `weights_best/` | `weights_best.pyt` | fp32 (unquantized, best epoch) | 0.9632 / 0.0207 | — | — | best-val-loss checkpoint; source of all 6 quantized variants below |
| `weights_fp16_both/` | `weights_fp16_both.pyt` | fp16, FCN+UNet | 0.9632 / 0.0206 | 0.9632 / 0.0209 | 48.8% | quantized `loss` reported as NaN — ask before trusting |
| `weights_fp16_fcn/` | `weights_fp16_fcn.pyt` | fp16, FCN only | 0.9632 / 0.0206 | 0.9632 / 0.0209 | 22.8% | metrics identical to `fp16_both` in the source JSON — possibly a training-side eval artifact, not independently confirmed here |
| `weights_fp16_unet/` | `weights_fp16_unet.pyt` | fp16, UNet only | 0.9632 / 0.0206 | 0.9632 / 0.0207 | 26.1% | clean, no NaN |
| `weights_bf16_both/` | `weights_bf16_both.pyt` | bf16, FCN+UNet | 0.9632 / 0.0206 | 0.9634 / 0.0211 | 49.3% | no NaN in any bf16 variant |
| `weights_bf16_fcn/` | `weights_bf16_fcn.pyt` | bf16, FCN only | 0.9632 / 0.0206 | 0.9635 / 0.0210 | 23.2% | |
| `weights_bf16_unet/` | `weights_bf16_unet.pyt` | bf16, UNet only | 0.9632 / 0.0206 | 0.9633 / 0.0204 | 26.1% | closest of all six to the FP32 baseline |

## Known gap

`benchmark_pvfinder_batch.sh` only exposes `--cnn-weights` (overrides
`pvfinder_unet.weight_file`) — there is no equivalent flag for
`fc_weights.bin`'s path today, so swapping in one of these `fc_weights.bin`
files for a benchmark run needs a manual sequence-config edit until that
flag exists. Not fixed here since it wasn't asked for; flagging so it
isn't a surprise when Phase 0 of `docs/pvfinder_followup_plan.md` gets
picked up.

## Not converted here (available if wanted)

The same `asym_*` sweep (1, 2.5, 5, 7, 9, 11, 13, 15, 17, 19) exists
under the same `latentChannels-4` directory, plus separate
`_sc_add`/`_sc_none` skip-connection-ablation directories at this same
`latentChannels=4`/`n=16` operating point — see
`docs/pvfinder_followup_plan.md` Phase 0 for what's there. Only `asym_5`
was converted since that's what was pointed to; ask if the rest should be
generated too.
