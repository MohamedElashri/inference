#!/usr/bin/env python3
"""
validate_unet.py — Numerical validation of the Allen UNet inference against PyTorch.

Usage:
    make -C weights validate MODEL=<name>        (normal use, after make dump)
    python3 weights/scripts/validate_unet.py --dump-dir DUMP_DIR [--weights WEIGHTS_PATH]
                                   [--device cpu|cuda] [--plot]

Reads:
    <dump_dir>/allen_ncw_input.bin   — NCW input tensor dumped by Allen
    <dump_dir>/allen_kde_output.bin  — KDE output tensor dumped by Allen
    (written by pvfinder_unet when its dump_validation property is set)

Runs the same NCW input through the PyTorch model (the UNet without skip
connections) and compares outputs. The UNet width (N_FEAT) and latentChannels
are read from the checkpoint; the Allen build and the checkpoint must describe
the same model.

Binary file format (written by PVFinderUNet.cu):
    uint32  magic   = 0xAB1E
    uint32  n_events
    float32 data[n_events * ...]
"""

import argparse
import os
import struct
import sys
import numpy as np

# weights/scripts/ -> weights/ -> repository root
WEIGHTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(WEIGHTS_DIR)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Validate Allen UNet against PyTorch")
parser.add_argument("--dump-dir",  default="validation_dump",
                    help="Directory containing allen_ncw_input.bin and allen_kde_output.bin")
parser.add_argument("--weights",
                    default=os.path.join(WEIGHTS_DIR, "checkpoints", "unet16_lc4_scnone_asym5_final.pyt"),
                    help="PyTorch weight file (.pyt); default: the unet16_lc4_scnone_asym5_final checkpoint fetched by the pipeline")
parser.add_argument("--device",    default="cpu", choices=["cpu", "cuda"],
                    help="Device for PyTorch inference (default: cpu)")
parser.add_argument("--plot",      action="store_true",
                    help="Save comparison plots to <dump_dir>/plots/")
parser.add_argument("--report",    default="",
                    help="Write a machine-readable JSON report to this path")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load the checkpoint first: its shapes decide how the dumps are read
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(REPO_ROOT, "pvfinder_pytorch"))
import torch

# utils.py imports awkward which may not be installed; stub it out since
# we only need the model class, not the data-loading helpers.
import types
if "awkward" not in sys.modules:
    sys.modules["awkward"] = types.ModuleType("awkward")

from utils import TrackIntervalsToKDE_HDplusUNet100 as Model

if not os.path.exists(args.weights):
    print(f"ERROR: weight file not found: {args.weights}")
    sys.exit(1)

print(f"Loading weights from {args.weights} ...")
state_dict = torch.load(args.weights, map_location="cpu")
if hasattr(state_dict, "state_dict"):
    state_dict = state_dict.state_dict()

# rcbn1 is Conv1d(latentChannels -> N_FEAT); layerK is Linear(in -> nOutK).
nUNetChannels, latentChannels = state_dict["rcbn1.0.weight"].shape[:2]
if (state_dict["up2.0.weight"].shape[0] != nUNetChannels
        or state_dict["out_intermediate.weight"].shape[1] != nUNetChannels):
    print("ERROR: this checkpoint has skip connections; Allen's UNet (and this validator) "
          "implement the model without them")
    sys.exit(2)
nOut1, nOut2, nOut3, nOut4, nOut5 = (state_dict[f"layer{k}.weight"].shape[0] for k in range(1, 6))
print(f"  checkpoint: N_FEAT={nUNetChannels}  latentChannels={latentChannels}  "
      f"FC hidden={nOut1},{nOut2},{nOut3},{nOut4},{nOut5}")

model = Model(nOut1, nOut2, nOut3, nOut4, nOut5,
              latentChannels=latentChannels, n=nUNetChannels)
model.load_state_dict(state_dict)
model.eval()

device = torch.device(args.device)
model = model.to(device)

# ---------------------------------------------------------------------------
# Read Allen binary dumps
# ---------------------------------------------------------------------------
MAGIC = 0xAB1E

def read_dump(path, elems_per_event):
    with open(path, "rb") as f:
        magic, n_events = struct.unpack("<II", f.read(8))
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic:#x}, expected {MAGIC:#x}")
    total = n_events * elems_per_event
    data = np.frombuffer(open(path, "rb").read()[8:], dtype=np.float32)
    if data.size != total:
        raise ValueError(f"{path}: expected {total} floats, got {data.size} "
                         f"(does the Allen build's latentChannels match the checkpoint's {latentChannels}?)")
    return n_events, data

N_INTERVALS    = 40
N_CHANNELS     = latentChannels
W_IN           = 100

ncw_path = os.path.join(args.dump_dir, "allen_ncw_input.bin")
kde_path = os.path.join(args.dump_dir, "allen_kde_output.bin")

print(f"Reading {ncw_path} ...")
n_events, ncw_flat = read_dump(ncw_path, N_INTERVALS * N_CHANNELS * W_IN)
print(f"Reading {kde_path} ...")
_, kde_flat = read_dump(kde_path, N_INTERVALS * W_IN)

# Reshape to PyTorch-natural dimensions
# NCW: [n_events * 40, C=latentChannels, W=100]
ncw_tensor = ncw_flat.reshape(n_events * N_INTERVALS, N_CHANNELS, W_IN)
# Allen KDE: [n_events, 40, 100]  (flat: n_events*40*100)
allen_kde = kde_flat.reshape(n_events * N_INTERVALS, W_IN)

print(f"  n_events={n_events}  ncw={ncw_tensor.shape}  allen_kde={allen_kde.shape}")

# ---------------------------------------------------------------------------
# Run PyTorch UNet inference
#
# The dumped NCW input is y0 = [N*40, C=latentChannels, W=100], which is the
# output of the FC aggregation stage — exactly the input to rcbn1.
# We run only the UNet portion of the model (rcbn1 onward), bypassing the
# FC layers (layer1..layer6A) that expect raw per-track features.
# ---------------------------------------------------------------------------
import torch.nn.functional as F

def run_unet_only(model, y0):
    """Run the UNet portion of the model starting from y0 = [N, C, W=100].

    Returns the KDE plus every intermediate stage, so a mismatch can be traced
    to the stage where it first appears.
    """
    # model is in eval mode; no dropout. n = N_FEAT below. No skip connections.
    stages = {}
    x1 = model.rcbn1(y0)                                 # [N, n, 100]
    stages["x1_rcbn1"] = x1
    x2 = model.d(model.rcbn2(x1))                        # [N, n, 50]
    stages["x2_rcbn2_mp"] = x2
    x  = model.d(model.rcbn3(x2))                        # [N, n, 25]
    stages["x3_rcbn3_mp"] = x
    x  = model.up1(x)                                    # [N, n, 50]
    stages["xu1_up1"] = x
    x  = model.up2(x)                                    # [N, n, 100]
    stages["xu2_up2"] = x
    x  = model.out_intermediate(x)                       # [N, n, 100]
    stages["x_oint"] = x
    logits = model.outc(x)                               # [N, 1, 100]
    stages["logits"] = logits
    y_pred = F.softplus(logits).squeeze(1) * 0.001       # [N, 100]
    stages["pt_kde"] = y_pred
    return y_pred, {k: v.cpu().numpy() for k, v in stages.items()}

print("Running PyTorch UNet inference (from y0) ...")
y0_t = torch.tensor(ncw_tensor, dtype=torch.float32).to(device)

with torch.no_grad():
    pt_tensor, stages = run_unet_only(model, y0_t)
    pt_out = pt_tensor.cpu().numpy()                     # [N*40, 100]

# Per-stage statistics on the PyTorch side: says at which stage an unexpected
# magnitude (or a non-finite value) first appears when the KDE disagrees.
# Statistics are over finite entries only, since intervals whose FC input is
# already non-finite would otherwise make every stage read NaN.
print("\n  PyTorch intermediate stage statistics")
layer_stats = {}
for lname, arr in stages.items():
    finite = np.isfinite(arr)
    st = {"shape": list(arr.shape), "non_finite": int((~finite).sum())}
    if finite.any():
        vals = arr[finite]
        st.update(mean=float(vals.mean()), std=float(vals.std()),
                  min=float(vals.min()), max=float(vals.max()))
    else:
        st.update(mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"))
    layer_stats[lname] = st
    print(f"    {lname:14s} shape={str(st['shape']):24s} mean={st['mean']:.4e} "
          f"std={st['std']:.4e} min={st['min']:.4e} max={st['max']:.4e} "
          f"non-finite={st['non_finite']}")

# Squeeze any residual channel dim
pt_kde = pt_out.reshape(n_events * N_INTERVALS, W_IN)

# ---------------------------------------------------------------------------
# Numerical comparison
#
# Intervals whose FC input already contains NaN are excluded: PyTorch
# propagates NaN through ReLU while Allen's bias+ReLU kernel maps it to 0, so
# those intervals cannot agree and say nothing about the UNet implementation.
# They are counted and reported, never silently dropped.
# ---------------------------------------------------------------------------
finite_input = np.isfinite(ncw_tensor).all(axis=(1, 2))
n_nan_input = int((~finite_input).sum())
print(f"\n  Intervals with non-finite FC input (excluded): {n_nan_input} / {finite_input.size}")
if n_nan_input:
    print(f"    Allen KDE finite on them: {bool(np.isfinite(allen_kde[~finite_input]).all())}")
nonfinite_out = int((~np.isfinite(allen_kde[finite_input])).sum() + (~np.isfinite(pt_kde[finite_input])).sum())
print(f"  Non-finite outputs on finite-input intervals: {nonfinite_out}")

diff     = np.where(finite_input[:, None], allen_kde - pt_kde, 0.0)
abs_diff = np.abs(diff)
rel_diff = abs_diff / (np.abs(np.where(finite_input[:, None], pt_kde, 0.0)) + 1e-9)

print("\n=== Numerical Comparison ===")
print(f"  Max  abs diff : {abs_diff.max():.6e}")
print(f"  Mean abs diff : {abs_diff.mean():.6e}")
print(f"  Median abs diff : {np.median(abs_diff):.6e}")
print(f"  Max  rel diff : {rel_diff.max():.6e}")
print(f"  Mean rel diff : {rel_diff.mean():.6e}")
print(f"  RMS  diff     : {np.sqrt((diff**2).mean()):.6e}")

# Per-event max abs diff
per_event_max = abs_diff.reshape(n_events, N_INTERVALS * W_IN).max(axis=1)
print(f"\n  Per-event max abs diff (first 10): "
      f"{[f'{v:.3e}' for v in per_event_max[:10]]}")

# Distribution, correlation and region statistics (finite-input intervals only)
fin_al = allen_kde[finite_input].astype(np.float64)
fin_pt = pt_kde[finite_input].astype(np.float64)
fin_abs = np.abs(fin_al - fin_pt)
fin_rel = fin_abs / (np.abs(fin_pt) + 1e-9)
PCT_LABELS = ["P50", "P90", "P95", "P99", "P99.9", "Pmax"]
abs_pcts = np.percentile(fin_abs, [50, 90, 95, 99, 99.9, 100])
rel_pcts = np.percentile(fin_rel, [50, 90, 95, 99, 99.9, 100])
print(f"\n  {'Percentile':10s}  {'Abs diff':>14s}  {'Rel diff':>14s}")
for lbl, av, rv in zip(PCT_LABELS, abs_pcts, rel_pcts):
    print(f"  {lbl:10s}  {av:14.6e}  {rv:14.6e}")

flat_pt, flat_al = fin_pt.ravel(), fin_al.ravel()
pearson_r = float(np.corrcoef(flat_pt, flat_al)[0, 1])
ss_tot = np.sum((flat_al - flat_al.mean()) ** 2)
r_squared = float(1.0 - np.sum((flat_al - flat_pt) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
print(f"\n  Pearson r: {pearson_r:.10f}   R^2: {r_squared:.10f}")

sig_mask = fin_pt > 1e-3
bg_mask = fin_pt < 1e-4
sig_worst = float(fin_abs[sig_mask].max()) if sig_mask.any() else 0.0
bg_worst = float(fin_abs[bg_mask].max()) if bg_mask.any() else 0.0
print(f"  Signal bins (PyTorch KDE > 1e-3): {int(sig_mask.sum())}, worst abs diff {sig_worst:.3e}")
print(f"  Background bins (PyTorch KDE < 1e-4): {int(bg_mask.sum())}, worst abs diff {bg_worst:.3e}")

worst_ev = int(np.argmax(per_event_max))
ev_diff = abs_diff.reshape(n_events, N_INTERVALS, W_IN)[worst_ev]
worst_iv = int(np.argmax(ev_diff.max(axis=1)))
print(f"  Worst event {worst_ev}, interval {worst_iv}: max abs diff {ev_diff[worst_iv].max():.3e}, "
      f"PyTorch peak {pt_kde.reshape(n_events, N_INTERVALS, W_IN)[worst_ev, worst_iv].max():.4e}, "
      f"Allen peak {allen_kde.reshape(n_events, N_INTERVALS, W_IN)[worst_ev, worst_iv].max():.4e}")

# Verdict. The 1e-3 tier decides the exit status (an exact FP32 cuDNN path
# against PyTorch fp32 lands far below it); the finer tiers show how much
# headroom there is, which is what separates an exact path from an approximate
# one such as FP16 or BF16.
TIERS = [("fp32_noise  (< 1e-5, ideal)", 1e-5),
         ("tight       (< 1e-4, good)", 1e-4),
         ("acceptable  (< 1e-3, fp32 ok)", 1e-3),
         ("loose       (< 1e-2, marginal)", 1e-2)]
threshold = 1e-3
worst = abs_diff.max()
status = "PASS" if (worst < threshold and nonfinite_out == 0) else "FAIL"
print()
tier_results, best_tier = {}, None
for label, thr in TIERS:
    ok = bool(worst < thr)
    tier_results[label.split()[0]] = ok
    print(f"  {label:34s} worst={worst:.3e}  {'PASS' if ok else 'FAIL'}")
    if ok and best_tier is None:
        best_tier = label.split()[0]
tier_note = f", best tier {best_tier}" if best_tier else ", exceeds every tier"
print(f"\n  Threshold: {threshold:.0e}  →  {status}  (worst={worst:.3e}{tier_note})")
sig_under_tight = bool(sig_worst < 1e-4)
print(f"  Signal region (worst {sig_worst:.3e}) under 1e-4: {'yes' if sig_under_tight else 'no'}")

if args.report:
    import json
    report = {
        "dump_dir": args.dump_dir,
        "weights": args.weights,
        "n_events": int(n_events),
        "n_intervals": int(finite_input.size),
        "non_finite_input_intervals": n_nan_input,
        "max_abs_diff": float(worst),
        "abs_percentiles": {l: float(v) for l, v in zip(PCT_LABELS, abs_pcts)},
        "rel_percentiles": {l: float(v) for l, v in zip(PCT_LABELS, rel_pcts)},
        "pearson_r": pearson_r,
        "r_squared": r_squared,
        "signal_region": {"n_bins": int(sig_mask.sum()), "max_abs_diff": sig_worst},
        "background_region": {"n_bins": int(bg_mask.sum()), "max_abs_diff": bg_worst},
        "worst_event": {"event": worst_ev, "interval": worst_iv},
        "layer_stats": layer_stats,
        "tiers": tier_results,
        "best_tier": best_tier,
        "signal_region_under_1e-4": sig_under_tight,
        "per_event_max_abs_diff": per_event_max.tolist(),
        "threshold": threshold,
        "status": status,
    }
    with open(args.report, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"  JSON report written to {args.report}")

# ---------------------------------------------------------------------------
# Plots (optional)
# ---------------------------------------------------------------------------
if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(args.dump_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot first N_PLOT intervals across events where signal is present
    N_PLOT = min(6, n_events * N_INTERVALS)
    # Pick intervals with largest PyTorch output peak
    peak_vals = pt_kde.max(axis=1)
    top_idxs  = np.argsort(peak_vals)[::-1][:N_PLOT]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, idx in zip(axes, top_idxs):
        ev_id  = idx // N_INTERVALS
        int_id = idx % N_INTERVALS
        ax.plot(pt_kde[idx],    label="PyTorch", color="red",  lw=1.5)
        ax.plot(allen_kde[idx], label="Allen",   color="blue", lw=1.5, ls="--")
        ax.set_title(f"event {ev_id}  interval {int_id}")
        ax.set_xlabel("bin")
        ax.set_ylabel("KDE")
        ax.legend(fontsize=8)
    fig.suptitle("Allen vs PyTorch UNet KDE output (top-peak intervals)")
    plt.tight_layout()
    out_path = os.path.join(plot_dir, "kde_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved plot: {out_path}")

    # Scatter: all values
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    flat_pt = pt_kde.flatten()
    flat_al = allen_kde.flatten()
    # Only plot values above noise
    mask = flat_pt > 1e-5
    ax2.scatter(flat_pt[mask], flat_al[mask], s=0.5, alpha=0.3)
    lim = max(flat_pt[mask].max(), flat_al[mask].max()) * 1.05
    ax2.plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
    ax2.set_xlabel("PyTorch KDE")
    ax2.set_ylabel("Allen KDE")
    ax2.set_title("Allen vs PyTorch (scatter, values > 1e-5)")
    ax2.legend()
    scatter_path = os.path.join(plot_dir, "kde_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    print(f"  Saved scatter: {scatter_path}")

print("\nDone.")
sys.exit(0 if status == "PASS" else 1)
