#!/usr/bin/env python3
"""
validate_unet.py — Numerical validation of the Allen UNet inference against PyTorch.

Usage:
    python3 tools/validate_unet.py --dump-dir DUMP_DIR [--weights WEIGHTS_PATH]
                                   [--device cpu|cuda] [--plot]

Reads:
    <dump_dir>/allen_ncw_input.bin   — NCW input tensor dumped by Allen
    <dump_dir>/allen_kde_output.bin  — KDE output tensor dumped by Allen
    (written by pvfinder_unet when its dump_validation property is set)

Runs the same NCW input through the PyTorch model and compares outputs. The
UNet width (N_FEAT) and latentChannels are read from the checkpoint, so the
same script validates 16- and 64-channel models; the Allen build and the
checkpoint must describe the same model.

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Validate Allen UNet against PyTorch")
parser.add_argument("--dump-dir",  default="validation_dump",
                    help="Directory containing allen_ncw_input.bin and allen_kde_output.bin")
parser.add_argument("--weights",
                    default=os.path.join(
                        REPO_ROOT, "pvfinder_pytorch", "weights", "16-channel",
                        "FCN-20-channels_UNet-16-channels_nBinsPerSlice-100_latentChannels-8_iter9_final.pyt"),
                    help="PyTorch weight file (.pyt); default: the 16-channel latentChannels-8 model")
parser.add_argument("--device",    default="cpu", choices=["cpu", "cuda"],
                    help="Device for PyTorch inference (default: cpu)")
parser.add_argument("--plot",      action="store_true",
                    help="Save comparison plots to <dump_dir>/plots/")
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
from utils import combine

if not os.path.exists(args.weights):
    print(f"ERROR: weight file not found: {args.weights}")
    sys.exit(1)

print(f"Loading weights from {args.weights} ...")
state_dict = torch.load(args.weights, map_location="cpu")
if hasattr(state_dict, "state_dict"):
    state_dict = state_dict.state_dict()

# rcbn1 is Conv1d(latentChannels -> N_FEAT); layerK is Linear(in -> nOutK).
nUNetChannels, latentChannels = state_dict["rcbn1.0.weight"].shape[:2]
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
    """Run the UNet portion of the model starting from y0 = [N, C, W=100]."""
    # model is in eval mode; no dropout. n = N_FEAT below.
    x1 = model.rcbn1(y0)                                 # [N, n, 100]
    x2 = model.d(model.rcbn2(x1))                        # [N, n, 50]
    x  = model.d(model.rcbn3(x2))                        # [N, n, 25]
    x  = model.up1(x)                                    # [N, n, 50]

    # combine(x, x2, mode='concat') -> [N, 2n, 50]
    x  = model.up2(combine(x, x2, mode=model.mode))      # [N, n, 100]

    # out_intermediate expects concat(x, x1) -> [N, 2n, 100]
    x  = model.out_intermediate(combine(x, x1, mode=model.mode))  # [N, n, 100]
    logits = model.outc(x)                               # [N, 1, 100]
    y_pred = F.softplus(logits).squeeze(1) * 0.001       # [N, 100]
    return y_pred

print("Running PyTorch UNet inference (from y0) ...")
y0_t = torch.tensor(ncw_tensor, dtype=torch.float32).to(device)

with torch.no_grad():
    pt_out = run_unet_only(model, y0_t).cpu().numpy()    # [N*40, 100]

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

# Pass/fail threshold — expect cuDNN fp32 vs PyTorch fp32 differences < 1e-3
threshold = 1e-3
worst = abs_diff.max()
status = "PASS" if (worst < threshold and nonfinite_out == 0) else "FAIL"
print(f"\n  Threshold: {threshold:.0e}  →  {status}  (worst={worst:.3e})")

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
