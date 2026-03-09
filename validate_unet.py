#!/usr/bin/env python3
"""
validate_unet.py — Rigorous numerical validation of the Allen cuDNN UNet inference
                   against the reference PyTorch model.

Usage:
    python3 validate_unet.py [--dump-dir DUMP_DIR] [--weights WEIGHTS_PATH]
                              [--device cpu|cuda] [--plot] [--report REPORT_PATH]

Reads:
    <dump_dir>/allen_ncw_input.bin   — NCW input tensor dumped by Allen
    <dump_dir>/allen_kde_output.bin  — KDE output tensor dumped by Allen

Runs the identical NCW input through PyTorch and performs a rigorous comparison:
  1. Global statistics (max/mean/median/RMS abs & relative error)
  2. Per-event and per-interval breakdowns
  3. Pearson correlation and R² of Allen vs PyTorch
  4. Error distribution percentiles (P50/P90/P95/P99/P99.9/max)
  5. Signal-region analysis (bins with KDE > threshold)
  6. Per-channel input statistics to verify weight loading
  7. Intermediate layer-by-layer comparison (Allen vs PyTorch at each UNet stage)
  8. Pass/fail verdict with tiered thresholds (tight/loose/fp32-noise)
  9. Plots: overlay, scatter, residual histogram, error vs signal level (--plot)
 10. Machine-readable JSON report (--report)

Binary file format (written by PVFinderUNet.cu):
    uint32  magic   = 0xAB1E
    uint32  n_events
    float32 data[n_events * ...]
"""

import argparse
import json
import os
import struct
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Rigorous Allen UNet vs PyTorch validation")
parser.add_argument("--dump-dir", default="validation_dump",
                    help="Directory with allen_ncw_input.bin and allen_kde_output.bin")
parser.add_argument("--weights",
                    default="pvfinder_pytorch/weights/"
                            "07Sept2023_t2hists_HDplusUNet100_iter12Ca_200epochs_2em5_5p0_final.pyt",
                    help="PyTorch weight file (.pyt)")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device for PyTorch inference (default: cpu)")
parser.add_argument("--plot", action="store_true",
                    help="Save comparison plots to <dump_dir>/plots/")
parser.add_argument("--report", default="",
                    help="Write machine-readable JSON summary to this path")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
SEP  = "=" * 72
SEP2 = "-" * 72

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subsection(title):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def stat_block(name, arr):
    """Print a compact statistical summary of a flat numpy array."""
    a = arr.flatten()
    pcts = np.percentile(a, [50, 90, 95, 99, 99.9, 100])
    print(f"  {name}:")
    print(f"    mean={a.mean():.4e}  std={a.std():.4e}  min={a.min():.4e}  max={a.max():.4e}")
    print(f"    P50={pcts[0]:.4e}  P90={pcts[1]:.4e}  P95={pcts[2]:.4e}  "
          f"P99={pcts[3]:.4e}  P99.9={pcts[4]:.4e}  Pmax={pcts[5]:.4e}")
    return pcts

# ---------------------------------------------------------------------------
# Read Allen binary dumps
# ---------------------------------------------------------------------------
MAGIC = 0xAB1E

def read_dump(path, elems_per_event):
    with open(path, "rb") as f:
        raw = f.read()
    magic, n_events = struct.unpack_from("<II", raw, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic:#x}, expected {MAGIC:#x}")
    data = np.frombuffer(raw, dtype=np.float32, offset=8)
    total = n_events * elems_per_event
    if data.size != total:
        raise ValueError(f"{path}: expected {total} floats, got {data.size}")
    return n_events, data.copy()

N_INTERVALS = 40
N_CHANNELS  = 8
W_IN        = 100

ncw_path = os.path.join(args.dump_dir, "allen_ncw_input.bin")
kde_path = os.path.join(args.dump_dir, "allen_kde_output.bin")

section("Loading Allen Dumps")
print(f"  NCW  : {ncw_path}")
print(f"  KDE  : {kde_path}")

n_events, ncw_flat = read_dump(ncw_path, N_INTERVALS * N_CHANNELS * W_IN)
_, kde_flat        = read_dump(kde_path, N_INTERVALS * W_IN)

# NCW: [N_evt*40, C=8, W=100]
ncw_tensor = ncw_flat.reshape(n_events * N_INTERVALS, N_CHANNELS, W_IN)
# KDE: [N_evt*40, W=100]
allen_kde  = kde_flat.reshape(n_events * N_INTERVALS, W_IN)

N = n_events * N_INTERVALS   # total intervals
print(f"  n_events={n_events}  intervals={N}  ncw={ncw_tensor.shape}  allen_kde={allen_kde.shape}")

subsection("Input (NCW) statistics")
for c in range(N_CHANNELS):
    ch = ncw_tensor[:, c, :]
    print(f"    channel {c}: mean={ch.mean():.4e}  std={ch.std():.4e}  "
          f"min={ch.min():.4e}  max={ch.max():.4e}")

subsection("Allen KDE output statistics")
stat_block("Allen KDE", allen_kde)
sig_mask_allen = allen_kde > 1e-3
print(f"  Bins above 1e-3: {sig_mask_allen.sum()} / {allen_kde.size} "
      f"({100*sig_mask_allen.mean():.2f}%)")

# ---------------------------------------------------------------------------
# Load PyTorch model
# ---------------------------------------------------------------------------
section("Loading PyTorch Model")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pvfinder_pytorch"))
import torch
import torch.nn.functional as F

import types
import sys as _sys
if "awkward" not in _sys.modules:
    _sys.modules["awkward"] = types.ModuleType("awkward")

from utils import TrackIntervalsToKDE_HDplusUNet100 as Model, combine

nOut1 = nOut2 = nOut3 = nOut4 = nOut5 = 20
latentChannels = 8
nUNetChannels  = 64

model = Model(nOut1, nOut2, nOut3, nOut4, nOut5,
              latentChannels=latentChannels, n=nUNetChannels)

if not os.path.exists(args.weights):
    print(f"ERROR: weight file not found: {args.weights}")
    sys.exit(1)

print(f"  Weights: {args.weights}")
d = torch.load(args.weights, map_location="cpu", weights_only=True)
model.load_state_dict(d)
model.eval()

device = torch.device(args.device)
model  = model.to(device)
print(f"  Device : {device}")

# ---------------------------------------------------------------------------
# PyTorch UNet forward — capture intermediate tensors for layer-by-layer study
# ---------------------------------------------------------------------------
section("PyTorch UNet Inference + Intermediate Captures")

y0_t = torch.tensor(ncw_tensor, dtype=torch.float32).to(device)

intermediates = {}

with torch.no_grad():
    x1 = model.rcbn1(y0_t)                             # [N, 64, 100]
    intermediates["x1_rcbn1"]    = x1.cpu().numpy()

    x2 = model.d(model.rcbn2(x1))                      # [N, 64,  50]
    intermediates["x2_rcbn2_mp"] = x2.cpu().numpy()

    x3 = model.d(model.rcbn3(x2))                      # [N, 64,  25]
    intermediates["x3_rcbn3_mp"] = x3.cpu().numpy()

    xu1 = model.up1(x3)                                # [N, 64,  50]
    intermediates["xu1_up1"]     = xu1.cpu().numpy()

    xu2 = model.up2(combine(xu1, x2, mode=model.mode)) # [N, 64, 100]
    intermediates["xu2_up2"]     = xu2.cpu().numpy()

    xoi = model.out_intermediate(combine(xu2, x1, mode=model.mode))  # [N, 64, 100]
    intermediates["x_oint"]      = xoi.cpu().numpy()

    logits = model.outc(xoi)                           # [N,  1, 100]
    intermediates["logits"]      = logits.cpu().numpy()

    y_pred = F.softplus(logits).squeeze(1) * 0.001    # [N, 100]
    intermediates["pt_kde"]      = y_pred.cpu().numpy()

pt_kde = intermediates["pt_kde"].reshape(N, W_IN)

subsection("PyTorch KDE output statistics")
stat_block("PyTorch KDE", pt_kde)

# ---------------------------------------------------------------------------
# Intermediate layer statistics (PyTorch side — useful for debugging deviations)
# ---------------------------------------------------------------------------
subsection("PyTorch intermediate layer statistics")
layer_names = ["x1_rcbn1", "x2_rcbn2_mp", "x3_rcbn3_mp",
               "xu1_up1", "xu2_up2", "x_oint", "logits", "pt_kde"]
for lname in layer_names:
    a = intermediates[lname]
    print(f"    {lname:20s}: shape={a.shape}  "
          f"mean={a.mean():.4e}  std={a.std():.4e}  "
          f"min={a.min():.4e}  max={a.max():.4e}")

# ---------------------------------------------------------------------------
# Core numerical comparison: Allen vs PyTorch
# ---------------------------------------------------------------------------
section("Numerical Comparison: Allen vs PyTorch")

diff     = allen_kde.astype(np.float64) - pt_kde.astype(np.float64)
abs_diff = np.abs(diff)
pt_abs   = np.abs(pt_kde).astype(np.float64)
rel_diff = abs_diff / (pt_abs + 1e-9)

subsection("Global error statistics")
print(f"  Max  abs diff       : {abs_diff.max():.6e}")
print(f"  Mean abs diff       : {abs_diff.mean():.6e}")
print(f"  Median abs diff     : {np.median(abs_diff):.6e}")
print(f"  RMS diff            : {np.sqrt((diff**2).mean()):.6e}")
print(f"  Max  rel diff       : {rel_diff.max():.6e}")
print(f"  Mean rel diff       : {rel_diff.mean():.6e}")
print(f"  Median rel diff     : {np.median(rel_diff):.6e}")

subsection("Error distribution percentiles")
abs_pcts = np.percentile(abs_diff.flatten(), [50, 90, 95, 99, 99.9, 100])
rel_pcts = np.percentile(rel_diff.flatten(), [50, 90, 95, 99, 99.9, 100])
labels   = ["P50", "P90", "P95", "P99", "P99.9", "Pmax"]
print(f"  {'Percentile':10s}  {'Abs diff':>14s}  {'Rel diff':>14s}")
print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
for lbl, av, rv in zip(labels, abs_pcts, rel_pcts):
    print(f"  {lbl:10s}  {av:14.6e}  {rv:14.6e}")

subsection("Correlation (Allen vs PyTorch)")
flat_pt = pt_kde.flatten().astype(np.float64)
flat_al = allen_kde.flatten().astype(np.float64)
pearson_r = np.corrcoef(flat_pt, flat_al)[0, 1]
ss_res = np.sum((flat_al - flat_pt)**2)
ss_tot = np.sum((flat_al - flat_al.mean())**2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
print(f"  Pearson r  : {pearson_r:.10f}")
print(f"  R²         : {r_squared:.10f}")
print(f"  Slope (Allen/PyTorch mean ratio): {flat_al.mean()/flat_pt.mean():.8f}"
      if flat_pt.mean() != 0 else "  Slope: undefined (PT mean=0)")

subsection("Signal-region analysis  (PyTorch KDE > 1e-3)")
sig_mask = pt_kde > 1e-3
n_sig    = sig_mask.sum()
print(f"  Signal bins (PyTorch > 1e-3): {n_sig} / {pt_kde.size} ({100*n_sig/pt_kde.size:.2f}%)")
if n_sig > 0:
    sig_abs = abs_diff[sig_mask]
    sig_rel = rel_diff[sig_mask]
    print(f"  Signal abs diff: mean={sig_abs.mean():.4e}  max={sig_abs.max():.4e}  "
          f"P99={np.percentile(sig_abs,99):.4e}")
    print(f"  Signal rel diff: mean={sig_rel.mean():.4e}  max={sig_rel.max():.4e}  "
          f"P99={np.percentile(sig_rel,99):.4e}")

subsection("Background-region analysis  (PyTorch KDE < 1e-4)")
bg_mask = pt_kde < 1e-4
n_bg    = bg_mask.sum()
if n_bg > 0:
    bg_abs = abs_diff[bg_mask]
    print(f"  Background bins: {n_bg}  abs diff: mean={bg_abs.mean():.4e}  max={bg_abs.max():.4e}")

subsection("Per-event breakdown")
per_event_abs = abs_diff.reshape(n_events, N_INTERVALS * W_IN)
per_event_max_abs  = per_event_abs.max(axis=1)
per_event_mean_abs = per_event_abs.mean(axis=1)
print(f"  Max  per-event max  abs diff : {per_event_max_abs.max():.4e}")
print(f"  Mean per-event max  abs diff : {per_event_max_abs.mean():.4e}")
print(f"  Mean per-event mean abs diff : {per_event_mean_abs.mean():.4e}")
print(f"\n  Per-event max abs diff (first 20 events):")
for i in range(min(20, n_events)):
    flag = " <-- WORST" if per_event_max_abs[i] == per_event_max_abs.max() else ""
    print(f"    event {i:4d}: max_abs={per_event_max_abs[i]:.4e}  "
          f"mean_abs={per_event_mean_abs[i]:.4e}{flag}")

worst_ev = int(np.argmax(per_event_max_abs))
subsection(f"Worst event deep-dive  (event {worst_ev})")
wev_pt = pt_kde.reshape(n_events, N_INTERVALS, W_IN)[worst_ev]  # [40, 100]
wev_al = allen_kde.reshape(n_events, N_INTERVALS, W_IN)[worst_ev]
wev_diff = np.abs(wev_pt - wev_al)
worst_int = int(np.argmax(wev_diff.max(axis=1)))
print(f"  Worst interval within event {worst_ev}: interval {worst_int}")
print(f"  PyTorch  peak: {wev_pt[worst_int].max():.6e}  "
      f"at bin {wev_pt[worst_int].argmax()}")
print(f"  Allen    peak: {wev_al[worst_int].max():.6e}  "
      f"at bin {wev_al[worst_int].argmax()}")
print(f"  Max abs diff in that interval: {wev_diff[worst_int].max():.6e}")
print(f"  First 10 bins — PyTorch: {wev_pt[worst_int,:10]}")
print(f"  First 10 bins — Allen  : {wev_al[worst_int,:10]}")
print(f"  First 10 bins — |diff| : {wev_diff[worst_int,:10]}")

# ---------------------------------------------------------------------------
# Pass / Fail verdict — tiered thresholds
# ---------------------------------------------------------------------------
section("Pass / Fail Verdict")

THRESHOLDS = {
    "fp32_noise  (< 1e-5, ideal)"   : 1e-5,
    "tight       (< 1e-4, good)"    : 1e-4,
    "acceptable  (< 1e-3, fp32 ok)" : 1e-3,
    "loose       (< 1e-2, marginal)": 1e-2,
}

worst_abs = float(abs_diff.max())
overall   = None
for label, thr in THRESHOLDS.items():
    ok = worst_abs < thr
    print(f"  {label:40s}  worst={worst_abs:.3e}  {'PASS' if ok else 'FAIL'}")
    if ok and overall is None:
        overall = label

print()
if overall is not None:
    print(f"  >>> Overall: PASS  (meets '{overall}' threshold)")
else:
    print(f"  >>> Overall: FAIL  (worst abs diff {worst_abs:.3e} exceeds all thresholds)")

# Signal-region verdict (stricter — only signal bins matter for PV finding)
sig_worst = float(abs_diff[sig_mask].max()) if n_sig > 0 else 0.0
sig_ok_tight = sig_worst < 1e-4
sig_ok_loose  = sig_worst < 1e-3
print(f"\n  Signal-region (KDE>1e-3) verdict:")
print(f"    worst abs diff = {sig_worst:.3e}  "
      f"tight(1e-4)={'PASS' if sig_ok_tight else 'FAIL'}  "
      f"acceptable(1e-3)={'PASS' if sig_ok_loose else 'FAIL'}")

# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------
report = {
    "n_events"        : int(n_events),
    "n_intervals"     : N,
    "weights"         : args.weights,
    "device"          : str(device),
    "global_stats": {
        "max_abs_diff"    : float(abs_diff.max()),
        "mean_abs_diff"   : float(abs_diff.mean()),
        "median_abs_diff" : float(np.median(abs_diff)),
        "rms_diff"        : float(np.sqrt((diff**2).mean())),
        "max_rel_diff"    : float(rel_diff.max()),
        "mean_rel_diff"   : float(rel_diff.mean()),
        "pearson_r"       : float(pearson_r),
        "r_squared"       : float(r_squared),
    },
    "abs_percentiles" : {lbl: float(v) for lbl, v in zip(labels, abs_pcts)},
    "rel_percentiles" : {lbl: float(v) for lbl, v in zip(labels, rel_pcts)},
    "signal_region"   : {
        "n_signal_bins"   : int(n_sig),
        "max_abs_diff"    : sig_worst,
        "pass_tight"      : bool(sig_ok_tight),
        "pass_acceptable" : bool(sig_ok_loose),
    },
    "per_event_max_abs_diff" : per_event_max_abs.tolist(),
    "verdict"         : overall if overall else "FAIL",
}

if args.report:
    with open(args.report, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\n  JSON report written to: {args.report}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plot_dir = os.path.join(args.dump_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    section(f"Generating Plots → {plot_dir}")

    # ---- 1. KDE overlay: top-6 peak intervals ----
    N_PLOT   = min(6, N)
    peak_vals = pt_kde.max(axis=1)
    top_idxs  = np.argsort(peak_vals)[::-1][:N_PLOT]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, idx in zip(axes, top_idxs):
        ev_id  = idx // N_INTERVALS
        int_id = idx % N_INTERVALS
        ax.plot(pt_kde[idx],    label="PyTorch", color="crimson",  lw=1.8)
        ax.plot(allen_kde[idx], label="Allen",   color="steelblue", lw=1.8, ls="--")
        ax.fill_between(range(W_IN),
                        np.abs(pt_kde[idx] - allen_kde[idx]),
                        alpha=0.25, color="orange", label="|diff|")
        ax.set_title(f"ev {ev_id}  int {int_id}", fontsize=9)
        ax.set_xlabel("bin")
        ax.set_ylabel("KDE")
        ax.legend(fontsize=7)
    fig.suptitle("Allen vs PyTorch UNet — top-peak intervals", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(plot_dir, "01_kde_overlay.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")

    # ---- 2. Worst event overlay ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    worst_ints = np.argsort(wev_diff.max(axis=1))[::-1][:8]
    for ax, wi in zip(axes, worst_ints):
        ax.plot(wev_pt[wi],   label="PyTorch", color="crimson",   lw=1.5)
        ax.plot(wev_al[wi],   label="Allen",   color="steelblue", lw=1.5, ls="--")
        ax.set_title(f"worst ev {worst_ev}  int {wi}", fontsize=8)
        ax.set_xlabel("bin"); ax.set_ylabel("KDE")
        ax.legend(fontsize=6)
    fig.suptitle(f"Worst event (ev {worst_ev}) — top-8 intervals by max |diff|",
                 fontweight="bold")
    plt.tight_layout()
    p = os.path.join(plot_dir, "02_worst_event_overlay.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")

    # ---- 3. Scatter: Allen vs PyTorch (log scale) ----
    fig, ax = plt.subplots(figsize=(7, 7))
    above = flat_pt > 1e-6
    ax.scatter(flat_pt[above], flat_al[above], s=0.3, alpha=0.2, c="steelblue")
    lim = max(flat_pt[above].max(), flat_al[above].max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, label=f"y=x  (r={pearson_r:.8f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("PyTorch KDE"); ax.set_ylabel("Allen KDE")
    ax.set_title("Allen vs PyTorch — scatter (log-log, KDE > 1e-6)")
    ax.legend(fontsize=9)
    p = os.path.join(plot_dir, "03_scatter_loglog.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ---- 4. Absolute error histogram ----
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = abs_diff.flatten()
    vals_nz = vals[vals > 0]
    ax.hist(np.log10(vals_nz + 1e-12), bins=100, color="steelblue", edgecolor="none",
            alpha=0.8, label=f"n={len(vals_nz)}")
    for thr, col, lbl in [(1e-5, "green", "1e-5"), (1e-4, "orange", "1e-4"),
                           (1e-3, "red",   "1e-3"), (1e-2, "darkred", "1e-2")]:
        ax.axvline(np.log10(thr), color=col, ls="--", lw=1.5, label=lbl)
    ax.set_xlabel("log10(|Allen − PyTorch|)")
    ax.set_ylabel("Count")
    ax.set_title("Absolute error distribution (all bins)")
    ax.legend(fontsize=8)
    p = os.path.join(plot_dir, "04_abs_error_histogram.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ---- 5. Relative error histogram (signal bins only) ----
    if n_sig > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        rvals = rel_diff[sig_mask].flatten()
        ax.hist(np.log10(rvals + 1e-12), bins=100, color="crimson", edgecolor="none",
                alpha=0.8, label=f"signal bins n={len(rvals)}")
        for thr, col, lbl in [(1e-4, "orange", "1e-4"), (1e-3, "red", "1e-3"),
                               (1e-2, "darkred", "1e-2")]:
            ax.axvline(np.log10(thr), color=col, ls="--", lw=1.5, label=lbl)
        ax.set_xlabel("log10(relative error)  [signal bins, KDE>1e-3]")
        ax.set_ylabel("Count")
        ax.set_title("Relative error distribution — signal region")
        ax.legend(fontsize=8)
        p = os.path.join(plot_dir, "05_rel_error_signal_hist.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p}")

    # ---- 6. Error vs signal level (2-D density) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    pt_above = flat_pt[flat_pt > 1e-6]
    ae_above = abs_diff.flatten()[flat_pt > 1e-6]
    h, xedges, yedges = np.histogram2d(
        np.log10(pt_above + 1e-12), np.log10(ae_above + 1e-12), bins=80)
    h = np.ma.masked_where(h == 0, h)
    ax.pcolormesh(xedges, yedges, h.T, norm=LogNorm(), cmap="plasma")
    ax.set_xlabel("log10(PyTorch KDE)")
    ax.set_ylabel("log10(|diff|)")
    ax.set_title("Error magnitude vs signal level (2-D density)")
    p = os.path.join(plot_dir, "06_error_vs_signal_2d.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ---- 7. Per-event max abs diff bar chart ----
    fig, ax = plt.subplots(figsize=(max(8, n_events//4), 4))
    colors = ["crimson" if v > 1e-3 else ("orange" if v > 1e-4 else "steelblue")
              for v in per_event_max_abs]
    ax.bar(range(n_events), per_event_max_abs, color=colors, width=0.8)
    ax.axhline(1e-3, color="red",    ls="--", lw=1.2, label="1e-3 threshold")
    ax.axhline(1e-4, color="orange", ls="--", lw=1.2, label="1e-4 threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Event index"); ax.set_ylabel("Max |diff|")
    ax.set_title("Per-event maximum absolute difference (Allen vs PyTorch)")
    ax.legend(fontsize=8)
    p = os.path.join(plot_dir, "07_per_event_max_diff.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    print(f"\n  All plots saved to: {plot_dir}")

print(f"\n{'='*72}")
print("  Done.")
print(f"{'='*72}\n")
