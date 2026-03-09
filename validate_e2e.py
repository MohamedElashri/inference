#!/usr/bin/env python3
"""
validate_e2e.py — End-to-end numerical validation of the Allen PVFinder pipeline
                  against the reference PyTorch model.

Pipeline stages compared:
  Stage 1 — FC aggregation only (no UNet)
             Allen: allen_fc_histogram.bin  [n_events, 40, 100]
             PyTorch: run FC layers + masked track sum + softplus on same tracks

  Stage 2 — NCW (FC output going into UNet)
             Allen: allen_ncw_features.bin  [n_events, 40, 8, 100]
             PyTorch: y0 = masked sum of layer6A outputs  [n_events*40, 8, 100]

  Stage 3 — Final KDE (FC + UNet)
             Allen: allen_kde_output.bin    [n_events, 40, 100]
             PyTorch: full forward()        [n_events*40, 100]

Reads from dump_dir:
  allen_track_features.bin  — [n_events, 40, 9, 250] padded track features (sentinel -99)
  allen_fc_histogram.bin    — [n_events, 40, 100]
  allen_ncw_features.bin    — [n_events, 40, 8, 100]
  allen_kde_output.bin      — [n_events, 40, 100]

Usage:
  python3 validate_e2e.py [--dump-dir validation_dump]
                           [--weights pvfinder_pytorch/weights/...pyt]
                           [--fc-weights fc_weights.bin]
                           [--device cpu|cuda]
                           [--plot] [--report e2e_report.json]
"""

import argparse, json, os, struct, sys
import numpy as np

parser = argparse.ArgumentParser(description="End-to-end Allen vs PyTorch PVFinder validation")
parser.add_argument("--dump-dir",   default="validation_dump")
parser.add_argument("--weights",
                    default="pvfinder_pytorch/weights/"
                            "07Sept2023_t2hists_HDplusUNet100_iter12Ca_200epochs_2em5_5p0_final.pyt")
parser.add_argument("--fc-weights", default="fc_weights.bin",
                    help="Binary FC weight file (same format used by Allen)")
parser.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--plot",       action="store_true")
parser.add_argument("--report",     default="")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
SEP  = "=" * 72
SEP2 = "-" * 72
MAGIC = 0xAB1E

def section(t):    print(f"\n{SEP}\n  {t}\n{SEP}")
def subsection(t): print(f"\n{SEP2}\n  {t}\n{SEP2}")

def read_dump(path, elems_per_event):
    with open(path, "rb") as f:
        raw = f.read()
    magic, n_events = struct.unpack_from("<II", raw, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic:#x}")
    data = np.frombuffer(raw, dtype=np.float32, offset=8).copy()
    total = n_events * elems_per_event
    if data.size != total:
        raise ValueError(f"{path}: expected {total} floats, got {data.size}")
    return n_events, data

def stat(label, arr):
    a = arr.flatten()
    p = np.percentile(a, [50, 90, 95, 99, 99.9, 100])
    print(f"  {label}:")
    print(f"    mean={a.mean():.4e}  std={a.std():.4e}  min={a.min():.4e}  max={a.max():.4e}")
    print(f"    P50={p[0]:.4e}  P90={p[1]:.4e}  P95={p[2]:.4e}  "
          f"P99={p[3]:.4e}  P99.9={p[4]:.4e}  Pmax={p[5]:.4e}")
    return p

def compare(name, allen, pytorch):
    """Print a full comparison block, return dict of metrics."""
    diff     = allen.astype(np.float64) - pytorch.astype(np.float64)
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / (np.abs(pytorch).astype(np.float64) + 1e-9)

    subsection(f"[{name}] Error statistics")
    print(f"  Max  abs diff  : {abs_diff.max():.6e}")
    print(f"  Mean abs diff  : {abs_diff.mean():.6e}")
    print(f"  Median abs diff: {np.median(abs_diff):.6e}")
    print(f"  RMS  diff      : {np.sqrt((diff**2).mean()):.6e}")
    print(f"  Max  rel diff  : {rel_diff.max():.6e}")
    print(f"  Mean rel diff  : {rel_diff.mean():.6e}")

    abs_p = np.percentile(abs_diff.flatten(), [50,90,95,99,99.9,100])
    rel_p = np.percentile(rel_diff.flatten(), [50,90,95,99,99.9,100])
    lbls  = ["P50","P90","P95","P99","P99.9","Pmax"]
    print(f"\n  {'Pct':6s}  {'Abs diff':>14s}  {'Rel diff':>14s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}")
    for l, av, rv in zip(lbls, abs_p, rel_p):
        print(f"  {l:6s}  {av:14.6e}  {rv:14.6e}")

    flat_a = allen.flatten().astype(np.float64)
    flat_p = pytorch.flatten().astype(np.float64)
    pearson = np.corrcoef(flat_p, flat_a)[0,1]
    ss_res  = np.sum((flat_a - flat_p)**2)
    ss_tot  = np.sum((flat_a - flat_a.mean())**2)
    r2      = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    print(f"\n  Pearson r : {pearson:.10f}")
    print(f"  R²        : {r2:.10f}")

    # signal region
    sig  = pytorch > 1e-3
    n_sig = sig.sum()
    sig_worst = float(abs_diff[sig].max()) if n_sig > 0 else 0.0

    # verdict
    worst = float(abs_diff.max())
    thrs = [("fp32_noise<1e-5",1e-5),("tight<1e-4",1e-4),
            ("acceptable<1e-3",1e-3),("loose<1e-2",1e-2)]
    verdict = "FAIL"
    for lbl, thr in thrs:
        if worst < thr:
            verdict = lbl
            break
    print(f"\n  Verdict: {verdict}  (worst abs={worst:.3e})")
    if n_sig > 0:
        sv = "PASS" if sig_worst < 1e-3 else "FAIL"
        print(f"  Signal-region (KDE>1e-3): worst={sig_worst:.3e}  {sv}")

    return dict(max_abs=worst, mean_abs=float(abs_diff.mean()),
                rms=float(np.sqrt((diff**2).mean())),
                pearson_r=float(pearson), r_squared=float(r2),
                abs_percentiles={l:float(v) for l,v in zip(lbls,abs_p)},
                signal_worst=sig_worst, verdict=verdict)

# ---------------------------------------------------------------------------
# Read Allen dumps
# ---------------------------------------------------------------------------
section("Loading Allen Dumps")

N_INT=40; N_FEAT=9; MAX_TRK=250; N_CH=8; W=100

trk_path  = os.path.join(args.dump_dir, "allen_track_features.bin")
fch_path  = os.path.join(args.dump_dir, "allen_fc_histogram.bin")
ncw_path  = os.path.join(args.dump_dir, "allen_ncw_features.bin")
kde_path  = os.path.join(args.dump_dir, "allen_kde_output.bin")

for p in [trk_path, fch_path, ncw_path, kde_path]:
    if not os.path.exists(p):
        print(f"  MISSING: {p}")
        print("  Run Allen with pvfinder_fc_aggregation.dump_validation=<dump_dir>")
        print("  and pvfinder_unet.dump_validation=<dump_dir>  then re-run this script.")
        sys.exit(1)
    print(f"  OK  : {p}  ({os.path.getsize(p)//1024} KB)")

n_events, trk_flat = read_dump(trk_path, N_INT * N_FEAT * MAX_TRK)
_,        fch_flat = read_dump(fch_path, N_INT * W)
_,        ncw_flat = read_dump(ncw_path, N_INT * N_CH * W)
_,        kde_flat = read_dump(kde_path, N_INT * W)

N = n_events * N_INT   # total intervals

# Shapes
# trk : [N_evt, 40, 9, 250]  → [N_evt*40, 9, 250]  (feature-major within interval)
# fch : [N_evt*40, 100]
# ncw : [N_evt*40, 8, 100]
# kde : [N_evt*40, 100]
allen_trk = trk_flat.reshape(N, N_FEAT, MAX_TRK)   # [N, 9, 250]
allen_fch = fch_flat.reshape(N, W)                  # [N, 100]
allen_ncw = ncw_flat.reshape(N, N_CH, W)            # [N, 8, 100]
allen_kde = kde_flat.reshape(N, W)                  # [N, 100]

print(f"\n  n_events={n_events}  intervals={N}")
print(f"  trk={allen_trk.shape}  fch={allen_fch.shape}  "
      f"ncw={allen_ncw.shape}  kde={allen_kde.shape}")

subsection("Allen dump statistics")
stat("Allen FC histogram", allen_fch)
stat("Allen NCW",          allen_ncw)
stat("Allen KDE",          allen_kde)

# ---------------------------------------------------------------------------
# Load PyTorch model + FC weights
# ---------------------------------------------------------------------------
section("Loading PyTorch Model")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pvfinder_pytorch"))

import torch
import torch.nn.functional as F
import types, sys as _sys
if "awkward" not in _sys.modules:
    _sys.modules["awkward"] = types.ModuleType("awkward")
from utils import TrackIntervalsToKDE_HDplusUNet100 as Model, combine

model = Model(20,20,20,20,20, latentChannels=8, n=64)
if not os.path.exists(args.weights):
    print(f"ERROR: weight file not found: {args.weights}"); sys.exit(1)
d = torch.load(args.weights, map_location="cpu", weights_only=True)
model.load_state_dict(d)
model.eval()
device = torch.device(args.device)
model  = model.to(device)
print(f"  UNet weights : {args.weights}")
print(f"  Device       : {device}")

# ---------------------------------------------------------------------------
# PyTorch full forward on Allen's per-interval track tensors
#
# allen_trk[i] = [9, 250] with sentinel -99 for empty slots
# PyTorch forward(x) expects x: [batch, 9, n_tracks] with mask x[:,0,:]>-98
# We feed each interval individually (batch=1) or in batches.
# ---------------------------------------------------------------------------
section("PyTorch Full Forward (FC + UNet) on Allen Track Data")

# Run in one batched call for efficiency
x_t = torch.tensor(allen_trk, dtype=torch.float32).to(device)  # [N, 9, 250]

with torch.no_grad():
    # ---- Replicate forward() manually to capture intermediate outputs ----
    nEvts   = x_t.shape[0]      # = N (intervals treated as batch)
    nTrks   = x_t.shape[2]      # 250

    # Mask: valid tracks have x[:,0,:] > -98
    mask = x_t[:, 0, :] > -98.
    filt = mask.float()                                   # [N, 250]
    f2   = filt.unsqueeze(2).expand(-1, -1, W)           # [N, 250, 100]

    x = x_t.transpose(1, 2)                              # [N, 250, 9]
    leaky = torch.nn.LeakyReLU(0.01)

    x = leaky(model.layer1(x))
    x = leaky(model.layer2(x))
    x = leaky(model.layer3(x))
    x = leaky(model.layer4(x))
    x = leaky(model.layer5(x))
    x = leaky(model.layer6A(x))                          # [N, 250, 800]

    # NO dropout in eval mode (model.fc6dropout is a no-op)
    x = model.fc6dropout(x)

    x  = x.view(nEvts, nTrks, 8, W)                     # [N, 250, 8, 100]
    f2 = f2.unsqueeze(2)                                 # [N, 250, 1, 100]
    x  = torch.mul(f2, x)
    y0 = torch.sum(x, dim=1)                             # [N, 8, 100]  ← NCW

    # Normalise by track count (Allen applies weight = 1/n_local)
    n_valid = mask.sum(dim=1).float().clamp(min=1)       # [N]
    y0_norm = y0 / n_valid.view(-1, 1, 1)                # [N, 8, 100]

    # ---- PyTorch FC-only histogram (replicate Allen's s_hist accumulation) ----
    # Allen accumulates: for each track t, for each bin k:
    #   chan_sum = sum_c( leaky_relu( layer6A_out[t, c*100+k] ) )
    #   s_hist[k] += softplus(chan_sum)
    # then divides by n_local.
    # We already have x_t (layer6A output) before the view, shape [N, 250, 800]
    # Recompute from saved x before view:
    with torch.no_grad():
        # Re-run L1-L6A cleanly
        xfc = x_t.transpose(1,2)
        xfc = leaky(model.layer1(xfc))
        xfc = leaky(model.layer2(xfc))
        xfc = leaky(model.layer3(xfc))
        xfc = leaky(model.layer4(xfc))
        xfc = leaky(model.layer5(xfc))
        xfc = leaky(model.layer6A(xfc))       # [N, 250, 800]
        xfc = xfc.view(nEvts, nTrks, 8, W)   # [N, 250, 8, 100]
        xfc_masked = xfc * filt.unsqueeze(2).unsqueeze(3)  # zero out invalid tracks
        # chan_sum per (interval, bin): sum over channels of leaky(layer6A)
        chan_sum = xfc_masked.sum(dim=2)       # [N, 250, 100]
        # softplus per track per bin, then sum over tracks, then divide
        fc_hist_pt = F.softplus(chan_sum).sum(dim=1) / n_valid.view(-1,1)  # [N, 100]

    # ---- UNet on y0_norm ----
    x1  = model.rcbn1(y0_norm)
    x2  = model.d(model.rcbn2(x1))
    x3  = model.d(model.rcbn3(x2))
    xu1 = model.up1(x3)
    xu2 = model.up2(combine(xu1, x2, mode=model.mode))
    xoi = model.out_intermediate(combine(xu2, x1, mode=model.mode))
    logits = model.outc(xoi)
    pt_kde = F.softplus(logits).squeeze(1) * 0.001       # [N, 100]

pt_fc_hist = fc_hist_pt.cpu().numpy().reshape(N, W)
pt_ncw     = y0_norm.cpu().numpy().reshape(N, N_CH, W)
pt_kde_np  = pt_kde.cpu().numpy().reshape(N, W)

subsection("PyTorch output statistics")
stat("PyTorch FC histogram", pt_fc_hist)
stat("PyTorch NCW",          pt_ncw)
stat("PyTorch KDE",          pt_kde_np)

# ---------------------------------------------------------------------------
# Three-stage comparison
# ---------------------------------------------------------------------------
section("Stage 1: FC Aggregation Histogram  (Allen vs PyTorch)")
r1 = compare("FC histogram", allen_fch, pt_fc_hist)

section("Stage 2: NCW Features  (Allen vs PyTorch)")
r2 = compare("NCW", allen_ncw, pt_ncw)

section("Stage 3: Final KDE  (Allen vs PyTorch, end-to-end)")
r3 = compare("KDE e2e", allen_kde, pt_kde_np)

# Per-event summary
section("Per-Event Breakdown (Final KDE)")
per_ev_max = np.abs(allen_kde - pt_kde_np).reshape(n_events, N_INT * W).max(axis=1)
for i in range(min(n_events, 20)):
    flag = " <-- WORST" if per_ev_max[i] == per_ev_max.max() else ""
    print(f"  event {i:4d}: max_abs={per_ev_max[i]:.4e}{flag}")

# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------
report = dict(n_events=int(n_events), n_intervals=N,
              weights=args.weights, device=str(device),
              stage1_fc_histogram=r1, stage2_ncw=r2, stage3_kde_e2e=r3,
              per_event_kde_max_abs=per_ev_max.tolist())
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

    plot_dir = os.path.join(args.dump_dir, "plots_e2e")
    os.makedirs(plot_dir, exist_ok=True)
    section(f"Generating Plots → {plot_dir}")

    def save_comparison_plots(allen_arr, pt_arr, stage_name, prefix):
        flat_a = allen_arr.flatten().astype(np.float64)
        flat_p = pt_arr.flatten().astype(np.float64)
        abs_d  = np.abs(flat_a - flat_p)

        # Error histogram
        fig, ax = plt.subplots(figsize=(8,5))
        nz = abs_d[abs_d > 0]
        if len(nz):
            ax.hist(np.log10(nz+1e-12), bins=100, color="steelblue",
                    edgecolor="none", alpha=0.8)
        for thr, col, lbl in [(1e-5,"green","1e-5"),(1e-4,"orange","1e-4"),
                               (1e-3,"red","1e-3"),(1e-2,"darkred","1e-2")]:
            ax.axvline(np.log10(thr), color=col, ls="--", lw=1.5, label=lbl)
        ax.set_xlabel("log10(|Allen − PyTorch|)")
        ax.set_ylabel("Count")
        ax.set_title(f"{stage_name} — absolute error distribution")
        ax.legend(fontsize=8)
        p = os.path.join(plot_dir, f"{prefix}_err_hist.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p}")

        # Scatter (log-log)
        above = (flat_p > 1e-6) & (flat_a > 1e-6)
        if above.sum() > 100:
            fig, ax = plt.subplots(figsize=(6,6))
            r = np.corrcoef(flat_p[above], flat_a[above])[0,1]
            ax.scatter(flat_p[above], flat_a[above], s=0.3, alpha=0.2, c="steelblue")
            lim = max(flat_p[above].max(), flat_a[above].max()) * 1.1
            ax.plot([0,lim],[0,lim],"r--",lw=1, label=f"y=x  r={r:.8f}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("PyTorch"); ax.set_ylabel("Allen")
            ax.set_title(f"{stage_name} — scatter (log-log)")
            ax.legend(fontsize=8)
            p = os.path.join(plot_dir, f"{prefix}_scatter.png")
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  Saved: {p}")

    save_comparison_plots(allen_fch, pt_fc_hist, "FC histogram", "01_fc")
    save_comparison_plots(allen_ncw,  pt_ncw,     "NCW",          "02_ncw")
    save_comparison_plots(allen_kde,  pt_kde_np,  "Final KDE",    "03_kde")

    # KDE overlay — top 6 peak intervals
    N_PLOT   = min(6, N)
    top_idxs = np.argsort(pt_kde_np.max(axis=1))[::-1][:N_PLOT]
    fig, axes = plt.subplots(2, 3, figsize=(16,9))
    axes = axes.flatten()
    for ax, idx in zip(axes, top_idxs):
        ev_id  = idx // N_INT
        int_id = idx %  N_INT
        ax.plot(pt_kde_np[idx],  label="PyTorch", color="crimson",   lw=1.8)
        ax.plot(allen_kde[idx],  label="Allen",   color="steelblue", lw=1.8, ls="--")
        ax.fill_between(range(W),
                        np.abs(pt_kde_np[idx]-allen_kde[idx]),
                        alpha=0.3, color="orange", label="|diff|")
        ax.set_title(f"ev {ev_id}  int {int_id}", fontsize=9)
        ax.set_xlabel("bin"); ax.set_ylabel("KDE")
        ax.legend(fontsize=7)
    fig.suptitle("End-to-end Allen vs PyTorch — final KDE (top-peak intervals)",
                 fontweight="bold")
    plt.tight_layout()
    p = os.path.join(plot_dir, "04_kde_overlay.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")

    # Per-event bar chart
    fig, ax = plt.subplots(figsize=(max(8, n_events//4), 4))
    colors = ["crimson" if v>1e-3 else ("orange" if v>1e-4 else "steelblue")
              for v in per_ev_max]
    ax.bar(range(n_events), per_ev_max, color=colors)
    ax.axhline(1e-3, color="red",    ls="--", lw=1.2, label="1e-3")
    ax.axhline(1e-4, color="orange", ls="--", lw=1.2, label="1e-4")
    ax.set_yscale("log")
    ax.set_xlabel("Event"); ax.set_ylabel("Max |diff| (KDE)")
    ax.set_title("Per-event max absolute KDE difference (end-to-end)")
    ax.legend(fontsize=8)
    p = os.path.join(plot_dir, "05_per_event_kde_diff.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

print(f"\n{'='*72}\n  Done.\n{'='*72}\n")
