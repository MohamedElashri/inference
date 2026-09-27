#!/usr/bin/env python3
"""End-to-end check of Allen's PVFinder against the trained PyTorch model.

validate_fc.py and validate_unet.py each recompute one Allen stage from that
stage's own dumped input, so they prove the arithmetic but not that Allen
feeds the network what it was trained on. This script starts from Allen's raw
per-track features (allen_fc_track_features.bin), builds the network input
exactly as the training data was built (pv-finder t2hists arrays: per interval
of 100 bins / 10 mm, tracks within the interval +/- 2.5 mm that pass the
sigma cuts, features (z - interval edge, x, y, A..F), padded with -99 to 250
tracks), runs the full PyTorch model (FC + sum over tracks + UNet) and
compares its KDE with Allen's (allen_kde_output.bin).

It also prints physics-level numbers that separate a network fed the right
inputs from one that is not: the fraction of intervals with a KDE peak above
1e-3 (about 13% on the training team's validation sample) and the number of
peaks per event.

    python3 weights/scripts/validate_model.py --dump-dir DIR --weights MODEL.pyt [--report R.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

WEIGHTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(WEIGHTS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "pvfinder_pytorch"))
from utils import TrackIntervalsToKDE_HDplusUNet100 as Model  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_agreement as pa  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--dump-dir", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--max-tracks", type=int, default=250, help="tracks per interval in the training arrays")
parser.add_argument("--threshold", type=float, default=1e-3, help="PASS limit on max |Allen - PyTorch| KDE (FP32 path)")
parser.add_argument("--report", default="")
args = parser.parse_args()


def read(name, dtype, header_words):
    raw = np.fromfile(os.path.join(args.dump_dir, name), dtype=np.uint8)
    return np.frombuffer(raw[:4 * header_words].tobytes(), np.uint32), \
        np.frombuffer(raw[4 * header_words:].tobytes(), dtype)


hdr, feats = read("allen_fc_track_features.bin", np.float32, 4)
n_events, n_tracks = int(hdr[1]), int(hdr[2])
feats = feats.reshape(n_tracks, 9)
_, offsets = read("allen_fc_track_offsets.bin", np.uint32, 4)
_, kde = read("allen_kde_output.bin", np.float32, 2)
allen = kde.reshape(n_events, 40, 100)

# --- network input, built like the training arrays ----------------------------
x, y, z = feats[:, 0], feats[:, 1], feats[:, 2]
with np.errstate(divide="ignore", invalid="ignore"):
    sig = np.sqrt(np.abs(1.0 / feats[:, 3:6].astype(np.float64)))
    selected = (sig[:, 2] < 2) & (np.abs(x / sig[:, 0]) < 4) & (np.abs(y / sig[:, 1]) < 4)
X = np.full((n_events * 40, 9, args.max_tracks), -99.0, dtype=np.float32)
n_in = np.zeros(n_events * 40, dtype=np.int64)   # tracks per interval, before the cap
n_capped = 0
for e in range(n_events):
    lo_t = int(offsets[e])
    hi_t = int(offsets[e + 1]) if e + 1 < n_events else n_tracks
    ev_z, ev_sel, ev_f = z[lo_t:hi_t], selected[lo_t:hi_t], feats[lo_t:hi_t]
    for iv in range(40):
        lo = -100.0 + 10.0 * iv
        idx = np.nonzero(ev_sel & (ev_z > lo - 2.5) & (ev_z < lo + 12.5))[0]
        n_in[e * 40 + iv] = len(idx)
        if len(idx) > args.max_tracks:   # training keeps the smallest sigma_z
            n_capped += 1
            idx = idx[np.argsort(sig[lo_t + idx, 2])[:args.max_tracks]]
        if len(idx) == 0:
            continue
        f = ev_f[idx]
        rows = np.column_stack([f[:, 2] - lo, f[:, 0], f[:, 1], f[:, 3:9]])
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        X[e * 40 + iv, :, :len(idx)] = rows.T

# --- full PyTorch model -------------------------------------------------------
sd = torch.load(args.weights, map_location="cpu")
latent = sd["layer6A.weight"].shape[0] // 100
n_feat = sd["rcbn1.0.weight"].shape[0]
model = Model(20, 20, 20, 20, 20, latentChannels=latent, n=n_feat)
model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
model.eval()
with torch.no_grad():
    ref = torch.cat([model(torch.from_numpy(X[i:i + 4000])) for i in range(0, len(X), 4000)]).numpy()
ref = ref.reshape(n_events, 40, 100)

# --- comparison and physics-level summary ------------------------------------
# With pvfinder_fc_aggregation.skip_empty_intervals, intervals with fewer than
# min_interval_tracks tracks do not go through the UNet: Allen gives them the
# network's response to an empty interval. With the minimum at 1 that is
# exact; above 1 it is a deliberate approximation, so those intervals are left
# out of the comparison and reported separately.
min_tracks = 0
bf16_path = pa.bf16_path(args.dump_dir)
cfg_path = os.path.join(args.dump_dir, "config.json")
if os.path.isfile(cfg_path):
    with open(cfg_path) as fp:
        cfg = json.load(fp)
    fc_cfg = cfg.get("pvfinder_fc_aggregation", {})
    if fc_cfg.get("skip_empty_intervals", False):
        min_tracks = int(fc_cfg.get("min_interval_tracks", 1))
skipped = ((n_in > 0) & (n_in < min_tracks)).reshape(n_events, 40)
d = np.abs(allen.astype(np.float64) - ref)
d_skipped = d[skipped]
d[skipped] = 0.0
worst = float(d.max())
peaks = pa.peak_agreement(allen, ref)
finite = bool(np.isfinite(allen).all())
if bf16_path:
    criterion = pa.CRITERION
    ok = pa.passes(peaks)
else:
    criterion = f"FP32 path: max |Allen - PyTorch| < {args.threshold:g}"
    ok = worst < args.threshold
status = "PASS" if ok and finite else "FAIL"
peak_a, peak_r = allen.max(-1), ref.max(-1)
non_empty = (X[:, 0, :] > -98).any(1).reshape(n_events, 40)
summary = {
    "n_events": n_events,
    "empty_interval_fraction": float(1 - non_empty.mean()),
    "intervals_capped_at_max_tracks": n_capped,
    "max_abs_diff": worst,
    "mean_abs_diff": float(d.mean()),
    "intervals_with_peak_above_1e-3": {"allen": float((peak_a > 1e-3).mean()), "pytorch": float((peak_r > 1e-3).mean())},
    "bins_above_1e-3_per_event": {"allen": float((allen > 1e-3).sum() / n_events), "pytorch": float((ref > 1e-3).sum() / n_events)},
    "min_interval_tracks": min_tracks,
    "skipped_intervals": {
        "count": int(skipped.sum()),
        "fraction_of_non_empty": float(skipped.sum() / max(int(non_empty.sum()), 1)),
        "pytorch_max_kde": float(peak_r[skipped].max()) if skipped.any() else 0.0,
        "pytorch_intervals_with_peak_above_1e-3": int((peak_r[skipped] > 1e-3).sum()),
        "pytorch_intervals_with_peak_above_1e-3_all": int((peak_r > 1e-3).sum()),
        "max_abs_diff": float(d_skipped.max()) if d_skipped.size else 0.0,
    },
    "threshold": args.threshold,
    "bf16_path": bf16_path,
    "peaks": peaks,
    "criterion": criterion,
    "status": status,
}
print(f"events {n_events}, empty intervals {summary['empty_interval_fraction']:.3f}, "
      f"intervals over {args.max_tracks} tracks {n_capped}")
print(f"Allen vs PyTorch (full model from Allen's track features): max |diff| {worst:.3e}, "
      f"mean {summary['mean_abs_diff']:.3e}")
print(f"intervals with a KDE peak > 1e-3: Allen {summary['intervals_with_peak_above_1e-3']['allen']:.4f}, "
      f"PyTorch {summary['intervals_with_peak_above_1e-3']['pytorch']:.4f} (training validation sample: ~0.13)")
print(f"KDE bins > 1e-3 per event: Allen {summary['bins_above_1e-3_per_event']['allen']:.1f}, "
      f"PyTorch {summary['bins_above_1e-3_per_event']['pytorch']:.1f}")
if min_tracks > 1:
    sk = summary["skipped_intervals"]
    print(f"intervals with 1..{min_tracks - 1} tracks (not run through the UNet, min_interval_tracks={min_tracks}): "
          f"{sk['count']} ({100 * sk['fraction_of_non_empty']:.1f}% of non-empty), excluded from the diff above; "
          f"PyTorch there: max KDE {sk['pytorch_max_kde']:.3e}, {sk['pytorch_intervals_with_peak_above_1e-3']} of "
          f"{sk['pytorch_intervals_with_peak_above_1e-3_all']} intervals with a peak > 1e-3; "
          f"max |Allen - PyTorch| there {sk['max_abs_diff']:.3e}")
print(pa.describe(peaks))
print(f"criterion ({criterion}): {status}")
print(status)
if args.report:
    with open(args.report, "w") as fp:
        json.dump(summary, fp, indent=2)
sys.exit(0 if status == "PASS" else 1)
