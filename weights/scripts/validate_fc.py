#!/usr/bin/env python3
"""
validate_fc.py - Check Allen's FC aggregation against the trained checkpoint.

Recomputes, in float64, exactly what pvfinder_fc_aggregation computes, from the
checkpoint's FC layers and Allen's own dumped track features and
track-to-interval assignment, then compares with Allen's outputs:

  per track t:     h = LeakyReLU(L5(...LeakyReLU(L1(features_t))))
                   z_t = LeakyReLU(W6A h + b6A)                 [latent*100]
  per interval:    s = sum of z_t over the interval's tracks (CSR, boundary
                       tracks counted in both intervals)
                   interval_features = s / n_local
                   histogram[k]      = softplus(sum_c s[c*100+k]) / n_local
                   with softplus(x) = log(1 + exp(x))

A wrong FC weight file (e.g. a transposed layer6A) shows up as a large
mismatch here; the UNet validator (validate_unet.py) cannot see it because it
starts from Allen's FC output.

Comparison metric: float32 rounding error scales with the magnitude of the
terms being added, not with the (possibly cancelled) result. Some checkpoints
have intermediate sums of order 1e6, so a correct float32 implementation can
differ from the float64 reference by O(0.1) on O(1) outputs. Each difference is
therefore measured in float32 ulps of a magnitude envelope propagated through
the network (sum_j |W_ij| * envelope_j + |b_i| per layer, averaged per interval
like the features). The envelope over-estimates the true term magnitudes, so
the limit is one ulp: on the 16-channel models correct runs measure at most
~1e-3 ulps, while a transposed layer 6A measures ~700 (histogram) to ~5000
(interval features).

Dump: make -C weights dump MODEL=<name>, or run Allen with
pvfinder_fc_aggregation.dump_validation=<dir>. Files carry
a header uint32 {0xFC01, n_events, n_tracks, n_latent_channels}.

Usage:
    make -C weights validate MODEL=<name>        (normal use, after make dump)
    python3 weights/scripts/validate_fc.py --dump-dir DIR [--weights MODEL.pyt]
"""

import argparse
import os
import sys

import numpy as np

# weights/scripts/ -> weights/ -> repository root
WEIGHTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(WEIGHTS_DIR)

parser = argparse.ArgumentParser(description="Validate Allen FC aggregation against a checkpoint")
parser.add_argument("--dump-dir", required=True,
                    help="directory written by pvfinder_fc_aggregation.dump_validation")
parser.add_argument("--weights",
                    default=os.path.join(WEIGHTS_DIR, "checkpoints", "unet16_lc4_scnone_asym5_final.pyt"),
                    help="checkpoint (.pyt) the Allen run is supposed to implement")
parser.add_argument("--max-f32-ulps", type=float, default=1.0,
                    help="PASS limit on |Allen - reference|, in float32 rounding steps (ulps) of the "
                         "magnitude envelope feeding each output (default 1)")
parser.add_argument("--report", default="",
                    help="optional: write the results as JSON to this path")
args = parser.parse_args()

MAGIC = 0xFC01


def read(name, dtype):
    path = os.path.join(args.dump_dir, name)
    raw = np.fromfile(path, dtype=np.uint8)
    header = np.frombuffer(raw[:16].tobytes(), dtype=np.uint32)
    if header[0] != MAGIC:
        sys.exit(f"{path}: bad magic {header[0]:#x}")
    return header, np.frombuffer(raw[16:].tobytes(), dtype=dtype)


header, csr = read("allen_fc_csr.bin", np.int32)
n_events, n_tracks, n_latent = (int(v) for v in header[1:])
_, track_idx = read("allen_fc_track_idx.bin", np.int32)
_, offsets = read("allen_fc_track_offsets.bin", np.uint32)
_, feats = read("allen_fc_track_features.bin", np.float32)
_, ifeat = read("allen_fc_interval_features.bin", np.float32)
_, hist = read("allen_fc_histogram.bin", np.float32)

L6A = n_latent * 100
csr = csr.reshape(n_events, 42)
feats = feats.reshape(n_tracks, 9).astype(np.float64)
ifeat = ifeat.reshape(n_events, 40, L6A)
hist = hist.reshape(n_events, 40, 100)
print(f"dump: {n_events} events, {n_tracks} tracks, latentChannels={n_latent}")

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
import torch

sd = torch.load(args.weights, map_location="cpu")
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()
W = {k: sd[f"layer{k}.weight"].double().numpy() for k in ("1", "2", "3", "4", "5", "6A")}
B = {k: sd[f"layer{k}.bias"].double().numpy() for k in ("1", "2", "3", "4", "5", "6A")}
if W["6A"].shape != (L6A, 20):
    sys.exit(f"checkpoint layer6A is {W['6A'].shape}, but the Allen build uses latentChannels={n_latent} "
             f"({L6A}x20): build and checkpoint describe different models")
print(f"checkpoint: {args.weights}")

ok = True
report = {"dump_dir": args.dump_dir, "weights": args.weights, "max_f32_ulps": args.max_f32_ulps}

# ---------------------------------------------------------------------------
# Recompute FC per track, then aggregate with Allen's CSR
# ---------------------------------------------------------------------------
def leaky(x):
    return np.where(x > 0, x, 0.01 * x)


# Allen's CSR (which tracks feed which interval) against the training rules:
# interval i covers z in [-100 + 10 i, -90 + 10 i) extended by 2.5 mm each
# side; tracks need sigma_z < 2 and |x| / sigma_x, |y| / sigma_y < 4, with
# sigma = 1 / sqrt(|A|), 1 / sqrt(|B|), 1 / sqrt(|C|).
x_, y_, z_ = feats[:, 0], feats[:, 1], feats[:, 2]
with np.errstate(divide="ignore", invalid="ignore"):
    sig = np.sqrt(np.abs(1.0 / feats[:, 3:6]))
    selected = (sig[:, 2] < 2) & (np.abs(x_ / sig[:, 0]) < 4) & (np.abs(y_ / sig[:, 1]) < 4)
csr_bad = 0
for e in range(n_events):
    off = int(offsets[e])
    n_ev_tracks = (int(offsets[e + 1]) if e + 1 < n_events else n_tracks) - off
    zz, sel = z_[off:off + n_ev_tracks], selected[off:off + n_ev_tracks]
    n_entries = int(csr[e, 41])
    local = track_idx[off * 2: off * 2 + n_entries]
    for iv in range(40):
        lo = -100.0 + 10.0 * iv
        want = set(np.nonzero(sel & (zz > lo - 2.5) & (zz < lo + 12.5))[0].tolist())
        got = set(local[csr[e, iv]:csr[e, iv + 1]].tolist())
        csr_bad += want != got
status = "OK" if csr_bad == 0 else "MISMATCH"
print(f"track-to-interval assignment vs training rules: {csr_bad} of {n_events * 40} intervals differ  -> {status}")
report["csr"] = {"intervals_differing": int(csr_bad), "status": status}
ok &= csr_bad == 0

# One network input per (track, interval) entry, in the training order
# (z - interval lower edge, x, y, A..F).
entry_in, entry_slot = [], []
for e in range(n_events):
    off = int(offsets[e])
    n_entries = int(csr[e, 41])
    gidx = off + track_idx[off * 2: off * 2 + n_entries].astype(np.int64)
    for iv in range(40):
        a, b = int(csr[e, iv]), int(csr[e, iv + 1])
        if a == b:
            continue
        f = feats[gidx[a:b]]
        entry_in.append(np.column_stack([f[:, 2] - (-100.0 + 10.0 * iv), f[:, 0], f[:, 1], f[:, 3:9]]))
        entry_slot.append(np.full(b - a, e * 40 + iv))
entry_in = np.concatenate(entry_in)
entry_slot = np.concatenate(entry_slot)

h = entry_in
envelope = np.abs(entry_in)                             # magnitude of the terms float32 adds
for k in ("1", "2", "3", "4", "5"):
    h = leaky(h @ W[k].T + B[k])
    envelope = envelope @ np.abs(W[k]).T + np.abs(B[k])
z = leaky(h @ W["6A"].T + B["6A"])                      # [n_entries, L6A]
envelope = envelope @ np.abs(W["6A"]).T + np.abs(B["6A"])

n_slots = n_events * 40
s_sum = np.zeros((n_slots, L6A)); np.add.at(s_sum, entry_slot, z)
e_sum = np.zeros((n_slots, L6A)); np.add.at(e_sum, entry_slot, envelope)
n_loc = np.bincount(entry_slot, minlength=n_slots).astype(np.float64)
nz = n_loc > 0
ref_feat = np.zeros((n_slots, L6A)); env_feat = np.zeros((n_slots, L6A))
ref_hist = np.zeros((n_slots, 100)); env_hist = np.zeros((n_slots, 100))
ref_feat[nz] = s_sum[nz]                                  # UNet input: sum over the interval's tracks
env_feat[nz] = e_sum[nz]
chan = s_sum[nz].reshape(-1, n_latent, 100).sum(axis=1)
ref_hist[nz] = np.logaddexp(0.0, chan) / n_loc[nz, None]   # exact softplus
env_hist[nz] = e_sum[nz].reshape(-1, n_latent, 100).sum(axis=1) / n_loc[nz, None]
ref_feat = ref_feat.reshape(n_events, 40, L6A); env_feat = env_feat.reshape(n_events, 40, L6A)
ref_hist = ref_hist.reshape(n_events, 40, 100); env_hist = env_hist.reshape(n_events, 40, 100)


F32_EPS = float(np.finfo(np.float32).eps)
BF16_EPS = 2.0 ** -7   # bfloat16 keeps 8 significant bits

# pvfinder_fc_aggregation.precision = bfloat16 (unet_input_dtype and
# l6a_dtype in older dumps): the features are stored as bfloat16, rounded on
# purpose, so measure them in bfloat16 ulps instead.
features_eps, features_unit = F32_EPS, "float32"
# A bfloat16 L6A (pvfinder_fc_aggregation.l6a_dtype) rounds its inputs, so
# every per-entry term is off by bfloat16 rounding: both outputs are measured
# in bfloat16 ulps.
hist_eps, hist_unit = F32_EPS, "float32"
# With skip_empty_intervals, intervals with fewer than min_interval_tracks
# tracks get no feature row (the UNet skips them), so the dump holds zeros
# there: leave those slots out of the feature comparison (their histogram is
# still written and compared).
min_tracks = 0
cfg_path = os.path.join(args.dump_dir, "config.json")
if os.path.isfile(cfg_path):
    import json
    with open(cfg_path) as fp:
        fc_cfg = json.load(fp).get("pvfinder_fc_aggregation", {})
    if fc_cfg.get("skip_empty_intervals", False):
        min_tracks = int(fc_cfg.get("min_interval_tracks", 1))
    if fc_cfg.get("precision") == "bfloat16":
        features_eps, features_unit = BF16_EPS, "bfloat16"
        hist_eps, hist_unit = BF16_EPS, "bfloat16"
    if fc_cfg.get("unet_input_dtype") == "bfloat16":
        features_eps, features_unit = BF16_EPS, "bfloat16"
    l6a = fc_cfg.get("l6a_dtype", "auto")
    if l6a == "bfloat16" or (l6a == "auto" and fc_cfg.get("fc_fused", True)
                             and fc_cfg.get("unet_input_dtype") == "bfloat16"):
        features_eps, features_unit = BF16_EPS, "bfloat16"
        hist_eps, hist_unit = BF16_EPS, "bfloat16"
report["features_dtype"] = features_unit
report["histogram_dtype"] = hist_unit


def compare(name, allen, ref, env, eps=F32_EPS, unit="float32"):
    global ok
    finite_slot = np.isfinite(ref).all(axis=-1) & np.isfinite(allen).all(axis=-1) & np.isfinite(env).all(axis=-1)
    n_bad = int((~finite_slot).sum())
    a = allen[finite_slot].astype(np.float64)
    r = ref[finite_slot]
    abs_d = np.abs(a - r)
    rel_d = abs_d / np.maximum(np.abs(r), 1.0)
    ulps = abs_d / (eps * np.maximum(env[finite_slot], 1.0))
    worst = float(ulps.max()) if ulps.size else 0.0
    status = "PASS" if worst < args.max_f32_ulps else "FAIL"
    print(f"{name}: slots compared {int(finite_slot.sum())}, non-finite slots skipped {n_bad}; "
          f"max|diff| {abs_d.max():.3e}, max rel diff {rel_d.max():.3e}, "
          f"max error {worst:.3g} {unit} ulps (limit {args.max_f32_ulps:g}; "
          f"largest term magnitude {env[finite_slot].max():.3e})  -> {status}")
    if status == "FAIL":
        ok = False
    report[name.strip().replace(" ", "_")] = {
        "slots_compared": int(finite_slot.sum()), "non_finite_slots": n_bad,
        "max_abs_diff": float(abs_d.max()), "max_rel_diff": float(rel_d.max()),
        "max_f32_ulps": worst, "largest_term": float(env[finite_slot].max()), "status": status,
    }


no_row = ((n_loc > 0) & (n_loc < min_tracks)).reshape(n_events, 40)
report["feature_slots_without_row"] = int(no_row.sum())
if no_row.any():
    print(f"interval features: {int(no_row.sum())} slots with 1..{min_tracks - 1} tracks have no row "
          f"(min_interval_tracks={min_tracks}), not compared")
feat_nan = np.where(no_row[..., None], np.nan, 0.0)   # non-finite slots are skipped by compare()
compare("interval features", ifeat + feat_nan, ref_feat, env_feat, features_eps, features_unit)
compare("histogram        ", hist, ref_hist, env_hist, hist_eps, hist_unit)
print("PASS" if ok else "FAIL")
if args.report:
    import json
    report.update(n_events=int(n_events), n_tracks=int(n_tracks), latent_channels=int(n_latent),
                  status="PASS" if ok else "FAIL")
    with open(args.report, "w") as fp:
        json.dump(report, fp, indent=2)
sys.exit(0 if ok else 1)
