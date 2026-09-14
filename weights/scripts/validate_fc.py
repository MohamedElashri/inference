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
                   histogram[k]      = softplus_allen(sum_c s[c*100+k]) / n_local
                   with softplus_allen(x) = x if x > 0 else log(1 + exp(x))

A wrong FC weight file (e.g. a transposed layer6A) shows up as a large
mismatch here; the UNet validator (validate_unet.py) cannot see it because it
starts from Allen's FC output.

Dump: run Allen with pvfinder_fc_aggregation.dump_validation=<dir>. Files carry
a header uint32 {0xFC01, n_events, n_tracks, n_latent_channels}.

Usage:
    python3 tools/validate_fc.py --dump-dir DIR [--weights MODEL.pyt] [--fc-bin fc_weights.bin]
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Validate Allen FC aggregation against a checkpoint")
parser.add_argument("--dump-dir", required=True,
                    help="directory written by pvfinder_fc_aggregation.dump_validation")
parser.add_argument("--weights",
                    default=os.path.join(
                        REPO_ROOT, "pvfinder_pytorch", "weights", "16-channel",
                        "FCN-20-channels_UNet-16-channels_nBinsPerSlice-100_latentChannels-8_iter9_final.pyt"),
                    help="checkpoint (.pyt) the Allen run is supposed to implement")
parser.add_argument("--fc-bin", default="",
                    help="optional: the fc_weights .bin Allen loaded; checked against the checkpoint")
parser.add_argument("--threshold", type=float, default=1e-3,
                    help="max relative difference for PASS (default 1e-3)")
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

# ---------------------------------------------------------------------------
# Optional: is the .bin Allen loaded the checkpoint, in the layout Allen expects?
# ---------------------------------------------------------------------------
if args.fc_bin:
    blob = np.fromfile(args.fc_bin, dtype=np.float32)
    expected = np.concatenate([np.concatenate([W[k].astype(np.float32).ravel(), B[k].astype(np.float32)])
                               for k in ("1", "2", "3", "4", "5", "6A")])
    if blob.size != expected.size:
        print(f"fc-bin: {args.fc_bin} has {blob.size} floats, checkpoint needs {expected.size}  -> MISMATCH")
        ok = False
    elif np.array_equal(blob, expected):
        print(f"fc-bin: {args.fc_bin} == checkpoint in Allen's layout  -> OK")
    else:
        w6a = blob[1880:1880 + L6A * 20]
        head_same = np.array_equal(blob[:1880], expected[:1880])
        if head_same and np.array_equal(w6a, W["6A"].T.astype(np.float32).ravel()):
            why = "layer6A is transposed (Allen transposes again on load)"
        elif not head_same:
            why = "layers 1-5 differ: this file is from a different checkpoint"
        else:
            why = "layer6A differs"
        print(f"fc-bin: {args.fc_bin} does NOT match the checkpoint: {why}  -> MISMATCH")
        ok = False

# ---------------------------------------------------------------------------
# Recompute FC per track, then aggregate with Allen's CSR
# ---------------------------------------------------------------------------
def leaky(x):
    return np.where(x > 0, x, 0.01 * x)


h = feats
for k in ("1", "2", "3", "4", "5"):
    h = leaky(h @ W[k].T + B[k])
z = leaky(h @ W["6A"].T + B["6A"])                      # [n_tracks, L6A]

ref_feat = np.zeros((n_events, 40, L6A))
ref_hist = np.zeros((n_events, 40, 100))
for e in range(n_events):
    off = int(offsets[e])
    n_entries = int(csr[e, 41])
    gidx = off + track_idx[off * 2: off * 2 + n_entries].astype(np.int64)
    for iv in range(40):
        a, b = int(csr[e, iv]), int(csr[e, iv + 1])
        n_local = b - a
        if n_local == 0:
            continue
        s = z[gidx[a:b]].sum(axis=0)
        ref_feat[e, iv] = s / n_local
        chan = s.reshape(n_latent, 100).sum(axis=0)
        sp = np.where(chan > 0, chan, np.log1p(np.exp(np.minimum(chan, 0))))
        ref_hist[e, iv] = sp / n_local


def compare(name, allen, ref):
    global ok
    finite_slot = np.isfinite(ref).all(axis=-1) & np.isfinite(allen).all(axis=-1)
    n_bad = int((~finite_slot).sum())
    a = allen[finite_slot].astype(np.float64)
    r = ref[finite_slot]
    abs_d = np.abs(a - r)
    rel_d = abs_d / np.maximum(np.abs(r), 1.0)
    worst = float(rel_d.max()) if rel_d.size else 0.0
    status = "PASS" if worst < args.threshold else "FAIL"
    print(f"{name}: slots compared {int(finite_slot.sum())}, non-finite slots skipped {n_bad}; "
          f"max|diff| {abs_d.max():.3e}, max rel diff {worst:.3e}  -> {status}")
    if status == "FAIL":
        ok = False


compare("interval features", ifeat, ref_feat)
compare("histogram        ", hist, ref_hist)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
