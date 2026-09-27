#!/usr/bin/env python3
"""Check Allen's 9 per-track PVFinder input features, track by track.

validate_model.py checks the network given Allen's features; this checks the
features themselves (PVFinderTrackFeatures.cuh), in three parts:

1. Allen vs the rules, per track. From each track's VELO Kalman state and the
   beamline (allen_fc_track_states.bin, allen_fc_beamline.bin, written by the
   FC validation dump), recompute in double precision the POCA and the three
   ellipsoid axes, and get A..F from the axes with the training team's own
   Compute_tracks_ellipsoid (pv-finder_v2 tools/ellipsoid.py, --training-repo).
   Compare with allen_fc_track_features.bin.
2. The rules vs the training sample, per track (--training-h5). The raw
   training file stores each track's state (recon_x, y, z, tx, ty) and the
   POCA and axes the training ntuples computed from it. Rebuild the POCA, the
   axis directions and the major / minor length ratio from the state with the
   same rules and compare with the stored values.
3. The one input with no per-track reference: the minor-axis length. Allen
   takes it as sqrt(c00) of the Kalman state; its distribution is compared
   with the training sample's minor-axis lengths.

    python3 weights/scripts/validate_features.py --dump-dir DIR \\
        [--training-repo PATH] [--training-h5 FILE] [--report R.json]
"""
import argparse
import json
import os
import sys
import types
import zlib

import numpy as np

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--dump-dir", required=True)
parser.add_argument("--training-repo", default="/data/home/melashri/iris/model_work/pv-finder_v2",
                    help="pv-finder_v2 checkout, for tools/ellipsoid.py (skipped if absent)")
parser.add_argument("--training-h5", default="/share/lazy/sokoloff/ML-data_AA/pv_HLT1CPU_MinBiasMagDown_14Nov.h5",
                    help="raw training file, for parts 2 and 3 (skipped if absent)")
parser.add_argument("--training-events", type=int, default=2000)
parser.add_argument("--rtol", type=float, default=1e-4,
                    help="PASS limit on Allen vs recomputed ellipsoid (relative matrix difference)")
parser.add_argument("--poca-tol", type=float, default=1e-3, help="PASS limit on Allen vs recomputed POCA, mm")
parser.add_argument("--report", default="")
args = parser.parse_args()
report = {}


def read_dump(name, width):
    raw = np.fromfile(os.path.join(args.dump_dir, name), dtype=np.uint8)
    hdr = np.frombuffer(raw[:16].tobytes(), np.uint32)
    return hdr, np.frombuffer(raw[16:].tobytes(), np.float32).reshape(-1, width).astype(np.float64)


def poca_and_axes(x, y, z, tx, ty):
    """POCA to the beam (z axis of the frame) and unit axes e1, e2, e3 = track.

    The rules of PVFinderTrackFeatures.cuh, in double precision."""
    x0, y0 = x - z * tx, y - z * ty
    t_sq = tx * tx + ty * ty
    pz = np.where(t_sq > 1e-8, -(x0 * tx + y0 * ty) / np.where(t_sq > 1e-8, t_sq, 1.0), 0.0)
    px, py = x0 + tx * pz, y0 + ty * pz
    d = np.stack([tx, ty, np.ones_like(tx)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    sin_t = np.hypot(d[:, 0], d[:, 1])
    safe = np.where(sin_t > 0, sin_t, 1.0)
    e1 = np.where((sin_t > 0)[:, None], np.stack([-d[:, 1] / safe, d[:, 0] / safe, np.zeros_like(safe)], 1),
                  np.array([1.0, 0.0, 0.0]))
    e2 = np.cross(d, e1)
    ratio = np.where(sin_t > 0, np.minimum(d[:, 2] / safe, 2048.0), 2048.0)
    return px, py, pz, e1, e2, d, ratio


def ellipsoid_ABCDEF(u1, u2, u3):
    """A..F from the three axes (vectors with lengths), by the training code if available."""
    tools = os.path.join(args.training_repo, "tools", "ellipsoid.py")
    names = ("minor_axis1", "minor_axis2", "major_axis")
    if os.path.isfile(tools):
        sys.path.insert(0, args.training_repo)
        # ellipsoid.py imports Timer from utils.utilities (which needs pandas)
        # but does not use it in Compute_tracks_ellipsoid.
        stub = types.ModuleType("utils.utilities")
        stub.Timer = None
        sys.modules.setdefault("utils.utilities", stub)
        from tools.ellipsoid import Compute_tracks_ellipsoid  # noqa: E402
        ns = types.SimpleNamespace(**{f"{n}_{c}": u[:, i] for n, u in zip(names, (u1, u2, u3))
                                      for i, c in enumerate("xyz")})
        out = Compute_tracks_ellipsoid(_Defaults(ns))
        return np.stack([getattr(out, f"poca_{k}") for k in "ABCDEF"], 1), "pv-finder_v2 tools/ellipsoid.py"
    # Same arithmetic as tools/ellipsoid.py: sum over axes of u u^T / |u|^4.
    w = [u / (u * u).sum(1, keepdims=True) for u in (u1, u2, u3)]
    s = lambda i, j: sum(v[:, i] * v[:, j] for v in w)  # noqa: E731
    return np.stack([s(0, 0), s(1, 1), s(2, 2), s(0, 1), s(0, 2), s(1, 2)], 1), "reimplementation (repo not found)"


class _Defaults:
    """Attribute access for Compute_tracks_ellipsoid: the axes, None for the fields it only passes through."""
    def __init__(self, ns):
        self._ns = ns

    def __getattr__(self, k):
        return getattr(self._ns, k, None)


# --- 1. Allen vs the rules, per track -------------------------------------------
hdr, feats = read_dump("allen_fc_track_features.bin", 9)
_, states = read_dump("allen_fc_track_states.bin", 6)
_, beam = read_dump("allen_fc_beamline.bin", 5)
beam = beam.ravel()
x, y, z, tx, ty, c00 = states.T
bx = beam[0] + beam[3] * (z - beam[2])
by = beam[1] + beam[4] * (z - beam[2])
px, py, pz, e1, e2, e3, ratio = poca_and_axes(x - bx, y - by, z, tx - beam[3], ty - beam[4])
valid = np.hypot(px, py) < 1000.0
road = np.sqrt(np.where(c00 > 0, c00, 1.0))
abcdef, source = ellipsoid_ABCDEF(e1 * road[:, None], e2 * road[:, None], e3 * (road * ratio)[:, None])
abcdef[~(c00 > 0)] = 0.0   # no ellipsoid: zeros, which fail the FC track selection
ref = np.concatenate([np.stack([px, py, pz], 1), abcdef], 1)
ref[~valid] = 0.0


def ellipsoid_matrix(a):
    A, B, C, D, E, F = a.T
    return np.stack([np.stack([A, D, E], -1), np.stack([D, B, F], -1), np.stack([E, F, C], -1)], -2)


# POCA: absolute difference in mm. Ellipsoid: difference of the whole matrix
# relative to its size (its off-diagonal terms pass through zero, so a
# per-element relative test is not meaningful).
has_ellipsoid = valid & (c00 > 0)
poca_diff = np.abs(feats[:, :3] - ref[:, :3]).max(1)
m_ref = ellipsoid_matrix(ref[has_ellipsoid, 3:])
m_rel = np.linalg.norm(ellipsoid_matrix(feats[has_ellipsoid, 3:]) - m_ref, axis=(1, 2)) / np.linalg.norm(m_ref, axis=(1, 2))
no_ellipsoid_zero = bool((feats[~has_ellipsoid, 3:] == 0).all())
part1_ok = bool(np.isfinite(feats).all() and poca_diff.max() < args.poca_tol and m_rel.max() < args.rtol
                and no_ellipsoid_zero)
print(f"1. Allen features vs the rules recomputed in double precision ({len(feats)} tracks, "
      f"{int(hdr[1])} events; A..F from {source}):")
print(f"   POCA: max |diff| {poca_diff.max():.2e} mm (limit {args.poca_tol:g}); ellipsoid A..F: max relative "
      f"matrix difference {m_rel.max():.2e} (limit {args.rtol:g}); tracks without an ellipsoid (c00 <= 0): "
      f"{int((~(c00 > 0)).sum())}, A..F zero: {no_ellipsoid_zero}; all finite: {bool(np.isfinite(feats).all())}")
print(f"   {'PASS' if part1_ok else 'FAIL'}")
report["allen_vs_rules"] = {"tracks": int(len(feats)), "ellipsoid_source": source,
                            "poca_max_abs_diff_mm": float(poca_diff.max()),
                            "ellipsoid_max_rel_matrix_diff": float(m_rel.max()),
                            "tracks_without_ellipsoid": int((~(c00 > 0)).sum()),
                            "status": "PASS" if part1_ok else "FAIL"}


# --- 2. and 3. the rules vs the training sample -------------------------------
def read_jagged(f, key, n_events):
    schema = json.loads(bytes(f[f"{key}/schema.json"][:]).decode())
    counts_arg, content_arg = schema["schema"]["args"]
    counts = np.frombuffer(zlib.decompress(f[f"{key}/1"][:].tobytes()), np.int64)[:n_events]
    dtype = np.dtype(content_arg["args"][1]["dtype"])
    raw = f[f"{key}/3"][:].tobytes()
    if content_arg["args"][0].get("call") == ["zlib", "decompress"]:
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype)[:int(counts.sum())].astype(np.float64)


part2_ok = None
if os.path.isfile(args.training_h5):
    import h5py
    with h5py.File(args.training_h5, "r") as f:
        n = args.training_events
        t = {k: read_jagged(f, k, n) for k in (
            "recon_x", "recon_y", "recon_z", "recon_tx", "recon_ty", "poca_x", "poca_y", "poca_z",
            "minor_axis1_x", "minor_axis1_y", "minor_axis1_z", "minor_axis2_x", "minor_axis2_y", "minor_axis2_z",
            "major_axis_x", "major_axis_y", "major_axis_z")}
    u1 = np.stack([t["minor_axis1_x"], t["minor_axis1_y"], t["minor_axis1_z"]], 1)
    u2 = np.stack([t["minor_axis2_x"], t["minor_axis2_y"], t["minor_axis2_z"]], 1)
    u3 = np.stack([t["major_axis_x"], t["major_axis_y"], t["major_axis_z"]], 1)
    # The training sample's beam is on the z axis.
    qx, qy, qz, f1, f2, f3, fratio = poca_and_axes(t["recon_x"], t["recon_y"], t["recon_z"],
                                                  t["recon_tx"], t["recon_ty"])
    l1, l2, l3 = (np.linalg.norm(u, axis=1) for u in (u1, u2, u3))
    good = (l1 > 0) & (l2 > 0) & (l3 > 0)

    def angle(u, length, e):
        return np.degrees(np.arccos(np.clip(np.abs((u * e).sum(1)) / np.where(length > 0, length, 1.0), 0.0, 1.0)))

    d_poca = np.max(np.abs(np.stack([qx - t["poca_x"], qy - t["poca_y"], qz - t["poca_z"]], 1)), 1)

    def matrix(a):
        A, B, C, D, E, F = a.T
        return np.stack([np.stack([A, D, E], -1), np.stack([D, B, F], -1), np.stack([E, F, C], -1)], -2)
    # The ellipsoid the network sees (A..F), from the stored axes and from the
    # rules with the stored minor-axis length.
    m_train = matrix(ellipsoid_ABCDEF(u1[good], u2[good], u3[good])[0])
    m_rules = matrix(ellipsoid_ABCDEF((f1 * l1[:, None])[good], (f2 * l1[:, None])[good],
                                      (f3 * (l1 * fratio)[:, None])[good])[0])
    m_rel = np.linalg.norm(m_rules - m_train, axis=(1, 2)) / np.linalg.norm(m_train, axis=(1, 2))
    checks = {
        "poca_within_1um": float((d_poca < 1e-3)[good].mean()),
        "major_axis_within_0.01deg_of_track": float((angle(u3, l3, f3) < 0.01)[good].mean()),
        "major_minor_ratio_within_0.1pct": float((np.abs(l3 / np.where(l1 > 0, l1, 1) / fratio - 1) < 1e-3)[good].mean()),
        "ellipsoid_matrix_within_1pct": float((m_rel < 1e-2).mean()),
    }
    part2_ok = all(v > 0.99 for v in checks.values())
    print(f"2. the rules vs the training sample, per track ({int(good.sum())} tracks, {n} events; "
          f"fraction agreeing):")
    for k, v in checks.items():
        print(f"   {k}: {100 * v:.3f}%")
    a1 = angle(u1, l1, f1)[good]
    print(f"   (minor axis 1 vs beam x track, degrees, median / 99%: {np.median(a1):.4f} / {np.quantile(a1, 0.99):.3f};"
          f" the ntuples build it from the POCA offset, the same direction up to rounding, which grows as"
          f" the track passes closer to the beam; ellipsoid matrix difference median {np.median(m_rel):.1e},"
          f" 99% {np.quantile(m_rel, 0.99):.1e})")
    print(f"   {'PASS' if part2_ok else 'FAIL'} (each above 99%)")
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    tq, aq = np.quantile(l1[good], q), np.quantile(road[has_ellipsoid], q)
    print("3. minor-axis length (mm), training |minor axis| vs Allen sqrt(c00), quantiles 5/25/50/75/95%:")
    print("   training " + " ".join(f"{v:.4f}" for v in tq))
    print("   Allen    " + " ".join(f"{v:.4f}" for v in aq))
    report["rules_vs_training"] = {"tracks": int(good.sum()), "events": n, "checks": checks,
                                   "status": "PASS" if part2_ok else "FAIL"}
    report["minor_axis_quantiles"] = {"q": q, "training": tq.tolist(), "allen_sqrt_c00": aq.tolist()}
else:
    print(f"2./3. skipped: no training file at {args.training_h5}")

status = "PASS" if part1_ok and part2_ok is not False else "FAIL"
report["status"] = status
print(status)
if args.report:
    with open(args.report, "w") as fp:
        json.dump(report, fp, indent=2)
sys.exit(0 if status == "PASS" else 1)
