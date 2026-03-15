#!/usr/bin/env python3
"""
validate_vertices.py — Vertex-level validation of the Allen NN PV chain.

Usage:
    # Standalone NN sanity validation (physics checks on NN output only):
    python3 validate_vertices.py --nn-dump /path/to/allen_nn_final_vertices.bin

    # Head-to-head comparison with classical chain (requires both dumps):
    python3 validate_vertices.py \
        --nn-dump  /path/to/allen_nn_final_vertices.bin  \
        --ref-dump /path/to/allen_classical_vertices.bin \
        [--plot] [--report report.json]

Dump files are produced by:
    NN chain  : PVFinderNNCleanup   (magic 0xAB20, file allen_nn_final_vertices.bin)
                PVFinderVertexFitter (magic 0xAB1F, file allen_nn_vertices.bin — pre-cleanup)
    Any chain : PVFinderDumpVertices (magic 0xAB21, configurable filename)

Binary format (all three share the same per-record layout):
    uint32  magic
    uint32  n_events
    for each event e:
        uint32  n_vertices
        for each vertex v:
            float32  x, y, z
            float32  cov00, cov10, cov11, cov20, cov21, cov22
            float32  chi2
            int32    ndof
            float32  nTracks
"""

import argparse
import json
import math
import os
import struct
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Vertex-level validation of the Allen NN PV chain")
parser.add_argument(
    "--nn-dump", required=True, metavar="FILE",
    help="NN chain vertex dump (magic 0xAB20 from NNCleanup, or 0xAB1F from VertexFitter)")
parser.add_argument(
    "--ref-dump", default="", metavar="FILE",
    help="Reference (classical) vertex dump for head-to-head comparison "
         "(magic 0xAB21 from PVFinderDumpVertices). Optional.")
parser.add_argument(
    "--match-dz", type=float, default=5.0,
    help="Maximum |Δz| (mm) to call two vertices matched (default: 5.0)")
parser.add_argument(
    "--plot", action="store_true",
    help="Save comparison plots to <nn-dump-dir>/plots/")
parser.add_argument(
    "--report", default="", metavar="FILE",
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


def stat_block(name, arr, unit=""):
    """Print compact statistics for a flat array."""
    a = np.asarray(arr, dtype=np.float64).flatten()
    if a.size == 0:
        print(f"  {name}: (empty)")
        return
    pcts = np.percentile(a, [0, 25, 50, 75, 90, 99, 100])
    print(f"  {name}{(' (' + unit + ')') if unit else ''}:")
    print(f"    n={a.size}  mean={a.mean():.4e}  std={a.std():.4e}  "
          f"min={a.min():.4e}  max={a.max():.4e}")
    print(f"    P0={pcts[0]:.4e}  P25={pcts[1]:.4e}  P50={pcts[2]:.4e}  "
          f"P75={pcts[3]:.4e}  P90={pcts[4]:.4e}  P99={pcts[5]:.4e}  "
          f"Pmax={pcts[6]:.4e}")
    return pcts


# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------
# Per-vertex packed layout: 10 floats, 1 int32, 1 float  (48 bytes)
_VERTEX_FMT  = '<10fif'
_VERTEX_SIZE = struct.calcsize(_VERTEX_FMT)   # 48 bytes
_VALID_MAGICS = {0xAB1F, 0xAB20, 0xAB21}

# Named indices into the unpacked tuple
_X   , _Y   , _Z    = 0, 1, 2
_C00 , _C10 , _C11  = 3, 4, 5
_C20 , _C21 , _C22  = 6, 7, 8
_CHI2, _NDOF, _NTRACKS = 9, 10, 11


def read_vertices(path):
    """
    Read a vertex dump file.

    Returns
    -------
    n_events : int
    events   : list[list[tuple]]
        events[e] = list of vertex tuples (x, y, z, cov00..cov22, chi2, ndof, nTracks)
    magic    : int
    """
    with open(path, 'rb') as fh:
        raw = fh.read()

    magic, n_events = struct.unpack_from('<II', raw, 0)
    if magic not in _VALID_MAGICS:
        raise ValueError(
            f"{path}: unexpected magic {magic:#010x}; "
            f"expected one of {[f'{m:#010x}' for m in _VALID_MAGICS]}")

    offset = 8
    events = []
    for _ in range(n_events):
        nv, = struct.unpack_from('<I', raw, offset)
        offset += 4
        verts = []
        for _ in range(nv):
            verts.append(struct.unpack_from(_VERTEX_FMT, raw, offset))
            offset += _VERTEX_SIZE
        events.append(verts)

    if offset != len(raw):
        print(f"  WARNING: {path} has {len(raw) - offset} trailing bytes")

    return n_events, events, magic


# ---------------------------------------------------------------------------
# Matching helper
# ---------------------------------------------------------------------------
def match_vertices(nn_verts, ref_verts, max_dz_mm):
    """
    Greedy nearest-z matching between NN and reference vertex lists.

    Returns
    -------
    matched     : list[(nn_idx, ref_idx)]
    unmatched_nn  : list[nn_idx]
    unmatched_ref : list[ref_idx]
    """
    if not nn_verts or not ref_verts:
        return [], list(range(len(nn_verts))), list(range(len(ref_verts)))

    # Sort indices by z for cache-friendly access
    nn_order  = sorted(range(len(nn_verts)),  key=lambda i: nn_verts[i][_Z])
    ref_order = sorted(range(len(ref_verts)), key=lambda j: ref_verts[j][_Z])

    used_ref = set()
    matched  = []

    for ni in nn_order:
        best_rj  = None
        best_dz  = float('inf')
        nn_z     = nn_verts[ni][_Z]
        for rj in ref_order:
            if rj in used_ref:
                continue
            dz = abs(nn_z - ref_verts[rj][_Z])
            if dz < best_dz:
                best_dz = dz
                best_rj = rj
            elif dz > best_dz + max_dz_mm:
                # ref is sorted by z — once we pass the window, no closer match
                break
        if best_rj is not None and best_dz <= max_dz_mm:
            matched.append((ni, best_rj))
            used_ref.add(best_rj)

    matched_nn_set  = {m[0] for m in matched}
    matched_ref_set = {m[1] for m in matched}
    unmatched_nn  = [i for i in range(len(nn_verts))  if i not in matched_nn_set]
    unmatched_ref = [j for j in range(len(ref_verts)) if j not in matched_ref_set]

    return matched, unmatched_nn, unmatched_ref


# ---------------------------------------------------------------------------
# Section 1: Load dumps
# ---------------------------------------------------------------------------
section("Loading Vertex Dumps")

print(f"  NN dump : {args.nn_dump}")
nn_n_events, nn_events, nn_magic = read_vertices(args.nn_dump)
print(f"    magic={nn_magic:#010x}  n_events={nn_n_events}  "
      f"total_vertices={sum(len(e) for e in nn_events)}")

have_ref = bool(args.ref_dump)
if have_ref:
    print(f"  Ref dump: {args.ref_dump}")
    ref_n_events, ref_events, ref_magic = read_vertices(args.ref_dump)
    print(f"    magic={ref_magic:#010x}  n_events={ref_n_events}  "
          f"total_vertices={sum(len(e) for e in ref_events)}")
    if nn_n_events != ref_n_events:
        print(f"  WARNING: event count mismatch: NN={nn_n_events} vs Ref={ref_n_events}; "
              f"using min({nn_n_events}, {ref_n_events}) events for comparison")
    n_cmp_events = min(nn_n_events, ref_n_events)
else:
    print("  (no reference dump — running standalone NN validation only)")

# ---------------------------------------------------------------------------
# Flatten helpers
# ---------------------------------------------------------------------------
def flat_field(events, idx, dtype=np.float32):
    return np.array([v[idx] for e in events for v in e], dtype=dtype)

def per_event_counts(events):
    return np.array([len(e) for e in events], dtype=np.int32)

nn_nv_per_event = per_event_counts(nn_events)
nn_total = int(nn_nv_per_event.sum())

# ---------------------------------------------------------------------------
# Section 2: Standalone NN physics validation
# ---------------------------------------------------------------------------
section("Standalone NN Physics Validation")

# 2a. Vertex multiplicity
subsection("Vertex multiplicity per event")
print(f"  Total events  : {nn_n_events}")
print(f"  Total vertices: {nn_total}")
print(f"  Events with ≥1 PV: {(nn_nv_per_event >= 1).sum()} "
      f"({100*(nn_nv_per_event >= 1).mean():.1f}%)")
print(f"  Events with  0 PV: {(nn_nv_per_event == 0).sum()} "
      f"({100*(nn_nv_per_event == 0).mean():.1f}%)")
stat_block("PVs per event", nn_nv_per_event)

# 2b. Position ranges
subsection("Vertex positions")
if nn_total > 0:
    nn_z = flat_field(nn_events, _Z)
    nn_x = flat_field(nn_events, _X)
    nn_y = flat_field(nn_events, _Y)
    stat_block("z", nn_z, "mm")
    stat_block("x", nn_x, "mm")
    stat_block("y", nn_y, "mm")

    z_in_range = np.sum((nn_z >= -100.) & (nn_z <= 300.))
    print(f"\n  z ∈ [-100, +300] mm  : {z_in_range}/{nn_total} "
          f"({100*z_in_range/nn_total:.2f}%)")
    z_out_range = nn_total - z_in_range
    if z_out_range > 0:
        print(f"  WARNING: {z_out_range} vertices outside NN z-coverage!")

# 2c. Covariance diagonal (must be positive)
subsection("Covariance diagonal (positivity check)")
if nn_total > 0:
    nn_cov00 = flat_field(nn_events, _C00)
    nn_cov11 = flat_field(nn_events, _C11)
    nn_cov22 = flat_field(nn_events, _C22)
    cov_bad = ((nn_cov00 <= 0) | (nn_cov11 <= 0) | (nn_cov22 <= 0)).sum()
    print(f"  Vertices with any diagonal cov ≤ 0: {cov_bad}/{nn_total} "
          f"({'FAIL' if cov_bad else 'PASS'})")
    stat_block("σz = √cov22", np.sqrt(np.maximum(nn_cov22, 0)), "mm")
    stat_block("σx = √cov00", np.sqrt(np.maximum(nn_cov00, 0)), "mm")
    stat_block("σy = √cov11", np.sqrt(np.maximum(nn_cov11, 0)), "mm")

# 2d. Track weights (nTracks)
subsection("nTracks (weighted track count)")
if nn_total > 0:
    nn_ntracks = flat_field(nn_events, _NTRACKS)
    stat_block("nTracks", nn_ntracks)
    below_min = (nn_ntracks < 4.).sum()
    if below_min:
        print(f"  WARNING: {below_min} vertices have nTracks < 4 "
              f"(below pp_minNumTracksPerVertex)")

# 2e. chi2/ndof
subsection("Fit quality (chi2 / ndof)")
if nn_total > 0:
    nn_chi2  = flat_field(nn_events, _CHI2)
    nn_ndof  = flat_field(nn_events, _NDOF, dtype=np.int32)
    ndof_pos = nn_ndof > 0
    if ndof_pos.sum() > 0:
        chi2_per_ndof = nn_chi2[ndof_pos] / nn_ndof[ndof_pos].astype(np.float32)
        stat_block("chi2/ndof", chi2_per_ndof)
    else:
        print("  WARNING: all ndof ≤ 0 — chi2/ndof undefined")

# 2f. NaN / Inf check
subsection("NaN / Inf check")
if nn_total > 0:
    all_fields = np.array([
        [v[_X], v[_Y], v[_Z],
         v[_C00], v[_C11], v[_C22],
         v[_CHI2], float(v[_NDOF]), v[_NTRACKS]]
        for e in nn_events for v in e
    ], dtype=np.float64)
    n_nan = int(np.isnan(all_fields).any(axis=1).sum())
    n_inf = int(np.isinf(all_fields).any(axis=1).sum())
    print(f"  Vertices with NaN in any field: {n_nan}  "
          f"({'FAIL' if n_nan else 'PASS'})")
    print(f"  Vertices with Inf in any field: {n_inf}  "
          f"({'FAIL' if n_inf else 'PASS'})")

# ---------------------------------------------------------------------------
# Section 3: Head-to-head comparison vs reference
# ---------------------------------------------------------------------------
if have_ref:
    section("Head-to-Head Comparison: NN vs Reference")

    ref_nv_per_event = per_event_counts(ref_events[:n_cmp_events])
    ref_total = int(ref_nv_per_event.sum())
    print(f"  Comparing {n_cmp_events} events: "
          f"NN={nn_nv_per_event[:n_cmp_events].sum()} total PVs, "
          f"Ref={ref_total} total PVs")

    # --- Per-event matching ---
    all_dz         = []
    all_dx         = []
    all_dy         = []
    all_pull_z     = []  # Δz / sqrt(cov22_nn + cov22_ref)
    all_nn_ntracks_matched = []
    n_matched_total = 0
    n_unmatched_nn  = 0
    n_unmatched_ref = 0

    for e in range(n_cmp_events):
        nn_v  = nn_events[e]
        ref_v = ref_events[e]
        matched, unm_nn, unm_ref = match_vertices(nn_v, ref_v, args.match_dz)

        n_matched_total += len(matched)
        n_unmatched_nn  += len(unm_nn)
        n_unmatched_ref += len(unm_ref)

        for (ni, ri) in matched:
            dz = nn_v[ni][_Z] - ref_v[ri][_Z]
            dx = nn_v[ni][_X] - ref_v[ri][_X]
            dy = nn_v[ni][_Y] - ref_v[ri][_Y]
            all_dz.append(dz)
            all_dx.append(dx)
            all_dy.append(dy)
            all_nn_ntracks_matched.append(nn_v[ni][_NTRACKS])

            # Pull: Δz / σ(Δz)  where σ²(Δz) = cov22_nn + cov22_ref
            cov22_nn  = nn_v[ni][_C22]
            cov22_ref = ref_v[ri][_C22]
            sig2 = cov22_nn + cov22_ref
            if sig2 > 0:
                all_pull_z.append(dz / math.sqrt(sig2))

    all_dz     = np.array(all_dz)
    all_dx     = np.array(all_dx)
    all_dy     = np.array(all_dy)
    all_pull_z = np.array(all_pull_z)

    # --- Matching statistics ---
    subsection("Matching statistics")
    nn_total_cmp = int(nn_nv_per_event[:n_cmp_events].sum())
    efficiency = n_matched_total / ref_total if ref_total > 0 else float('nan')
    fake_rate  = n_unmatched_nn  / nn_total_cmp if nn_total_cmp > 0 else float('nan')

    print(f"  Ref PVs              : {ref_total}")
    print(f"  NN  PVs              : {nn_total_cmp}")
    print(f"  Matched pairs        : {n_matched_total}")
    print(f"  Unmatched NN  (fake) : {n_unmatched_nn}")
    print(f"  Unmatched Ref (inefficiency): {n_unmatched_ref}")
    print(f"  Efficiency (matched/Ref): {efficiency:.4f}  ({100*efficiency:.2f}%)")
    print(f"  Fake rate  (unm_NN/NN)  : {fake_rate:.4f}  ({100*fake_rate:.2f}%)")

    # --- Position residuals ---
    subsection("Position residuals (matched pairs)")
    if all_dz.size > 0:
        stat_block("Δz (NN - Ref)", all_dz, "mm")
        stat_block("Δx (NN - Ref)", all_dx, "mm")
        stat_block("Δy (NN - Ref)", all_dy, "mm")
        print(f"\n  |Δz| < 0.10 mm: {(np.abs(all_dz) < 0.10).mean()*100:.1f}%")
        print(f"  |Δz| < 0.50 mm: {(np.abs(all_dz) < 0.50).mean()*100:.1f}%")
        print(f"  |Δz| < 1.00 mm: {(np.abs(all_dz) < 1.00).mean()*100:.1f}%")

    # --- Pull distribution ---
    subsection("z-pull distribution  Δz / √(cov22_NN + cov22_Ref)")
    if all_pull_z.size > 0:
        pull_mean = float(all_pull_z.mean())
        pull_std  = float(all_pull_z.std())
        print(f"  n pairs with valid σ: {all_pull_z.size}")
        print(f"  Pull mean : {pull_mean:.4f}   (target: ~0)")
        print(f"  Pull σ    : {pull_std:.4f}   (target: ~1.0  if covariance is correct)")
        pull_ok = 0.7 < pull_std < 1.5
        print(f"  Pull σ within [0.7, 1.5]: {'PASS' if pull_ok else 'FAIL'}")

    # --- Multiplicity scatter ---
    subsection("PV multiplicity per event (NN vs Ref)")
    nn_mul  = nn_nv_per_event[:n_cmp_events].astype(float)
    ref_mul = ref_nv_per_event.astype(float)
    diff_mul = nn_mul - ref_mul
    print(f"  Mean NN  PVs/event: {nn_mul.mean():.3f}")
    print(f"  Mean Ref PVs/event: {ref_mul.mean():.3f}")
    print(f"  Mean (NN-Ref)/event: {diff_mul.mean():+.3f}  "
          f"std={diff_mul.std():.3f}")
    for d in [-2, -1, 0, 1, 2]:
        frac = (diff_mul == d).mean()
        print(f"    NN - Ref = {d:+d}: {100*frac:.1f}%")

# ---------------------------------------------------------------------------
# Section 4: Pass / Fail Verdicts
# ---------------------------------------------------------------------------
section("Pass / Fail Verdicts")

verdicts = {}

# --- Standalone checks ---
subsection("Standalone NN checks")
if nn_total > 0:
    zero_pv_rate = float((nn_nv_per_event == 0).mean())
    pos_cov_ok   = (cov_bad == 0)
    z_range_ok   = (z_out_range == 0)
    nan_ok       = (n_nan == 0)
    inf_ok       = (n_inf == 0)

    print(f"  Events with 0 PVs (< 10% target): {100*zero_pv_rate:.1f}%  "
          f"{'PASS' if zero_pv_rate < 0.10 else 'WARN'}")
    print(f"  Covariance diagonal positive:  {'PASS' if pos_cov_ok else 'FAIL'}")
    print(f"  All z in [-100, +300] mm:      {'PASS' if z_range_ok else 'WARN'}")
    print(f"  No NaN in any field:           {'PASS' if nan_ok  else 'FAIL'}")
    print(f"  No Inf in any field:           {'PASS' if inf_ok  else 'FAIL'}")

    verdicts.update({
        "zero_pv_rate"         : zero_pv_rate,
        "pos_cov_diag"         : bool(pos_cov_ok),
        "z_in_range"           : bool(z_range_ok),
        "no_nan"               : bool(nan_ok),
        "no_inf"               : bool(inf_ok),
    })
else:
    print("  WARNING: no vertices in NN dump — all standalone checks skipped")
    verdicts["error"] = "no_vertices"

# --- Comparison checks (if reference available) ---
if have_ref and nn_total > 0:
    subsection("Head-to-head checks (NN vs classical)")

    EFFICIENCY_TARGET = 0.95
    FAKE_RATE_TARGET  = 0.05
    MEDIAN_DZ_TARGET  = 0.10   # mm
    PULL_STD_LO       = 0.7
    PULL_STD_HI       = 1.5

    eff_ok     = efficiency >= EFFICIENCY_TARGET
    fake_ok    = fake_rate  <= FAKE_RATE_TARGET
    med_dz_ok  = all_dz.size > 0 and float(np.median(np.abs(all_dz))) < MEDIAN_DZ_TARGET
    pull_ok_v  = all_pull_z.size > 0 and PULL_STD_LO < pull_std < PULL_STD_HI

    print(f"  Efficiency ≥ {EFFICIENCY_TARGET:.0%}:        "
          f"{100*efficiency:.2f}%  {'PASS' if eff_ok else 'FAIL'}")
    print(f"  Fake rate  ≤ {FAKE_RATE_TARGET:.0%}:         "
          f"{100*fake_rate:.2f}%  {'PASS' if fake_ok else 'FAIL'}")
    print(f"  Median |Δz| < {MEDIAN_DZ_TARGET:.2f} mm:    "
          f"{np.median(np.abs(all_dz)) if all_dz.size else float('nan'):.4f} mm  "
          f"{'PASS' if med_dz_ok else 'FAIL'}")
    print(f"  Pull σ ∈ [{PULL_STD_LO}, {PULL_STD_HI}]:     "
          f"{pull_std if all_pull_z.size else float('nan'):.4f}  "
          f"{'PASS' if pull_ok_v else 'FAIL'}")

    all_cmp_ok = eff_ok and fake_ok and med_dz_ok and pull_ok_v
    print(f"\n  >>> Overall comparison: {'PASS' if all_cmp_ok else 'FAIL'}")

    verdicts.update({
        "efficiency"           : float(efficiency),
        "fake_rate"            : float(fake_rate),
        "median_abs_dz_mm"     : float(np.median(np.abs(all_dz))) if all_dz.size else None,
        "pull_std"             : float(pull_std) if all_pull_z.size else None,
        "pass_efficiency"      : bool(eff_ok),
        "pass_fake_rate"       : bool(fake_ok),
        "pass_median_dz"       : bool(med_dz_ok),
        "pass_pull_std"        : bool(pull_ok_v),
        "overall_comparison"   : bool(all_cmp_ok),
    })

# ---------------------------------------------------------------------------
# Section 5: JSON report
# ---------------------------------------------------------------------------
report = {
    "nn_dump"     : args.nn_dump,
    "ref_dump"    : args.ref_dump,
    "n_events"    : int(nn_n_events),
    "nn_total_vertices": int(nn_total),
    "nn_stats": {
        "mean_pvs_per_event": float(nn_nv_per_event.mean()),
        "zero_pv_fraction"  : float((nn_nv_per_event == 0).mean()),
    } if nn_total > 0 else {},
    "verdicts"    : verdicts,
}

if have_ref and nn_total > 0:
    report["comparison"] = {
        "n_events_compared": int(n_cmp_events),
        "n_matched"        : int(n_matched_total),
        "n_unmatched_nn"   : int(n_unmatched_nn),
        "n_unmatched_ref"  : int(n_unmatched_ref),
        "efficiency"       : float(efficiency),
        "fake_rate"        : float(fake_rate),
        "dz_stats": {
            "mean_mm"  : float(all_dz.mean())    if all_dz.size else None,
            "std_mm"   : float(all_dz.std())     if all_dz.size else None,
            "median_mm": float(np.median(all_dz))if all_dz.size else None,
            "median_abs_mm": float(np.median(np.abs(all_dz))) if all_dz.size else None,
        },
        "pull_z_stats": {
            "mean" : float(all_pull_z.mean()) if all_pull_z.size else None,
            "std"  : float(all_pull_z.std())  if all_pull_z.size else None,
        },
    }

if args.report:
    with open(args.report, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\n  JSON report written to: {args.report}")

# ---------------------------------------------------------------------------
# Section 6: Optional plots
# ---------------------------------------------------------------------------
if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(os.path.dirname(args.nn_dump) or ".", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    section(f"Generating Plots → {plot_dir}")

    # ---- 1. PV multiplicity histogram ----
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.5, max(nn_nv_per_event.max() + 2,
                                (ref_nv_per_event.max() + 2 if have_ref else 0)), 1)
    ax.hist(nn_nv_per_event, bins=bins, alpha=0.7, label="NN chain",
            color="steelblue", edgecolor="white")
    if have_ref:
        ax.hist(ref_nv_per_event, bins=bins, alpha=0.5, label="Classical chain",
                color="crimson", edgecolor="white")
    ax.set_xlabel("N PVs per event")
    ax.set_ylabel("Events")
    ax.set_title("PV multiplicity per event")
    ax.legend()
    p = os.path.join(plot_dir, "01_multiplicity.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    # ---- 2. z distribution ----
    if nn_total > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        bins_z = np.linspace(-110., 310., 200)
        ax.hist(nn_z, bins=bins_z, alpha=0.7, label="NN chain",
                color="steelblue", edgecolor="none")
        if have_ref:
            ref_z_all = flat_field(ref_events[:n_cmp_events], _Z)
            ax.hist(ref_z_all, bins=bins_z, alpha=0.5, label="Classical chain",
                    color="crimson", edgecolor="none")
        ax.axvline(-100., color="black", ls="--", lw=1.2, label="NN z-range")
        ax.axvline( 300., color="black", ls="--", lw=1.2)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("Vertices")
        ax.set_title("Reconstructed PV z distribution")
        ax.legend()
        p = os.path.join(plot_dir, "02_z_distribution.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")

    # ---- 3. σz distribution ----
    if nn_total > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        sigma_z = np.sqrt(np.maximum(nn_cov22, 0))
        ax.hist(sigma_z, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_xlabel("σz = √cov22 (mm)")
        ax.set_ylabel("Vertices")
        ax.set_title("NN PV z uncertainty distribution")
        p = os.path.join(plot_dir, "03_sigma_z.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")

    if have_ref and all_dz.size > 0:
        # ---- 4. Δz histogram ----
        fig, ax = plt.subplots(figsize=(8, 5))
        dz_lim = min(5.0, float(np.abs(all_dz).max()) * 1.1)
        ax.hist(all_dz, bins=100, range=(-dz_lim, dz_lim),
                color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(0., color="red", ls="--", lw=1.2)
        ax.set_xlabel("Δz = z_NN − z_Ref (mm)")
        ax.set_ylabel("Matched pairs")
        ax.set_title(f"z residual  (median|Δz|={np.median(np.abs(all_dz)):.4f} mm)")
        p = os.path.join(plot_dir, "04_dz_residual.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")

        # ---- 5. Δx / Δy histograms ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, arr, lbl in zip(axes, [all_dx, all_dy], ["Δx", "Δy"]):
            lim = min(2., float(np.abs(arr).max()) * 1.1)
            ax.hist(arr, bins=100, range=(-lim, lim),
                    color="steelblue", edgecolor="none", alpha=0.8)
            ax.axvline(0., color="red", ls="--", lw=1.2)
            ax.set_xlabel(f"{lbl} = {lbl[1:]}_NN − {lbl[1:]}_Ref (mm)")
            ax.set_ylabel("Matched pairs")
            ax.set_title(f"{lbl} residual  (σ={arr.std():.4f} mm)")
        fig.suptitle("Transverse residuals (NN vs Classical)", fontweight="bold")
        plt.tight_layout()
        p = os.path.join(plot_dir, "05_transverse_residuals.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")

        # ---- 6. z-pull histogram ----
        if all_pull_z.size > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            pull_lim = min(10., float(np.abs(all_pull_z).max()) * 1.1)
            ax.hist(all_pull_z, bins=100, range=(-pull_lim, pull_lim),
                    color="steelblue", edgecolor="none", alpha=0.8,
                    density=True)
            z_pts = np.linspace(-pull_lim, pull_lim, 300)
            ax.plot(z_pts, np.exp(-0.5 * z_pts**2) / np.sqrt(2 * np.pi),
                    "r--", lw=1.5, label="N(0,1)")
            ax.set_xlabel("Δz / √(cov22_NN + cov22_Ref)")
            ax.set_ylabel("Density")
            ax.set_title(f"z pull  (mean={all_pull_z.mean():.3f}  σ={all_pull_z.std():.3f})")
            ax.legend()
            p = os.path.join(plot_dir, "06_zpull.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {p}")

        # ---- 7. Multiplicity scatter NN vs Ref ----
        fig, ax = plt.subplots(figsize=(7, 7))
        max_mul = max(nn_nv_per_event[:n_cmp_events].max(),
                      ref_nv_per_event.max(), 1)
        bins2d  = np.arange(-0.5, max_mul + 1.5, 1)
        h, xedges, yedges = np.histogram2d(
            ref_nv_per_event, nn_nv_per_event[:n_cmp_events], bins=bins2d)
        im = ax.pcolormesh(xedges, yedges, h.T,
                           cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax, label="Events")
        lim = max_mul + 0.5
        ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="NN = Ref")
        ax.set_xlabel("N PVs (Classical)")
        ax.set_ylabel("N PVs (NN)")
        ax.set_title("PV multiplicity: NN vs Classical")
        ax.legend()
        p = os.path.join(plot_dir, "07_multiplicity_scatter.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")

    print(f"\n  All plots saved to: {plot_dir}")

print(f"\n{'='*72}")
print("  Done.")
print(f"{'='*72}\n")
