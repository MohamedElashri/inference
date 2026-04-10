#!/usr/bin/env python3
"""
validate_zseeds.py — Event-by-event comparison of classical and NN PV z-seeds.

Usage:
    python3 validate_zseeds.py \
        --nn-dump  /path/to/allen_nn_zpeaks.bin \
        --ref-dump /path/to/allen_classical_zseeds.bin

Binary formats:
    NN chain        : magic 0xAB2F, file allen_nn_zpeaks.bin
    Classical chain : magic 0xAB2E, file allen_classical_zseeds.bin

Shared layout:
    uint32  magic
    uint32  n_events
    for each event e:
        uint32  n_seeds
        float32[n_seeds]  z_seeds
"""

import argparse
import json
import struct
import numpy as np


parser = argparse.ArgumentParser(
    description="Compare classical and NN PV z-seeds event-by-event")
parser.add_argument(
    "--nn-dump", required=True, metavar="FILE",
    help="NN z-seed dump (magic 0xAB2F, e.g. allen_nn_zpeaks.bin)")
parser.add_argument(
    "--ref-dump", required=True, metavar="FILE",
    help="Classical z-seed dump (magic 0xAB2E, e.g. allen_classical_zseeds.bin)")
parser.add_argument(
    "--match-dz", type=float, default=2.0,
    help="Maximum |Δz| in mm to count two seeds as matched (default: 2.0)")
parser.add_argument(
    "--report", default="", metavar="FILE",
    help="Write machine-readable JSON summary to this path")
args = parser.parse_args()


SEP = "=" * 72
SEP2 = "-" * 72
VALID_MAGICS = {0xAB2E, 0xAB2F}


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subsection(title):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


def stat_block(name, arr, unit="mm"):
    a = np.asarray(arr, dtype=np.float64).flatten()
    if a.size == 0:
        print(f"  {name}: (empty)")
        return None
    pcts = np.percentile(a, [0, 25, 50, 75, 90, 99, 100])
    print(f"  {name}{(' (' + unit + ')') if unit else ''}:")
    print(f"    n={a.size}  mean={a.mean():.4e}  std={a.std():.4e}  "
          f"min={a.min():.4e}  max={a.max():.4e}")
    print(f"    P0={pcts[0]:.4e}  P25={pcts[1]:.4e}  P50={pcts[2]:.4e}  "
          f"P75={pcts[3]:.4e}  P90={pcts[4]:.4e}  P99={pcts[5]:.4e}  "
          f"Pmax={pcts[6]:.4e}")
    return pcts


def read_zseeds(path):
    with open(path, "rb") as fh:
        raw = fh.read()

    magic, n_events = struct.unpack_from("<II", raw, 0)
    if magic not in VALID_MAGICS:
        raise ValueError(
            f"{path}: unexpected magic {magic:#010x}; "
            f"expected one of {[f'{m:#010x}' for m in sorted(VALID_MAGICS)]}")

    offset = 8
    events = []
    for _ in range(n_events):
        n_seeds, = struct.unpack_from("<I", raw, offset)
        offset += 4
        if n_seeds:
            fmt = f"<{n_seeds}f"
            seeds = list(struct.unpack_from(fmt, raw, offset))
            offset += 4 * n_seeds
        else:
            seeds = []
        events.append(seeds)

    if offset != len(raw):
        print(f"  WARNING: {path} has {len(raw) - offset} trailing bytes")

    return n_events, events, magic


def match_seeds(nn_seeds, ref_seeds, max_dz_mm):
    if not nn_seeds or not ref_seeds:
        return [], list(range(len(nn_seeds))), list(range(len(ref_seeds)))

    nn_order = sorted(range(len(nn_seeds)), key=lambda i: nn_seeds[i])
    ref_order = sorted(range(len(ref_seeds)), key=lambda i: ref_seeds[i])

    used_ref = set()
    matched = []

    for ni in nn_order:
        best_rj = None
        best_dz = float("inf")
        nn_z = nn_seeds[ni]
        for rj in ref_order:
            if rj in used_ref:
                continue
            dz = abs(nn_z - ref_seeds[rj])
            if dz < best_dz:
                best_dz = dz
                best_rj = rj
            elif dz > best_dz + max_dz_mm:
                break
        if best_rj is not None and best_dz <= max_dz_mm:
            matched.append((ni, best_rj))
            used_ref.add(best_rj)

    matched_nn = {m[0] for m in matched}
    matched_ref = {m[1] for m in matched}
    unmatched_nn = [i for i in range(len(nn_seeds)) if i not in matched_nn]
    unmatched_ref = [i for i in range(len(ref_seeds)) if i not in matched_ref]
    return matched, unmatched_nn, unmatched_ref


section("Loading Z-Seed Dumps")
print(f"  NN dump : {args.nn_dump}")
nn_n_events, nn_events, nn_magic = read_zseeds(args.nn_dump)
print(f"    magic={nn_magic:#010x}  n_events={nn_n_events}  total_seeds={sum(len(e) for e in nn_events)}")

print(f"  Ref dump: {args.ref_dump}")
ref_n_events, ref_events, ref_magic = read_zseeds(args.ref_dump)
print(f"    magic={ref_magic:#010x}  n_events={ref_n_events}  total_seeds={sum(len(e) for e in ref_events)}")

n_cmp_events = min(nn_n_events, ref_n_events)
if nn_n_events != ref_n_events:
    print(f"  WARNING: event count mismatch: NN={nn_n_events} vs Ref={ref_n_events}; "
          f"using {n_cmp_events} events")


section("Per-Event Seed Multiplicity")
nn_counts = np.array([len(e) for e in nn_events[:n_cmp_events]], dtype=np.int32)
ref_counts = np.array([len(e) for e in ref_events[:n_cmp_events]], dtype=np.int32)
count_diff = nn_counts - ref_counts
mean_nn_counts = float(nn_counts.mean()) if n_cmp_events else 0.0
mean_ref_counts = float(ref_counts.mean()) if n_cmp_events else 0.0
count_agreement = float((nn_counts == ref_counts).mean()) if n_cmp_events else 0.0

print(f"  Events compared            : {n_cmp_events}")
print(f"  NN mean seeds/event        : {mean_nn_counts:.4f}")
print(f"  Classical mean seeds/event : {mean_ref_counts:.4f}")
print(f"  Exact-count agreement      : {(nn_counts == ref_counts).sum()} / {n_cmp_events} "
      f"({100.0 * count_agreement:.2f}%)")
stat_block("NN minus classical seed count", count_diff, unit="")


section("Greedy Z Matching")
matched_dz = []
matched_abs_dz = []
unmatched_nn_total = 0
unmatched_ref_total = 0
events_with_mismatch = 0

for evt in range(n_cmp_events):
    matched, unmatched_nn, unmatched_ref = match_seeds(
        nn_events[evt], ref_events[evt], args.match_dz)
    if unmatched_nn or unmatched_ref or len(nn_events[evt]) != len(ref_events[evt]):
        events_with_mismatch += 1
    unmatched_nn_total += len(unmatched_nn)
    unmatched_ref_total += len(unmatched_ref)
    for nn_i, ref_i in matched:
        dz = nn_events[evt][nn_i] - ref_events[evt][ref_i]
        matched_dz.append(dz)
        matched_abs_dz.append(abs(dz))

nn_total = int(nn_counts.sum())
ref_total = int(ref_counts.sum())
matched_total = len(matched_dz)

print(f"  Match window |Δz| <= {args.match_dz:.3f} mm")
print(f"  Matched seed pairs         : {matched_total}")
print(f"  Unmatched NN seeds         : {unmatched_nn_total} / {nn_total} "
      f"({100.0 * unmatched_nn_total / nn_total:.2f}%)" if nn_total else "  Unmatched NN seeds         : 0 / 0")
print(f"  Unmatched classical seeds  : {unmatched_ref_total} / {ref_total} "
      f"({100.0 * unmatched_ref_total / ref_total:.2f}%)" if ref_total else "  Unmatched classical seeds  : 0 / 0")
print(f"  Events with any mismatch   : {events_with_mismatch} / {n_cmp_events} "
      f"({100.0 * events_with_mismatch / n_cmp_events:.2f}%)" if n_cmp_events else "  Events with any mismatch   : 0 / 0")

subsection("Matched Seed Δz")
stat_block("Signed Δz (NN - classical)", matched_dz)
stat_block("Absolute Δz", matched_abs_dz)


section("Largest Multiplicity Disagreements")
if n_cmp_events == 0:
    print("  No comparable events")
else:
    worst = np.argsort(np.abs(count_diff))[::-1][:10]
    print("  evt    nn_seeds   ref_seeds   diff")
    for evt in worst:
        print(f"  {evt:4d}   {nn_counts[evt]:8d}   {ref_counts[evt]:9d}   {count_diff[evt]:+4d}")


summary = {
    "nn_dump": args.nn_dump,
    "ref_dump": args.ref_dump,
    "match_dz_mm": args.match_dz,
    "events_compared": int(n_cmp_events),
    "nn_total_seeds": nn_total,
    "ref_total_seeds": ref_total,
    "matched_seed_pairs": matched_total,
    "unmatched_nn_seeds": unmatched_nn_total,
    "unmatched_ref_seeds": unmatched_ref_total,
    "events_with_mismatch": events_with_mismatch,
    "count_agreement_fraction": count_agreement,
    "mean_nn_seeds_per_event": mean_nn_counts,
    "mean_ref_seeds_per_event": mean_ref_counts,
    "mean_signed_dz_mm": float(np.mean(matched_dz)) if matched_dz else None,
    "mean_abs_dz_mm": float(np.mean(matched_abs_dz)) if matched_abs_dz else None,
    "max_abs_dz_mm": float(np.max(matched_abs_dz)) if matched_abs_dz else None,
}

if args.report:
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"\nWrote report to {args.report}")
