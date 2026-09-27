"""Check Allen's pvfinder_peak against pv-finder's peak finder on the same KDE.

Reads a dump directory with the UNet's KDE (allen_kde_output.bin, from
pvfinder_unet.dump_validation) and the seeds pvfinder_peak found in it
(allen_zpeaks.bin, from pvfinder_peak.dump_validation) of the same slice, runs
pv_locations_updated (copied from pv-finder_v2 utils/efficiency_utils.py, with
its numba float32 accumulators) on every event's KDE and compares.

Expected differences: none in the number of seeds; positions agree to float32
rounding (pv-finder sums bin * value in double before rounding to float32 and
converts bins to z by a division, Allen stays in float32).
"""
import argparse
import json
import os
import sys

import numpy as np

Z_MIN, BIN_WIDTH, N_BINS, MAX_SEEDS = -100.0, 0.1, 4000, 32
THRESHOLD, INTEGRAL_THRESHOLD, MIN_WIDTH = 0.07, 0.7, 0
MAX_DZ = 1e-4  # mm


def pv_locations_updated(targets, threshold, integral_threshold, min_width):
    """pv-finder_v2 efficiency_utils.pv_locations_updated, same arithmetic."""
    f32 = np.float32
    state = 0
    integral = f32(0.0)
    sum_weights_locs = f32(0.0)
    items = []
    peak_passed = False
    for i in range(len(targets)):
        if targets[i] >= threshold:
            state += 1
            integral = f32(integral + targets[i])
            sum_weights_locs = f32(float(sum_weights_locs) + i * float(targets[i]))
            if (targets[i - 1] > targets[i] + 0.05) and (targets[i - 1] > 1.1 * targets[i]):
                peak_passed = True
        if (targets[i] < threshold or i == len(targets) - 1 or (targets[i - 1] < targets[i] and peak_passed)) and state > 0:
            if state >= min_width and integral >= integral_threshold:
                items.append(f32(sum_weights_locs / integral + f32(0.5)))
            state = 0
            integral = f32(0.0)
            sum_weights_locs = f32(0.0)
            peak_passed = False
    return np.array(items, np.float32)


def read_kde(path):
    raw = np.fromfile(path, np.uint32, 2)
    n_events = int(raw[1])
    return np.fromfile(path, np.float32, offset=8).reshape(n_events, N_BINS)


def read_zpeaks(path):
    raw = np.fromfile(path, np.uint32, 2)
    n_events = int(raw[1])
    rec = np.fromfile(path, np.uint32, offset=8).reshape(n_events, 1 + MAX_SEEDS)
    counts = rec[:, 0].astype(int)
    seeds = rec[:, 1:].copy().view(np.float32)
    return [seeds[e, : counts[e]] for e in range(n_events)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    kde = read_kde(os.path.join(args.dump_dir, "allen_kde_output.bin"))
    allen = read_zpeaks(os.path.join(args.dump_dir, "allen_zpeaks.bin"))
    if len(allen) != len(kde):
        sys.exit(f"{len(kde)} events in the KDE dump, {len(allen)} in the seed dump")

    count_mismatch, n_seeds, max_dz, capped, first_bin_on = 0, 0, 0.0, 0, 0
    for e in range(len(kde)):
        # Allen takes the bin before the first as 0, pv-finder wraps around to
        # the last bin; that only matters when the first bin is above threshold.
        first_bin_on += int(kde[e, 0] >= THRESHOLD)
        ref_bins = pv_locations_updated(kde[e], THRESHOLD, INTEGRAL_THRESHOLD, MIN_WIDTH)
        ref = Z_MIN + ref_bins.astype(np.float64) / (1.0 / BIN_WIDTH)
        if len(ref) > MAX_SEEDS:
            capped += 1
            ref = ref[:MAX_SEEDS]
        if len(ref) != len(allen[e]):
            count_mismatch += 1
            continue
        n_seeds += len(ref)
        if len(ref):
            max_dz = max(max_dz, float(np.abs(ref - allen[e]).max()))

    ok = count_mismatch == 0 and max_dz < MAX_DZ
    report = {"events": len(kde), "seeds": n_seeds, "events_with_different_number_of_seeds": count_mismatch,
              "max_abs_dz_mm": max_dz, "events_over_seed_cap": capped,
              "events_with_first_bin_above_threshold": first_bin_on, "passed": ok}
    print(f"pvfinder_peak vs pv-finder pv_locations_updated: {len(kde)} events, {n_seeds} seeds, "
          f"{count_mismatch} events with a different number of seeds, max |dz| {max_dz:.2e} mm "
          f"(limit {MAX_DZ:g}), {capped} events over the {MAX_SEEDS}-seed cap, {first_bin_on} with the first bin "
          f"above threshold: {'PASS' if ok else 'FAIL'}")
    if args.report:
        with open(args.report, "w") as fp:
            json.dump(report, fp, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
