"""Peak-level agreement of two KDE outputs, for the reduced-precision (BF16) path.

The BF16 path rounds weights and activations to bfloat16 (8-bit mantissa), so
float32-level agreement with PyTorch is not expected. validate_model.py and
validate_unet.py judge it on what a peak finder would see instead.
"""
import json
import os

import numpy as np

# Limits, well above today's values (0.21%, 0.6% and 0.1% on 2024 minimum bias).
MAX_BINS_OFF = 0.01          # fraction of bins with |Allen - PyTorch| > 0.01
MAX_MEDIAN_PEAK_CHANGE = 0.02  # median relative change of peak height (peaks: local maxima above 0.07)
MAX_PEAK_MOVES = 0.01        # fraction of intervals with a peak whose highest bin moves by more than one bin
CRITERION = (f"BF16 path: bins off by > 0.01 < {MAX_BINS_OFF:.0%}, median peak height change "
             f"< {MAX_MEDIAN_PEAK_CHANGE:.0%}, highest bin moved > 1 bin < {MAX_PEAK_MOVES:.0%} of intervals with a peak")


def bf16_path(dump_dir):
    """True when the dump's Allen configuration ran the UNet's BF16 path."""
    cfg = os.path.join(dump_dir, "config.json")
    if not os.path.isfile(cfg):
        return False
    with open(cfg) as fp:
        unet = json.load(fp).get("pvfinder_unet", {})
    return unet.get("precision") == "bfloat16"


def peak_agreement(allen, ref):
    """Peak-level numbers for Allen vs PyTorch KDEs (any shape with 100 bins last)."""
    a = np.asarray(allen, np.float64).reshape(-1, 100)
    r = np.asarray(ref, np.float64).reshape(-1, 100)
    is_peak = (r[:, 1:-1] > r[:, :-2]) & (r[:, 1:-1] >= r[:, 2:]) & (r[:, 1:-1] > 0.07)
    iv, b = np.nonzero(is_peak)
    b = b + 1
    rel = np.abs(a[iv, b] - r[iv, b]) / r[iv, b]
    with_peak = r.max(1) > 0.07
    moved = np.abs(np.argmax(a[with_peak], 1) - np.argmax(r[with_peak], 1))
    return {
        "bins_off_by_more_than_0.01": float((np.abs(a - r) > 0.01).mean()),
        "n_peaks": int(len(rel)),
        "median_rel_peak_height_change": float(np.median(rel)) if len(rel) else 0.0,
        "max_rel_peak_height_change": float(rel.max()) if len(rel) else 0.0,
        "intervals_with_peak": int(with_peak.sum()),
        "highest_bin_moved": int((moved > 0).sum()),
        "highest_bin_moved_more_than_one_bin": int((moved > 1).sum()),
    }


def passes(p):
    return (p["bins_off_by_more_than_0.01"] < MAX_BINS_OFF
            and p["median_rel_peak_height_change"] < MAX_MEDIAN_PEAK_CHANGE
            and p["highest_bin_moved_more_than_one_bin"] < MAX_PEAK_MOVES * max(p["intervals_with_peak"], 1))


def describe(p):
    return (f"bins with |Allen - PyTorch| > 0.01: {100 * p['bins_off_by_more_than_0.01']:.2f}%; peaks > 0.07: "
            f"{p['n_peaks']}, height change median {100 * p['median_rel_peak_height_change']:.2f}%, max "
            f"{100 * p['max_rel_peak_height_change']:.1f}%; highest bin moved in {p['highest_bin_moved']} of "
            f"{p['intervals_with_peak']} intervals with a peak ({p['highest_bin_moved_more_than_one_bin']} by more "
            f"than one bin)")
