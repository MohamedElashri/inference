"""Event-by-event comparison of two primary-vertex reconstructions in Allen.

Reads the files written by pvfinder_pv_dump (sequence pvfinder_pv_validation):
the beamline PV finder's vertices and PVFinder's, with the MC PVs of the same
events. Reports, for both:

  * MC efficiency, false rate and z resolution with the rules of Allen's
    former standalone PrimaryVertexChecker, first over all z, then for
    MC and reconstructed PVs inside PVFinder's z range only. MC truth is read
    directly from the dumped MDF MC-PV banks; no standalone checker is run;
  * event by event, inside the range: MC PVs found by both, by one only, by
    neither; how many events have the same number of PVs; how far the two
    reconstructions' PVs are apart when they found the same MC PV.

Usage: compare_pvs.py --dir RUN_DIR [--zmin -100 --zmax 300] [--report out.json]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

MIN_TRACKS = 4       # PrimaryVertexChecker: reconstructible MC PV
DZ_ISOLATED = 10.0   # mm, 3D distance to the closest other MC PV
N_SIGMA_MATCH = 5.0  # match if |z_rec - z_mc| < 5 sigma_z(rec)


def read_dump(path):
    """{(batch, event): (rec[n, 9] float32, mc[m, 4] float64)}."""
    data = open(path, "rb").read()
    events, pos = {}, 0
    while pos < len(data):
        batch, ev, n_rec, n_mc = struct.unpack_from("<4I", data, pos)
        pos += 16
        rec = np.frombuffer(data, "<f4", n_rec * 9, pos).reshape(n_rec, 9)
        pos += rec.nbytes
        mc = np.frombuffer(data, "<f8", n_mc * 4, pos).reshape(n_mc, 4)
        pos += mc.nbytes
        events[(batch, ev)] = (rec, mc)
    return events


def check_event(rec, mc, zrange=None):
    """PrimaryVertexChecker on one event. With zrange, MC and reconstructed PVs
    outside it are dropped first. Returns counters and per-MC-PV results."""
    if zrange is not None:
        mc = mc[(mc[:, 2] >= zrange[0]) & (mc[:, 2] < zrange[1])]
        rec = rec[(rec[:, 2] >= zrange[0]) & (rec[:, 2] < zrange[1])]
    n_mc = len(mc)
    mc_rec_index = np.full(n_mc, -1)
    rec_mc_index = np.full(len(rec), -1)
    for i, pv in enumerate(rec):
        if n_mc == 0:
            break
        dz = np.abs(mc[:, 2] - float(pv[2]))
        j = int(np.argmin(dz))
        # cov22 <= 0 (it happens) never matches, as in the checker (sqrt gives NaN).
        if pv[5] > 0 and dz[j] < N_SIGMA_MATCH * np.sqrt(float(pv[5])):
            rec_mc_index[i] = j
            mc_rec_index[j] = i
    dist = np.full(n_mc, 999999.0)
    if n_mc >= 2:
        d = np.sqrt(((mc[:, None, :3] - mc[None, :, :3]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        dist = d.min(1)
    reco_ble = mc[:, 3] >= MIN_TRACKS
    isolated = dist > DZ_ISOLATED
    close = dist < DZ_ISOLATED
    found = mc_rec_index > -1
    counts = {
        "mc": int(reco_ble.sum()), "found": int(found.sum()),
        "mc_isolated": int((isolated & reco_ble).sum()), "found_isolated": int((found & isolated).sum()),
        "mc_close": int((close & reco_ble).sum()), "found_close": int((found & close).sum()),
        "fake": int((rec_mc_index < 0).sum()), "rec": int(len(rec)),
        "rec_bad_cov": int((rec[:, 5] <= 0).sum()),
    }
    return counts, mc, rec, mc_rec_index, reco_ble


def summarise(events, zrange=None):
    tot, dx, dy, dz, sigma_z = {}, [], [], [], []
    for rec, mc in events.values():
        c, mc_sel, rec_sel, mc_rec, reco_ble = check_event(rec, mc, zrange)
        for k, v in c.items():
            tot[k] = tot.get(k, 0) + v
        for j in np.nonzero((mc_rec > -1) & reco_ble)[0]:
            dx.append(float(rec_sel[mc_rec[j], 0]) - mc_sel[j, 0])
            dy.append(float(rec_sel[mc_rec[j], 1]) - mc_sel[j, 1])
            dz.append(float(rec_sel[mc_rec[j], 2]) - mc_sel[j, 2])
            sigma_z.append(np.sqrt(float(rec_sel[mc_rec[j], 5])))
    dx, dy, dz = np.array(dx), np.array(dy), np.array(dz)
    def ratio(a, b):
        return a / b if b else 0.0

    def core_um(d):
        """Half the 15.87-84.13 percentile range (a Gaussian's sigma), in um."""
        return 1e3 * float(0.5 * (np.percentile(d, 84.13) - np.percentile(d, 15.87))) if len(d) else 0.0

    return {
        "efficiency": ratio(tot["found"], tot["mc"]), "found": tot["found"], "mc": tot["mc"],
        "efficiency_isolated": ratio(tot["found_isolated"], tot["mc_isolated"]),
        "efficiency_close": ratio(tot["found_close"], tot["mc_close"]),
        "found_isolated": tot["found_isolated"], "mc_isolated": tot["mc_isolated"],
        "found_close": tot["found_close"], "mc_close": tot["mc_close"],
        "false_rate": ratio(tot["fake"], tot["found"] + tot["fake"]), "fake": tot["fake"],
        "reconstructed": tot["rec"], "reconstructed_with_cov22_not_positive": tot["rec_bad_cov"],
        "dz_mean_um": 1e3 * float(dz.mean()) if len(dz) else 0.0,
        "dz_rms_um": 1e3 * float(dz.std()) if len(dz) else 0.0,
        "dz_core_sigma_um": core_um(dz),
        "dx_core_sigma_um": core_um(dx),
        "dy_core_sigma_um": core_um(dy),
        "sigma_z_median_um": 1e3 * float(np.median(sigma_z)) if len(sigma_z) else 0.0,
    }


def pairwise(a_events, b_events, zrange):
    """Event-by-event comparison of reconstructions a and b inside zrange."""
    keys = sorted(set(a_events) & set(b_events))
    out = {"events": len(keys), "same_number_of_pvs": 0,
           "mc_found_by_both": 0, "mc_found_by_a_only": 0, "mc_found_by_b_only": 0, "mc_found_by_neither": 0,
           "a_only_isolated": 0, "b_only_isolated": 0,
           "a_only_ntracks": [], "b_only_ntracks": [], "neither_ntracks": [], "both_ntracks": []}
    dz_ab, a_only_z, b_only_z = [], [], []
    for key in keys:
        ca, mc, rec_a, mc_a, ble = check_event(*a_events[key], zrange)
        cb, _, rec_b, mc_b, _ = check_event(*b_events[key], zrange)
        out["same_number_of_pvs"] += int(len(rec_a) == len(rec_b))
        if len(mc) >= 2:
            d = np.sqrt(((mc[:, None, :3] - mc[None, :, :3]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            isolated = d.min(1) > DZ_ISOLATED
        else:
            isolated = np.ones(len(mc), bool)
        for j in np.nonzero(ble)[0]:
            fa, fb = mc_a[j] > -1, mc_b[j] > -1
            nt = int(mc[j, 3])
            if fa and fb:
                out["mc_found_by_both"] += 1
                out["both_ntracks"].append(nt)
                dz_ab.append(float(rec_a[mc_a[j], 2]) - float(rec_b[mc_b[j], 2]))
            elif fa:
                out["mc_found_by_a_only"] += 1
                out["a_only_isolated"] += int(isolated[j])
                out["a_only_ntracks"].append(nt)
                a_only_z.append(mc[j, 2])
            elif fb:
                out["mc_found_by_b_only"] += 1
                out["b_only_isolated"] += int(isolated[j])
                out["b_only_ntracks"].append(nt)
                b_only_z.append(mc[j, 2])
            else:
                out["mc_found_by_neither"] += 1
                out["neither_ntracks"].append(nt)
    dz_ab = np.array(dz_ab)
    for k in ("a_only_ntracks", "b_only_ntracks", "neither_ntracks", "both_ntracks"):
        v = out.pop(k)
        out[k.replace("ntracks", "median_mc_tracks")] = float(np.median(v)) if v else 0.0
    out["dz_a_minus_b_same_mc_pv_um"] = {
        "median": 1e3 * float(np.median(dz_ab)) if len(dz_ab) else 0.0,
        "rms": 1e3 * float(dz_ab.std()) if len(dz_ab) else 0.0,
        "fraction_within_100um": float((np.abs(dz_ab) < 0.1).mean()) if len(dz_ab) else 0.0,
    }
    return out


def fmt(name, s):
    return (f"  {name:<10} eff {100 * s['efficiency']:5.1f}% ({s['found']}/{s['mc']})  "
            f"isolated {100 * s['efficiency_isolated']:5.1f}% ({s['found_isolated']}/{s['mc_isolated']})  "
            f"close {100 * s['efficiency_close']:5.1f}% ({s['found_close']}/{s['mc_close']})  "
            f"false {100 * s['false_rate']:4.1f}% ({s['fake']})  "
            f"dz core {s['dz_core_sigma_um']:5.1f} um, rms {s['dz_rms_um']:6.1f} um, mean {s['dz_mean_um']:+5.1f} um")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="run directory with pvs_beamline.bin and pvs_pvfinder.bin")
    ap.add_argument("--zmin", type=float, default=-100.0)
    ap.add_argument("--zmax", type=float, default=300.0)
    ap.add_argument("--report", help="write the numbers as JSON")
    args = ap.parse_args()

    a = read_dump(os.path.join(args.dir, "pvs_beamline.bin"))
    b = read_dump(os.path.join(args.dir, "pvs_pvfinder.bin"))
    if set(a) != set(b):
        sys.exit("the two dumps hold different events")
    zr = (args.zmin, args.zmax)
    report = {
        "events": len(a),
        "all_z": {"beamline": summarise(a), "pvfinder": summarise(b)},
        "z_range": list(zr),
        "in_range": {"beamline": summarise(a, zr), "pvfinder": summarise(b, zr)},
        "event_by_event_in_range": pairwise(a, b, zr),
    }
    mc_all = sum(int((mc[:, 3] >= MIN_TRACKS).sum()) for _, mc in a.values())
    mc_in = report["in_range"]["beamline"]["mc"]
    print(f"{len(a)} events; reconstructible MC PVs {mc_all}, of which {mc_in} in {zr[0]:g} <= z < {zr[1]:g} mm")
    print("All z (Allen PrimaryVertexChecker rules; same numbers as pv_validator):")
    print(fmt("beamline", report["all_z"]["beamline"]))
    print(fmt("pvfinder", report["all_z"]["pvfinder"]))
    print(f"MC and reconstructed PVs in {zr[0]:g} <= z < {zr[1]:g} mm:")
    print(fmt("beamline", report["in_range"]["beamline"]))
    print(fmt("pvfinder", report["in_range"]["pvfinder"]))
    p = report["event_by_event_in_range"]
    print("Event by event, in range (a = beamline, b = pvfinder):")
    print(f"  same number of PVs in {p['same_number_of_pvs']}/{p['events']} events")
    print(f"  reconstructible MC PVs found by both {p['mc_found_by_both']}, beamline only {p['mc_found_by_a_only']} "
          f"({p['a_only_isolated']} isolated, median {p['a_only_median_mc_tracks']:g} tracks), pvfinder only "
          f"{p['mc_found_by_b_only']} ({p['b_only_isolated']} isolated, median {p['b_only_median_mc_tracks']:g} tracks), "
          f"neither {p['mc_found_by_neither']} (median {p['neither_median_mc_tracks']:g} tracks)")
    d = p["dz_a_minus_b_same_mc_pv_um"]
    print(f"  same MC PV found by both: z(beamline) - z(pvfinder) median {d['median']:+.1f} um, rms {d['rms']:.1f} um, "
          f"{100 * d['fraction_within_100um']:.1f}% within 100 um")
    if args.report:
        with open(args.report, "w") as fp:
            json.dump(report, fp, indent=2)


if __name__ == "__main__":
    main()
