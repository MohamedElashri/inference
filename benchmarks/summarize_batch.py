#!/usr/bin/env python3
"""Write human-readable summaries for every measured Allen sequence role."""
import argparse
import csv
from pathlib import Path

from runs import ALL_SEQUENCE_KEYS, results_block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    _, results = results_block(str(args.batch_dir))
    summary = results["summary"]
    medians = summary.get("median_events_per_s", {})
    roles = [role for role in ALL_SEQUENCE_KEYS if role in medians]
    # Preserve the original seven TSV columns for existing consumers.
    fields = ["run", "baseline", "fc", "unet", "fc_overhead_pct",
              "unet_overhead_pct", "unet_retention_pct"]
    for role in roles:
        if role not in ("baseline", "fc", "unet"):
            fields.extend((role, f"{role}_overhead_pct", f"{role}_retention_pct"))
    with (args.batch_dir / "summary.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for repeat in results["repeats"]:
            row = {"run": repeat["run"], **repeat["events_per_s"]}
            row.update({key: value for key, value in repeat.items() if key in fields})
            writer.writerow({key: (f"{row[key]:.2f}" if key != "run" else row[key])
                             for key in fields if key in row})
    labels = {"baseline": "baseline", "fc": "FC", "unet": "FC+UNet",
              "pvs": "full PV shadow chain", "replace": "PVFinder replacement",
              "hybrid": "hybrid PV chain"}
    lines = ["# PVFinder benchmark summary", "", f"- repeats: {len(results['repeats'])}"]
    for role in roles:
        lines.append(f"- median {labels[role]} events/s: {medians[role]:.2f}")
        for metric in ("overhead_pct", "retention_pct"):
            key = f"median_{role}_{metric}"
            if key in summary:
                lines.append(f"- median {labels[role]} {metric.removesuffix('_pct')}: {summary[key]:.2f}%")
    if "baseline_spread_pct" in summary:
        lines.append(f"- baseline spread: {summary['baseline_spread_pct']:.2f}%")
        lines.append(f"- contention status: {summary['contention']}")
    if summary.get("slice_splits"):
        lines.append("- slice splits observed; rates are not at the requested slice size")
    (args.batch_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
