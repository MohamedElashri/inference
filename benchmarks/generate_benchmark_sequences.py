#!/usr/bin/env python3
"""Generate the benchmark-only Allen sequences used by the PVFinder campaigns.

Allen only discovers sequences as configuration/python/AllenSequences/*.py
(flat glob), so these files have to live there, but they are generated and
git-ignored rather than tracked:

  hlt1_pp_liteN{,_pvfinder_benchmark,_pvfinder_unet_benchmark}.py  N=1..7
      hlt1_pp_default with progressively more HLT1 work removed, alone, with
      PVFinder FC, and with FC+UNet.
  velo_only{,_pvfinder_benchmark,_pvfinder_unet_benchmark}.py
  hlt1_pp_default_pvfinder{,_unet}_benchmark.py
      copies of tracked sequences under the names benchmark_pvfinder_batch.sh
      expects when PVF_SEQUENCES uses the "velo_only" / "hlt1_pp_default" stems.

Usage:
  benchmarks/generate_benchmark_sequences.py            # write the files
  benchmarks/generate_benchmark_sequences.py --check    # verify, write nothing
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_DIR = os.path.join(REPO_ROOT, "Allen", "configuration", "python", "AllenSequences")

# Each rung removes more HLT1 work than the one before (lite2 is the exception:
# downstream only, full Kalman kept). Values are setup_hlt1_node kwargs in order.
LITE = {
    1: [("with_ut", True), ("enableDownstream", True), ("with_fullKF", False)],
    2: [("with_ut", True), ("enableDownstream", False), ("with_fullKF", True)],
    3: [("with_ut", True), ("enableDownstream", False), ("with_fullKF", False)],
    4: [("with_ut", False), ("enableDownstream", False), ("with_fullKF", False)],
    5: [("with_ut", False), ("enableDownstream", False), ("with_fullKF", False),
        ("with_v0s", False)],
    6: [("with_ut", False), ("enableDownstream", False), ("with_fullKF", False),
        ("with_v0s", False), ("with_calo", False)],
    7: [("with_ut", False), ("enableDownstream", False), ("with_fullKF", False),
        ("with_v0s", False), ("with_calo", False), ("with_muon", False)],
}
REMOVED = {
    1: ["with_fullKF=False"],
    2: ["enableDownstream=False"],
    3: ["with_fullKF=False", "enableDownstream=False"],
    4: ["with_fullKF=False", "enableDownstream=False", "with_ut=False"],
    5: ["with_fullKF=False", "enableDownstream=False", "with_ut=False", "with_v0s=False"],
    6: ["with_fullKF=False", "enableDownstream=False", "with_ut=False", "with_v0s=False",
        "with_calo=False"],
    7: ["with_fullKF=False", "enableDownstream=False", "with_ut=False", "with_v0s=False",
        "with_calo=False", "with_muon=False"],
}

COPIES = {
    "velo_only.py": "velo.py",
    "velo_only_pvfinder_benchmark.py": "pvfinder_fc.py",
    "velo_only_pvfinder_unet_benchmark.py": "pvfinder_unet.py",
    "hlt1_pp_default_pvfinder_benchmark.py": "hlt1_pp_pvfinder_benchmark.py",
    "hlt1_pp_default_pvfinder_unet_benchmark.py": "hlt1_pp_pvfinder_unet_benchmark.py",
}

HEADER = """\
###############################################################################
# Reduced-work HLT1 benchmark sequence -- {name}
# Auto-generated: tests whether freeing HLT1 GPU work changes PVFinder's
# marginal cost at 16 streams.  Removed relative to hlt1_pp_default:
#   {removed}
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.velo_reconstruction import make_pr_velo_tracks
"""

SETUP_CALL = """\
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        threshold_settings=get_thresholds(
            "forward_then_matching_and_downstream_with_parkf_tuned_mu5p3_1200kHz"),
{kwargs}
    )
"""

BIND = """\
with make_velo_scifi_matches.bind(
        ghost_killer_threshold=0.8), make_pr_velo_tracks.bind(
            missing_modules=[21]):
"""

PLAIN_BODY = """
""" + BIND + """\
    hlt1_node = setup_hlt1_node(
""" + SETUP_CALL + """
generate(hlt1_node)
"""

PVF_IMPORTS = {
    "fc": "from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc\nimport os\n",
    "unet": ("from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc\n"
             "from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet\nimport os\n"),
}

PVF_TAIL = {
    "fc": """\
    producer = pvfinder_fc_output["dev_pvfinder_output_histogram"].producer
    hlt1_graph.children = tuple(list(hlt1_graph.children) + [producer])
""",
    "unet": """\
    _dump_dir = os.environ.get("PVFINDER_DUMP_DIR", "")
    pvfinder_unet_output = make_pvfinder_unet(
        pvfinder_fc_output,
        dump_validation=_dump_dir,
    )
    unet_producer = pvfinder_unet_output["unet_producer"]
    hlt1_graph.children = tuple(list(hlt1_graph.children) + [unet_producer])
""",
}

PVF_BODY = """

def hook_pvfinder_to_hlt1():
    hlt1_node_dict = setup_hlt1_node(
""" + SETUP_CALL + """
    hlt1_graph = hlt1_node_dict["control_flow_node"]
    reco = hlt1_node_dict["reconstruction"]

    pvfinder_fc_output = make_pvfinder_fc(reco["velo_tracks"])

{tail}
    return hlt1_graph


""" + BIND + """\
    benchmark_node = hook_pvfinder_to_hlt1()

generate(benchmark_node)
"""


def lite_files():
    for n, kw in LITE.items():
        kwargs = ",\n".join(f"        {k}={v}" for k, v in kw)
        removed = ", ".join(REMOVED[n])
        stem = f"hlt1_pp_lite{n}"
        yield f"{stem}.py", (HEADER.format(name=stem, removed=removed)
                             + PLAIN_BODY.format(kwargs=kwargs))
        for kind, suffix in (("fc", "_pvfinder_benchmark"), ("unet", "_pvfinder_unet_benchmark")):
            name = stem + suffix
            yield f"{name}.py", (HEADER.format(name=name, removed=removed) + PVF_IMPORTS[kind]
                                 + PVF_BODY.format(kwargs=kwargs, tail=PVF_TAIL[kind]))


def copy_files():
    for dst, src in COPIES.items():
        with open(os.path.join(SEQ_DIR, src)) as f:
            yield dst, f.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=SEQ_DIR, help="where to write (default: Allen's AllenSequences)")
    parser.add_argument("--check", action="store_true",
                        help="compare against existing files in --out-dir instead of writing")
    args = parser.parse_args()

    mismatched = 0
    for name, text in list(lite_files()) + list(copy_files()):
        path = os.path.join(args.out_dir, name)
        if args.check:
            try:
                with open(path) as f:
                    same = f.read() == text
            except FileNotFoundError:
                same = False
            if not same:
                print(f"differs or missing: {path}")
                mismatched += 1
        else:
            with open(path, "w") as f:
                f.write(text)
    total = len(LITE) * 3 + len(COPIES)
    if args.check:
        print(f"{total - mismatched}/{total} benchmark sequences match")
        return 1 if mismatched else 0
    print(f"wrote {total} benchmark sequences to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
