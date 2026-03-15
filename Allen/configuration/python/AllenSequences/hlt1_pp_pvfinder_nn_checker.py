###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
#
# HLT1 + NN PV chain + MC-truth checker sequence.
#
# Runs the full HLT1 reconstruction (including the CLASSICAL PV chain and its
# PV checker) AND the NN PV chain in parallel, so that both PV algorithms are
# validated on the same MC events in a single run.
#
# Checker outputs (in PrCheckerPlots.root):
#   "pv_validator"    — classical pv_beamline chain (standard HLT1 baseline)
#   "nn_pv_validator" — NN PV chain (pvfinder: FC → UNet → KDE → fitter → cleanup)
#
# Usage:
#   ./toolchain/wrapper ./Allen \
#       --sequence hlt1_pp_pvfinder_nn_checker \
#       --mdf <input.mdf> --events 1000
#
# Both chains share:
#   - The same VELO track reconstruction
#   - The same pv_beamline_extrapolate PVTrack objects (read-only sharing)
#   - The same MC events (host_mc_events_t) for truth matching
#
# The NN coverage is pp-only (z ∈ [-100, +300] mm); the classical chain
# covers z ∈ [-541, +307] mm including the SMOG2 region.
###############################################################################
import os
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.velo_reconstruction import make_pr_velo_tracks
from AllenConf.primary_vertex_reconstruction import make_nn_pvs
from AllenConf.validators import pv_validation


def hook_nn_pv_checker_to_hlt1():
    # ------------------------------------------------------------------ #
    # 1. Standard HLT1 with MC-truth checking enabled.                   #
    #    This sets up the classical PV chain and pv_validator.            #
    # ------------------------------------------------------------------ #
    hlt1_node_dict = setup_hlt1_node(
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        threshold_settings=get_thresholds(
            "forward_then_matching_and_downstream_with_parkf_tuned_mu5p3_1200kHz"
        ),
        with_ut=True,
        enableDownstream=True,
        with_fullKF=True,
        withMCChecking=True,
    )

    reco = hlt1_node_dict["reconstruction"]
    validator_node = hlt1_node_dict["validator_node"]

    # ------------------------------------------------------------------ #
    # 2. NN PV chain (shares VELO tracks from HLT1 reconstruction).      #
    #    make_nn_pvs() returns the same dict keys as make_pvs() so the   #
    #    pv_validation() helper wires it transparently.                  #
    # ------------------------------------------------------------------ #
    _dump_dir = os.environ.get("PVFINDER_DUMP_DIR", "")
    nn_pvs_output = make_nn_pvs(
        reco["velo_tracks"],
        dump_validation=_dump_dir)

    # ------------------------------------------------------------------ #
    # 3. NN PV checker — second PVChecker instance, same ROOT file.      #
    #    Output is labelled "nn_pv_validator" in PrCheckerPlots.root.    #
    # ------------------------------------------------------------------ #
    nn_pv_validator_alg = pv_validation(
        nn_pvs_output,
        name="nn_pv_validator")

    # ------------------------------------------------------------------ #
    # 4. Append the NN PV validator to the existing "Validators" node.   #
    #    Allen's scheduler resolves the NN chain data dependencies         #
    #    automatically (nn_pv_validator needs dev_multi_final_vertices    #
    #    from pvfinder_nn_cleanup, which in turn needs the full upstream  #
    #    NN chain — all pulled in transitively).                          #
    # ------------------------------------------------------------------ #
    validator_node.children = tuple(
        list(validator_node.children) + [nn_pv_validator_alg])

    return hlt1_node_dict["control_flow_node"]


with make_velo_scifi_matches.bind(
        ghost_killer_threshold=0.8), make_pr_velo_tracks.bind(
            missing_modules=[21]):
    checker_node = hook_nn_pv_checker_to_hlt1()

generate(checker_node)
