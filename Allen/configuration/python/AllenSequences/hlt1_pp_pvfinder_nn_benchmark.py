###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
#
# HLT1 + full NN PV chain benchmark sequence.
# Appends the complete NN PV chain (FC + UNet + KDE + fitter + cleanup) after
# standard HLT1 reconstruction, sharing the VELO tracks already produced by
# HLT1.
#
# Pipeline:
#   HLT1 default reco
#     └─> pv_beamline_extrapolate        (PVTrack objects for fitter)
#     └─> pvfinder_velo_feature_extraction
#     └─> pvfinder_fc_aggregation
#     └─> pvfinder_unet
#     └─> pvfinder_kde_peak_finder
#     └─> pvfinder_nn_calculate_denom
#     └─> pvfinder_vertex_fitter
#     └─> pvfinder_nn_cleanup            (final PV::Vertex array)
#
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.velo_reconstruction import make_pr_velo_tracks
from AllenConf.primary_vertex_reconstruction import make_nn_pvs


def hook_pvfinder_nn_to_hlt1():
    hlt1_node_dict = setup_hlt1_node(
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        threshold_settings=get_thresholds(
            "forward_then_matching_and_downstream_with_parkf_tuned_mu5p3_1200kHz"
        ),
        with_ut=True,
        enableDownstream=True,
        with_fullKF=True,
    )

    hlt1_graph = hlt1_node_dict["control_flow_node"]
    reco = hlt1_node_dict["reconstruction"]

    # Full NN PV chain: FC → UNet → KDE peaks → denom → fitter → cleanup
    nn_pvs_output = make_nn_pvs(reco["velo_tracks"])

    # Append the cleanup algorithm (last in the NN chain) as an extra child of
    # the HLT1 top node.  Allen evaluates children sequentially; appending here
    # means the NN PV chain runs after all HLT1 lines have been evaluated.
    nn_cleanup_producer = nn_pvs_output["dev_multi_final_vertices"].producer
    hlt1_graph.children = tuple(
        list(hlt1_graph.children) + [nn_cleanup_producer])

    return hlt1_graph


with make_velo_scifi_matches.bind(
        ghost_killer_threshold=0.8), make_pr_velo_tracks.bind(
            missing_modules=[21]):
    benchmark_node = hook_pvfinder_nn_to_hlt1()

generate(benchmark_node)
