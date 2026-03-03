###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
#
# HLT1 + PVFinder full pipeline benchmark sequence.
# Appends the complete PVFinder chain (FC + UNet) after standard HLT1 reco,
# sharing the VELO tracks already reconstructed by HLT1.
#
# Pipeline:
#   HLT1 default reco
#     └─> pvfinder_velo_feature_extraction
#     └─> pvfinder_fc_engine
#     └─> pvfinder_track_aggregation
#     └─> pvfinder_ncw_layout
#     └─> pvfinder_unet
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.velo_reconstruction import make_pr_velo_tracks
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet
from PyConf.control_flow import NodeLogic, CompositeNode
import os


def hook_pvfinder_unet_to_hlt1():
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

    # FC chain: feature extraction -> FC engine -> track aggregation
    pvfinder_fc_output = make_pvfinder_fc(reco["velo_tracks"])

    # UNet chain: NCW layout -> UNet inference
    _dump_dir = os.environ.get("PVFINDER_DUMP_DIR", "")
    pvfinder_unet_output = make_pvfinder_unet(
        pvfinder_fc_output,
        dump_validation=_dump_dir,
    )

    # Append the final UNet producer as an extra child of the HLT1 top node.
    # Allen evaluates children sequentially; appending here means PVFinder
    # runs after all HLT1 lines have been evaluated.
    unet_producer = pvfinder_unet_output["unet_producer"]
    hlt1_graph.children = tuple(list(hlt1_graph.children) + [unet_producer])

    return hlt1_graph


with make_velo_scifi_matches.bind(
        ghost_killer_threshold=0.8), make_pr_velo_tracks.bind(
            missing_modules=[21]):
    benchmark_node = hook_pvfinder_unet_to_hlt1()

generate(benchmark_node)
