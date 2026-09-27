###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
# HLT1 + PVFinder full pipeline benchmark sequence.
# Adds the complete PVFinder chain (FC + UNet) to standard HLT1, behind the
# HLT1 physics prefilters, sharing the VELO tracks reconstructed by HLT1.
# Pipeline:
#   HLT1 default reco
#     └─> pvfinder_fc_aggregation (computes the per-track features in its CSR build)
#     └─> pvfinder_unet (consumes dev_pvfinder_interval_features directly)

from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.HLT1 import setup_hlt1_node
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc, pvfinder_node
from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet
from AllenConf.velo_reconstruction import make_pr_velo_tracks
from AllenCore.generator import generate


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

    # FC chain: per-track features, FC network and sum over each interval's tracks (one algorithm).
    pvfinder_fc_output = make_pvfinder_fc(reco["velo_tracks"])

    # UNet inference consumes the FC interval features directly.
    pvfinder_unet_output = make_pvfinder_unet(pvfinder_fc_output)

    # Add PVFinder to the HLT1 top node, behind the HLT1 physics prefilters
    # (see pvfinder_node). Allen schedules it by its data dependencies: right
    # after the VELO Kalman filter, not after the lines.
    unet_producer = pvfinder_unet_output["unet_producer"]
    hlt1_graph.children = tuple(
        list(hlt1_graph.children) + [pvfinder_node(unet_producer)]
    )

    return hlt1_graph


with (
    make_velo_scifi_matches.bind(ghost_killer_threshold=0.8),
    make_pr_velo_tracks.bind(missing_modules=[21]),
):
    benchmark_node = hook_pvfinder_unet_to_hlt1()

generate(benchmark_node)
