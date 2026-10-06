###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
# HLT1 + PVFinder with its primary vertices: hlt1_pp_pvfinder_unet_benchmark
# plus pvfinder_peak and the beamline PV association and fit on its seeds, to
# measure what the vertices add. The HLT1 lines still use the beamline PVs.
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.HLT1 import setup_hlt1_node
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc, pvfinder_node
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenCore.generator import generate
from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet
from AllenConf.pvfinder_pv_reconstruction import make_pvfinder_pvs


def add_pvfinder(hlt1_graph):
    # Identical calls to HLT1 reconstruction reuse its algorithm instances.
    velo_tracks = make_velo_tracks(decode_velo())
    fc = make_pvfinder_fc(velo_tracks)
    unet = make_pvfinder_unet(fc)
    pvs = make_pvfinder_pvs(velo_tracks, unet["dev_pvfinder_kde_output"])
    producer = pvs["dev_multi_final_vertices"].producer
    hlt1_graph.children = tuple(list(hlt1_graph.children) + [pvfinder_node(producer)])
    return hlt1_graph


with make_velo_scifi_matches.bind(ghost_killer_threshold=0.8):
    benchmark_node = setup_hlt1_node(
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        threshold_settings=get_thresholds(
            "forward_then_matching_and_downstream_with_parkf_tuned_mu5p3_1200kHz"
        ),
        with_ut=True,
        enableDownstream=True,
        with_fullKF=True,
        user_hooks=add_pvfinder,
    )

generate(benchmark_node)
