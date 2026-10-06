###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.HLT1 import setup_hlt1_node
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc, pvfinder_node
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenCore.generator import generate


def add_pvfinder(hlt1_graph):
    # Identical calls to HLT1 reconstruction reuse its algorithm instances.
    velo_tracks = make_velo_tracks(decode_velo())
    fc = make_pvfinder_fc(velo_tracks)
    producer = fc["dev_pvfinder_output_histogram"].producer
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
