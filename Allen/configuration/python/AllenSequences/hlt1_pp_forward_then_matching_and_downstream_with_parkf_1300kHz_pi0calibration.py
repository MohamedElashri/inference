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
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.HLT1 import setup_hlt1_node
from AllenConf.hlt1_calibration_lines import (
    make_pi02gammagamma_inner_line,
    make_pi02gammagamma_line,
    make_pi02gammagamma_middle_line,
    make_pi02gammagamma_middleinnermixed_line,
    make_pi02gammagamma_middleoutermixed_line,
    make_pi02gammagamma_outer_line,
)
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenCore.generator import generate

with (
    make_velo_scifi_matches.bind(ghost_killer_threshold=0.8),
    make_pi02gammagamma_line.bind(pre_scaler=1.0),
    make_pi02gammagamma_inner_line.bind(pre_scaler=1.0),
    make_pi02gammagamma_middle_line.bind(pre_scaler=1.0),
    make_pi02gammagamma_outer_line.bind(pre_scaler=1.0),
    make_pi02gammagamma_middleinnermixed_line.bind(pre_scaler=1.0),
    make_pi02gammagamma_middleoutermixed_line.bind(pre_scaler=1.0),
):
    hlt1_node = setup_hlt1_node(
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        threshold_settings=get_thresholds(
            "forward_then_matching_and_downstream_with_parkf_tuned_mu5p3_1300kHz"
        ),
        with_fullKF=True,
        with_ut=True,
        enableDownstream=True,
        with_downstream_KF=True,
        with_ttracks=True,
        with_quirks=True,
    )

generate(hlt1_node)
