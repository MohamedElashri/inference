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
from AllenConf.hlt1_presets import MONITORING_CONFIG_PRESETS
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenCore.generator import generate


def modify_presets():
    MONITORING_CONFIG_PRESETS["pp"]["enable_bgi_full"] = True


with make_velo_scifi_matches.bind(ghost_killer_threshold=0.8):
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
        preset_modifiers=[modify_presets],
    )

generate(hlt1_node)
