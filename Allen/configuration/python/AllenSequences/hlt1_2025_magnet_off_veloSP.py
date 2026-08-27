###############################################################################
# (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenConf.enum_types import TrackingType
from AllenConf.HLT1 import setup_hlt1_node
from AllenConf.hlt1_presets import VELO_MICRO_BIAS_PRESETS
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.velo_reconstruction import decode_velo
from AllenCore.generator import generate


def modify_presets():
    VELO_MICRO_BIAS_PRESETS["pp"]["Hlt1VeloMicroBias"]["post_scaler"] = 1.0


with (
    make_velo_scifi_matches.bind(ghost_killer_threshold=0.8),
    decode_velo.bind(retina_decoding=False),
):  # Fully enable VELO micro bias
    hlt1_node = setup_hlt1_node(
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        with_ut=True,
        with_fullKF=True,
        enableAlignment=False,  # Disable alignment during magnet off
        enableDownstream=False,  # Downstream not used in technical lines
        enablePhysics=False,  # Only enable technical lines
        withSMOG2=False,
        preset_modifiers=[modify_presets],
    )  # Only enable technical lines

generate(hlt1_node)
