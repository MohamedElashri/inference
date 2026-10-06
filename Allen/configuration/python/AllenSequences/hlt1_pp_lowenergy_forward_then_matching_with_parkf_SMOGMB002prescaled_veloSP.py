###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "COPYING".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenConf.HLT1_lowenergy import setup_hlt1_node
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.velo_reconstruction import decode_velo
from AllenCore.generator import generate

with make_velo_scifi_matches.bind(ghost_killer_threshold=0.8):
    with decode_velo.bind(retina_decoding=False):
        hlt1_node = setup_hlt1_node(
            smog2_lumi_prescale=1.0,
            smog2_mb_prescale=0.02,
            bb_nobias_prescale=0.0,
        )

generate(hlt1_node)
