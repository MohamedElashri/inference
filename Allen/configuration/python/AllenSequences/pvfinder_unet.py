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

from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenCore.generator import generate
from PyConf.control_flow import CompositeNode, NodeLogic

decoded_velo = decode_velo()
velo_tracks = make_velo_tracks(decoded_velo)

# FC chain: per-track features, FC network and sum over each interval's tracks (one algorithm).
pvfinder_fc_output = make_pvfinder_fc(velo_tracks)

# UNet inference consumes the FC interval features directly.
# Set dump_validation to a directory path to write allen_ncw_input.bin and
# allen_kde_output.bin on the first processed slice (for numerical validation).
import os

_dump_dir = os.environ.get("PVFINDER_DUMP_DIR", "")
pvfinder_unet_output = make_pvfinder_unet(pvfinder_fc_output, dump_validation=_dump_dir)

# Drive the graph from the UNet producer (last algorithm in chain)
unet_producer = pvfinder_unet_output["unet_producer"]

node = CompositeNode(
    "PVFinderUNet", [unet_producer], NodeLogic.LAZY_AND, force_order=True
)

generate(node)
