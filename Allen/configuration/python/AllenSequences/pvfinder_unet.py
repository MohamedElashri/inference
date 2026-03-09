###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
###############################################################################

from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate

decoded_velo = decode_velo()
velo_tracks = make_velo_tracks(decoded_velo)

import os
_dump_dir = os.environ.get("PVFINDER_DUMP_DIR", "")

# FC chain: feature extraction -> FC engine -> track aggregation
pvfinder_fc_output = make_pvfinder_fc(velo_tracks, dump_validation=_dump_dir)

# UNet chain: NCW layout -> UNet inference
# Set dump_validation to a directory path to write allen_ncw_input.bin and
# allen_kde_output.bin on the first processed slice (for numerical validation).
pvfinder_unet_output = make_pvfinder_unet(pvfinder_fc_output, dump_validation=_dump_dir)

# Drive the graph from the UNet producer (last algorithm in chain)
unet_producer = pvfinder_unet_output["unet_producer"]

node = CompositeNode(
    "PVFinderUNet", [unet_producer], NodeLogic.NONLAZY_AND, force_order=True)

generate(node)
