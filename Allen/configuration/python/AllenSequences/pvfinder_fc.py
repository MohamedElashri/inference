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

from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenCore.generator import generate
from PyConf.control_flow import CompositeNode, NodeLogic

decoded_velo = decode_velo()
velo_tracks = make_velo_tracks(decoded_velo)

# Execute PVFinder feature extraction and FC aggregation.
pvfinder_fc_output = make_pvfinder_fc(velo_tracks)

# Isolate the final algorithm producer
aggregation_producer = pvfinder_fc_output["dev_pvfinder_output_histogram"].producer

node = CompositeNode(
    "PVFinderFC", [aggregation_producer], NodeLogic.LAZY_AND, force_order=True
)

config = {
    "control_flow_node": node,
    "reconstruction": {"velo_tracks": velo_tracks, "pvfinder_fc": pvfinder_fc_output},
}

generate(node)
