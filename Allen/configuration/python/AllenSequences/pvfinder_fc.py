###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################

from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate

decoded_velo = decode_velo()
velo_tracks = make_velo_tracks(decoded_velo)

# Execute PVFinder Feature Extraction -> FC Engine -> Track Aggregation
pvfinder_fc_output = make_pvfinder_fc(velo_tracks)

# Isolate the final algorithm producer
aggregation_producer = pvfinder_fc_output["dev_pvfinder_output_histogram"].producer

node = CompositeNode(
    "PVFinderFC", [aggregation_producer], NodeLogic.LAZY_AND, force_order=True)

config = {
    'control_flow_node': node,
    'reconstruction': {
        'velo_tracks': velo_tracks,
        'pvfinder_fc': pvfinder_fc_output
    }
}

generate(node)
