###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
#
# Standalone NN PV chain sequence (no HLT1 tracking).
#
# Pipeline:
#   decode_velo
#     └─> make_velo_tracks
#     └─> pv_beamline_extrapolate        (PVTrack objects for fitter)
#     └─> pvfinder_velo_feature_extraction
#     └─> pvfinder_fc_aggregation
#     └─> pvfinder_unet
#     └─> pvfinder_kde_peak_finder
#     └─> pvfinder_nn_calculate_denom
#     └─> pvfinder_vertex_fitter
#     └─> pvfinder_nn_cleanup            (final PV::Vertex array)
#
# Set the PVFINDER_DUMP_DIR environment variable to a directory path to write
# binary validation dumps (allen_nn_zpeaks.bin, allen_nn_vertices.bin) on the
# first processed slice.
###############################################################################
import os
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenConf.primary_vertex_reconstruction import make_nn_pvs
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate

decoded_velo = decode_velo()
velo_tracks = make_velo_tracks(decoded_velo)

_dump_dir = os.environ.get("PVFINDER_DUMP_DIR", "")

nn_pvs_output = make_nn_pvs(velo_tracks, dump_validation=_dump_dir)

# Drive the graph from the last algorithm in the chain (nn_cleanup).
nn_cleanup_producer = nn_pvs_output["dev_multi_final_vertices"].producer

node = CompositeNode(
    "PVFinderNN", [nn_cleanup_producer], NodeLogic.LAZY_AND, force_order=True)

generate(node)
