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
from AllenConf.hlt1_reconstruction import hlt1_reconstruction
from AllenConf.secondary_vertex_reconstruction import ParKF_cuts as chi2_cuts
from AllenConf.validators import VertexingValidator
from AllenCore.generator import generate
from PyConf.control_flow import CompositeNode, NodeLogic

reconstructed_objects = hlt1_reconstruction(
    with_calo=True,
    with_ut=True,
    with_muon=True,
    enableDownstream=False,
    tracking_type=TrackingType.FORWARD_THEN_MATCHING,
    velo_open=False,
    with_AC_split=False,
    with_rich=False,
    with_fullKF=True,
    with_ttracks=False,
    with_downstream_KF=False,
    track_max_chi2ndof=chi2_cuts.SV_track_max_chi2ndof,
)

three_body_svs = reconstructed_objects["three_body_svs"]
vertexing_validator = VertexingValidator(three_body_svs)

multibodySV_validator = CompositeNode(
    "Validator", [vertexing_validator], NodeLogic.LAZY_AND, force_order=True
)

generate(multibodySV_validator)
