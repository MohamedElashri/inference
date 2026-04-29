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
from AllenConf.codex_b import decode_codex, codex_clustering, codex_coincidence, codex_validate_event
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate
from AllenConf.primary_vertex_reconstruction import make_pvs

validation_sequence = codex_validate_event()

codex_validation_sequence = CompositeNode(
    "CODEX", [validation_sequence], NodeLogic.NONLAZY_AND, force_order=True)

generate(codex_validation_sequence)
