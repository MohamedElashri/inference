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
from AllenConf.hlt1_calibration_lines import make_passthrough_line
from AllenConf.odin import decode_odin
from AllenConf.persistency import make_persistency
from AllenConf.utils import line_maker
from AllenConf.validators import rate_validation
from AllenCore.generator import generate
from PyConf.control_flow import CompositeNode, NodeLogic

bank_providers = [decode_odin()["dev_odin_data"].producer]

from AllenConf.odin import make_nzs_filter

nzs_filter = [make_nzs_filter(nzsfilter=True)]
passthrough_line = line_maker(make_passthrough_line(pre_scaler=1), prefilter=nzs_filter)
line_algorithms = [passthrough_line[0]]

providers = CompositeNode(
    "Providers", bank_providers, NodeLogic.NONLAZY_AND, force_order=False
)

lines = CompositeNode(
    "AllLines", [passthrough_line[1]], NodeLogic.NONLAZY_OR, force_order=False
)

persistency_node, persistency_algorithms = make_persistency(line_algorithms)

passthrough_sequence = CompositeNode(
    "Passthrough",
    [providers, lines, persistency_node, rate_validation(lines=line_algorithms)],
    NodeLogic.NONLAZY_AND,
    force_order=True,
)

generate(passthrough_sequence)
