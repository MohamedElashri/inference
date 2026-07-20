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
from AllenConf.plume_reconstruction import decode_plume
from AllenCore.generator import generate
from PyConf.control_flow import CompositeNode, NodeLogic

decoded_plume = decode_plume()
algo = decoded_plume["plume_algo"]

decode = CompositeNode("decode_", [algo], NodeLogic.NONLAZY_OR, force_order=False)

generate(decode)
