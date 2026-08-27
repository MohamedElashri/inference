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
from AllenConf.validators import seeding_xz_validation
from AllenCore.generator import generate
from PyConf.control_flow import CompositeNode, NodeLogic

seed_xz = seeding_xz_validation()
seeding_sequence = CompositeNode(
    "SeedingXZValidation", [seed_xz], NodeLogic.LAZY_AND, force_order=True
)

generate(seeding_sequence)
