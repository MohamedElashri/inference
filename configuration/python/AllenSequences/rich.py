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
from AllenConf.rich_reconstruction import decode_rich, make_pixels
from AllenCore.generator import generate
from PyConf.control_flow import NodeLogic, CompositeNode

decoded_rich1 = decode_rich(rich=1)
decoded_rich2 = decode_rich(rich=2)

rich1_decoding = CompositeNode(
    "Rich1Decoding", [decoded_rich1["dev_smart_ids"].producer],
    NodeLogic.NONLAZY_OR,
    force_order=True)

rich2_decoding = CompositeNode(
    "Rich2Decoding", [decoded_rich2["dev_smart_ids"].producer],
    NodeLogic.NONLAZY_OR,
    force_order=True)

rich1_make_pixels = CompositeNode(
    "Rich1MakePixels",
    [make_pixels(decoded_rich1, rich=1)["dev_rich_pixels"].producer],
    NodeLogic.NONLAZY_OR,
    force_order=True)

rich2_make_pixels = CompositeNode(
    "Rich2MakePixels",
    [make_pixels(decoded_rich2, rich=2)["dev_rich_pixels"].producer],
    NodeLogic.NONLAZY_OR,
    force_order=True)

rich_node = CompositeNode(
    "RichNode",
    [rich1_decoding, rich1_make_pixels, rich2_decoding, rich2_make_pixels],
    NodeLogic.NONLAZY_OR,
    force_order=True)

generate(rich_node)
