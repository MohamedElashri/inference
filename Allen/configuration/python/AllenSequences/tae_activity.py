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
from AllenConf.filters import make_tae_activity_filter
from AllenConf.hlt1_calibration_lines import make_passthrough_line
from AllenConf.hlt1_reconstruction import hlt1_reconstruction
from AllenConf.odin import odin_error_filter, tae_filter
from AllenConf.persistency import make_persistency, make_routingbits_writer, rb_map
from AllenConf.utils import line_maker
from AllenConf.validators import rate_validation
from AllenCore.generator import generate
from PyConf.control_flow import CompositeNode, NodeLogic

lines = []

# Reconstruct objects needed as input for selection lines
reconstructed_objects = hlt1_reconstruction(
    with_calo=True,
    with_ut=False,
    with_muon=True,
    enableDownstream=False,
    tracking_type=TrackingType.FORWARD_THEN_MATCHING,
    velo_open=False,
    with_AC_split=False,
    with_rich=False,
)

tae_passthrough = True
tae_activity = True
routingbit_map = rb_map
routingbit_map.update(
    {
        "Hlt1ActivityPassthrough": 15,
    }
)
print(routingbit_map)

prefilters = [odin_error_filter("odin_error_filter")]
if tae_passthrough:
    if tae_activity:
        tae_activity_filter = make_tae_activity_filter(
            reconstructed_objects["long_tracks"],
            reconstructed_objects["velo_tracks"],
            min_tracks=10,
        )

        tae_filters = CompositeNode(
            "taefilter_node",
            [tae_activity_filter, tae_filter()],
            NodeLogic.LAZY_AND,
            force_order=True,
        )
    else:
        tae_filters = tae_filter()

    with line_maker.bind(prefilter=prefilters + [tae_filters]):
        lines += [
            line_maker(make_passthrough_line(name="Hlt1TAEPassthrough", pre_scaler=1))
        ]

    with line_maker.bind(prefilter=prefilters + [tae_activity_filter]):
        lines += [
            line_maker(
                make_passthrough_line(name="Hlt1ActivityPassthrough", pre_scaler=1)
            )
        ]

line_algorithms = [tup[0] for tup in lines]

with make_routingbits_writer.bind(rb_map=routingbit_map):
    persistency_node, _ = make_persistency(line_algorithms)

lines = CompositeNode(
    "AllLines", [tup[1] for tup in lines], NodeLogic.NONLAZY_OR, force_order=False
)

hlt1_node = CompositeNode(
    "ActivityTAE",
    [lines, persistency_node, rate_validation(lines=line_algorithms)],
    NodeLogic.NONLAZY_AND,
    force_order=True,
)

generate(hlt1_node)
