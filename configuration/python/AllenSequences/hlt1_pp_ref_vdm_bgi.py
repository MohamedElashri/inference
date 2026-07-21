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
from AllenConf.get_thresholds import get_thresholds
from AllenConf.HLT1_pp_ref import setup_hlt1_node
from AllenConf.hlt1_presets import BGI_ACTIVITY_PRESETS, MONITORING_CONFIG_PRESETS
from AllenCore.generator import generate


def modify_presets():
    MONITORING_CONFIG_PRESETS["pp"]["enable_bgi_full"] = True
    BGI_ACTIVITY_PRESETS["BGIPVsCylAll"]["max_vtx_z"] = 2000
    BGI_ACTIVITY_PRESETS["BGIPVsCylAll"]["min_vtx_z"] = -2000
    BGI_ACTIVITY_PRESETS["BGIPVsCylIR"]["enable"] = True
    BGI_ACTIVITY_PRESETS["Hlt1BGIPseudoPVsIRBeamBeam"]["enable"] = True


hlt1_node = setup_hlt1_node(
    tracking_type=TrackingType.FORWARD_THEN_MATCHING,
    threshold_settings=get_thresholds("pp_reference_run_2024"),
    preset_modifiers=[modify_presets],
)

generate(hlt1_node)
