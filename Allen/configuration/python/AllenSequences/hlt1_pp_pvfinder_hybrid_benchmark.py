###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
# HLT1 with PVFinder replacing the beamline PV finder inside PVFinder's z range
# (-100 < z < 300 mm), the beamline PV finder's seeds kept outside it
# (pvfinder_merge_seeds), one association and fit for all. Otherwise as
# hlt1_pp_pvfinder_replace_benchmark: every consumer of the
# primary vertices (track fits, lines, downstream tracking, lumi counters) gets
# PVFinder's vertices
# (pvfinder_peak seeds, then the beamline PV association and fit), and the
# beamline peak finding no longer runs. Measures what replacing the current PV
# finder costs today, with the vertices outside PVFinder's range still found.
import AllenConf.downstream_reconstruction as downstream_reconstruction
import AllenConf.hlt1_reconstruction as hlt1_reconstruction
import AllenConf.lumi_reconstruction as lumi_reconstruction
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.HLT1 import setup_hlt1_node
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
from AllenConf.pvfinder_pv_reconstruction import make_pvfinder_pvs
from AllenConf.pvfinder_unet_reconstruction import make_pvfinder_unet
from AllenConf.velo_reconstruction import make_pr_velo_tracks
from AllenCore.generator import generate


def make_pvs_from_pvfinder(velo_tracks, velo_open=False, **kwargs):
    """Drop-in for make_pvs (closed VELO only)."""
    assert not velo_open, "PVFinder is trained for the closed VELO"
    kde = make_pvfinder_unet(make_pvfinder_fc(velo_tracks))["dev_pvfinder_kde_output"]
    return make_pvfinder_pvs(velo_tracks, kde, fill_outside_range=True)


# Each of these modules builds its vertices with the make_pvs it imported.
for module in (hlt1_reconstruction, lumi_reconstruction, downstream_reconstruction):
    module.make_pvs = make_pvs_from_pvfinder

with (
    make_velo_scifi_matches.bind(ghost_killer_threshold=0.8),
    make_pr_velo_tracks.bind(missing_modules=[21]),
):
    hlt1_node_dict = setup_hlt1_node(
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        threshold_settings=get_thresholds(
            "forward_then_matching_and_downstream_with_parkf_tuned_mu5p3_1200kHz"
        ),
        with_ut=True,
        enableDownstream=True,
        with_fullKF=True,
    )

generate(hlt1_node_dict["control_flow_node"])
