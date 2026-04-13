###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration
#
# Baseline HLT1 benchmark sequence for apples-to-apples comparison against:
#   - HLT1 + PVFinder FC
#   - HLT1 + PVFinder FC + CNN
#
# This intentionally mirrors the same HLT1 setup and reconstruction bindings
# used by the PVFinder benchmark sequences, but does not append any PVFinder
# algorithms.
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.velo_reconstruction import make_pr_velo_tracks


with make_velo_scifi_matches.bind(
        ghost_killer_threshold=0.8), make_pr_velo_tracks.bind(
            missing_modules=[21]):
    benchmark_node = setup_hlt1_node(
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        threshold_settings=get_thresholds(
            "forward_then_matching_and_downstream_with_parkf_tuned_mu5p3_1200kHz"
        ),
        with_ut=True,
        enableDownstream=True,
        with_fullKF=True,
    )

generate(benchmark_node)
