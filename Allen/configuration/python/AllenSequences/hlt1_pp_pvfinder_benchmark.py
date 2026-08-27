###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate
from AllenConf.enum_types import TrackingType
from AllenConf.get_thresholds import get_thresholds
from AllenConf.matching_reconstruction import make_velo_scifi_matches
from AllenConf.velo_reconstruction import make_pr_velo_tracks
from AllenConf.pvfinder_fc_reconstruction import make_pvfinder_fc
from PyConf.control_flow import NodeLogic, CompositeNode

def hook_pvfinder_to_hlt1():
    from AllenConf.HLT1 import setup_hlt1_node
    hlt1_node_dict = setup_hlt1_node(
        tracking_type=TrackingType.FORWARD_THEN_MATCHING,
        threshold_settings=get_thresholds(
            f"forward_then_matching_and_downstream_with_parkf_tuned_mu5p3_1200kHz"
        ),
        with_ut=True,
        enableDownstream=True,
        with_fullKF=True,
    )
    
    # Rather than creating a NEW wrapper Node, inject the new algorithm directly into the top node
    hlt1_graph = hlt1_node_dict['control_flow_node']
    reco = hlt1_node_dict['reconstruction']
    
    pvfinder_fc_output = make_pvfinder_fc(reco['velo_tracks'])
    aggregation_producer = pvfinder_fc_output["dev_pvfinder_output_histogram"].producer
    
    hlt1_graph.children = tuple(list(hlt1_graph.children) + [aggregation_producer])
    return hlt1_graph

with make_velo_scifi_matches.bind(
        ghost_killer_threshold=0.8), make_pr_velo_tracks.bind(
            missing_modules=[21]):
    benchmark_node = hook_pvfinder_to_hlt1()

generate(benchmark_node)
