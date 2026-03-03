###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenCore.algorithms import (
    pvfinder_velo_feature_extraction_t,
    pvfinder_track_aggregation_t
)
from AllenConf.velo_reconstruction import run_velo_kalman_filter
from AllenConf.utils import initialize_number_of_events
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable

@configurable
def make_pvfinder_fc(velo_tracks, pv_name=""):
    number_of_events = initialize_number_of_events()
    host_number_of_events = number_of_events["host_number_of_events"]

    host_number_of_reconstructed_velo_tracks = velo_tracks[
        "host_number_of_reconstructed_velo_tracks"]

    velo_states = run_velo_kalman_filter(velo_tracks, pv_name)

    # 1. Feature Extraction (9 features per track)
    pvfinder_feature_extraction = make_algorithm(
        pvfinder_velo_feature_extraction_t,
        name="pvfinder_velo_feature_extraction" + pv_name,
        host_number_of_events_t=host_number_of_events,
        host_number_of_reconstructed_velo_tracks_t=host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_velo_states_view_t=velo_states["dev_velo_kalman_beamline_states_view"]
    )

    # 2. Fused FC+Aggregation — runs the full MLP per-track in registers,
    #    accumulates directly into interval features.  No global latent buffer.
    pvfinder_track_aggregation = make_algorithm(
        pvfinder_track_aggregation_t,
        name="pvfinder_track_aggregation" + pv_name,
        host_number_of_events_t=host_number_of_events,
        host_number_of_reconstructed_velo_tracks_t=host_number_of_reconstructed_velo_tracks,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_pvfinder_track_features_t=pvfinder_feature_extraction.dev_pvfinder_track_features_t
    )

    return {
        "dev_pvfinder_output_histogram": pvfinder_track_aggregation.dev_pvfinder_output_histogram_t,
        "dev_pvfinder_interval_features": pvfinder_track_aggregation.dev_pvfinder_interval_features_t,
        "host_number_of_events": host_number_of_events,
    }
