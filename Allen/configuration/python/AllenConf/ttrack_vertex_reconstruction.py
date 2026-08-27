###############################################################################
# (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################

from AllenCore.algorithms import (
    create_ttrack_particle_views_t,
    create_ttrack_vertex_views_t,
    create_ttrack_views_t,
    extrapolate_ttracks_t,
    filter_ttracks_t,
    make_ttrack_vertices_t,
)
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable

from AllenConf.muon_reconstruction import decode_muon, fake_muon_id, make_is_muon
from AllenConf.utils import initialize_number_of_events


@configurable
def make_ttrack_vertices(
    scifi_tracks,
    velo_scifi_matches,
    downstream_tracks,
    with_kalman_filter=True,
    with_muon=True,
):
    number_of_events = initialize_number_of_events()

    filtered_ttracks = make_algorithm(
        filter_ttracks_t,
        name="filter_ttracks_{hash}",
        # Basics
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        # Velo-SciFi matches
        dev_matched_is_scifi_track_used_t=velo_scifi_matches[
            "dev_matched_is_scifi_track_used"
        ],
        # Downstream tracks
        dev_multi_event_downstream_tracks_view_t=downstream_tracks[
            "dev_multi_event_downstream_tracks_view"
        ],
        # SciFi tracks
        host_number_of_scifi_tracks_t=scifi_tracks[
            "host_number_of_reconstructed_seeding_tracks"
        ],
        dev_seeding_tracks_t=scifi_tracks["seed_tracks"],
        dev_seeding_qop_t=scifi_tracks["dev_seeding_qop"],
        dev_seeding_states_t=scifi_tracks["dev_seeding_states"],
        dev_offsets_seeding_tracks_t=scifi_tracks["dev_offsets_scifi_seeds"],
        dev_scifi_track_view_t=scifi_tracks["dev_scifi_track_view"],
        with_kalman_filter=with_kalman_filter,
    )

    ttrack_views = make_algorithm(
        create_ttrack_views_t,
        name="create_ttrack_views_{hash}",
        # Basics
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        # SciFi tracks
        dev_scifi_track_view_t=scifi_tracks["dev_scifi_track_view"],
        # Prefiltered tracks
        host_number_of_filtered_scifi_tracks_t=filtered_ttracks.host_number_of_filtered_scifi_tracks_t,
        dev_offsets_filtered_seeding_tracks_t=filtered_ttracks.dev_offsets_filtered_seeding_tracks_t,
        dev_filtered_seeding_qop_t=filtered_ttracks.dev_filtered_seeding_qop_t,
        dev_filtered_seeding_states_t=filtered_ttracks.dev_filtered_seeding_states_t,
        dev_filtered_state_indexes_t=filtered_ttracks.dev_filtered_state_indexes_t,
    )

    # Lepton ID
    if with_muon:
        muonID = make_is_muon(
            decoded_muon=decode_muon(),
            host_number_of_tracks=filtered_ttracks.host_number_of_filtered_scifi_tracks_t,
            dev_multi_event_tracks_ptr=ttrack_views.dev_multi_event_ttracks_view_ptr_t,
            dev_velo_states=ttrack_views.dev_track_kalman_states_view_t,
            dev_scifi_states=filtered_ttracks.dev_muon_states_t,
            is_muon_name="is_muon_{hash}",
        )
    else:
        muonID = fake_muon_id(
            host_number_of_tracks=filtered_ttracks.host_number_of_filtered_scifi_tracks_t
        )

    ttrack_particle_views = make_algorithm(
        create_ttrack_particle_views_t,
        name="create_ttrack_particle_views_{hash}",
        # Basics
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        # Prefiltered tracks
        host_number_of_filtered_scifi_tracks_t=filtered_ttracks.host_number_of_filtered_scifi_tracks_t,
        dev_offsets_filtered_seeding_tracks_t=filtered_ttracks.dev_offsets_filtered_seeding_tracks_t,
        # Track views
        dev_ttrack_view_t=ttrack_views.dev_ttrack_view_t,
        dev_ttracks_view_t=ttrack_views.dev_ttracks_view_t,
        dev_track_kalman_states_view_t=ttrack_views.dev_track_kalman_states_view_t,
        # Muon ID
        dev_lepton_id_t=muonID["dev_lepton_id"],
    )

    extrapolated_tracks = make_algorithm(
        extrapolate_ttracks_t,
        name="extrapolate_ttracks_{hash}",
        # Basics
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        # SciFi tracks
        host_number_of_scifi_tracks_t=filtered_ttracks.host_number_of_filtered_scifi_tracks_t,
        dev_offsets_scifi_tracks_t=filtered_ttracks.dev_offsets_filtered_seeding_tracks_t,
        dev_scifi_track_qop_t=filtered_ttracks.dev_filtered_seeding_qop_t,
        dev_scifi_track_states_t=filtered_ttracks.dev_filtered_seeding_states_t,
    )

    ttrack_vertices = make_algorithm(
        make_ttrack_vertices_t,
        name="make_ttrack_vertices_{hash}",
        # Basics
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        # SciFi tracks
        dev_offsets_seeding_tracks_t=filtered_ttracks.dev_offsets_filtered_seeding_tracks_t,
        dev_filtered_ttracks_qop_t=filtered_ttracks.dev_filtered_seeding_qop_t,
        dev_filtered_ttracks_states_t=extrapolated_tracks.dev_extrapolated_ttrack_states_t,
    )

    vertex_views = make_algorithm(
        create_ttrack_vertex_views_t,
        name="create_ttrack_vertex_views_{hash}",
        # Basics
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        # Basic particles
        dev_multi_event_basic_particles_view_t=ttrack_particle_views.dev_multi_event_basic_particles_view_t,
        # TTrack vertices
        dev_tt_vertices_t=ttrack_vertices.dev_tt_vertices_t,
        dev_offsets_tt_vertices_t=ttrack_vertices.dev_offsets_tt_vertices_t,
        host_number_of_tt_vertices_t=ttrack_vertices.host_number_of_tt_vertices_t,
    )

    return {
        # Filtered tracks
        "host_number_of_filtered_scifi_tracks": filtered_ttracks.host_number_of_filtered_scifi_tracks_t,
        "dev_offsets_filtered_seeding_tracks": filtered_ttracks.dev_offsets_filtered_seeding_tracks_t,
        "dev_filtered_seeding_qop": filtered_ttracks.dev_filtered_seeding_qop_t,
        "dev_filtered_seeding_states": filtered_ttracks.dev_filtered_seeding_states_t,
        "dev_filtered_state_indexes": filtered_ttracks.dev_filtered_state_indexes_t,
        # Extrapolated tracks
        "dev_extrapolated_ttrack_states": extrapolated_tracks.dev_extrapolated_ttrack_states_t,
        # Views
        "dev_track_kalman_states": ttrack_views.dev_track_kalman_states_t,
        "dev_track_kalman_states_view": ttrack_views.dev_track_kalman_states_view_t,
        "dev_ttrack_view": ttrack_views.dev_ttrack_view_t,
        "dev_ttracks_view": ttrack_views.dev_ttracks_view_t,
        "dev_track_particle_view": ttrack_particle_views.dev_track_particle_view_t,
        "dev_track_particles_view": ttrack_particle_views.dev_track_particles_view_t,
        "dev_multi_event_basic_particles_view": ttrack_particle_views.dev_multi_event_basic_particles_view_t,
        "dev_multi_event_basic_particles_view_ptr": ttrack_particle_views.dev_multi_event_basic_particles_view_ptr_t,
        # Vertices
        "dev_tt_vertices": ttrack_vertices.dev_tt_vertices_t,
        "dev_offsets_tt_vertices": ttrack_vertices.dev_offsets_tt_vertices_t,
        "host_number_of_tt_vertices": ttrack_vertices.host_number_of_tt_vertices_t,
        # Views
        "dev_sv_fit_results": vertex_views.dev_sv_fit_results_t,
        "dev_sv_fit_results_view": vertex_views.dev_sv_fit_results_view_t,
        "dev_two_track_sv_track_pointers": vertex_views.dev_two_track_sv_track_pointers_t,
        "dev_two_track_composite_view": vertex_views.dev_two_track_composite_view_t,
        "dev_two_track_composites_view": vertex_views.dev_two_track_composites_view_t,
        "dev_multi_event_composites_view": vertex_views.dev_multi_event_composites_view_t,
        "dev_multi_event_composites_ptr": vertex_views.dev_multi_event_composites_ptr_t,
    }
