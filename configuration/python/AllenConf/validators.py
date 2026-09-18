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
import json

from AllenCore.algorithms import (
    data_quality_validator_long_t,
    data_quality_validator_occupancy_t,
    data_quality_validator_pv_t,
    data_quality_validator_velo_t,
    host_data_provider_t,
    host_rate_validator_t,
    mc_data_provider_t,
    vertexing_validator_t,
)
from AllenCore.generator import make_algorithm

from AllenConf.calo_reconstruction import decode_calo, make_ecal_clusters
from AllenConf.muon_reconstruction import decode_muon
from AllenConf.odin import decode_odin
from AllenConf.persistency import make_dec_reporter, make_gather_selections
from AllenConf.primary_vertex_reconstruction import make_pvs
from AllenConf.scifi_reconstruction import decode_scifi, make_seeding_XZ_tracks
from AllenConf.utils import initialize_number_of_events
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks


def mc_data_provider():
    host_mc_particle_banks = make_algorithm(
        host_data_provider_t, name="host_mc_particle_banks", bank_type="tracks"
    )
    host_mc_pv_banks = make_algorithm(
        host_data_provider_t, name="host_mc_pv_banks", bank_type="PVs"
    )
    number_of_events = initialize_number_of_events()
    return make_algorithm(
        mc_data_provider_t,
        name="mc_data_provider",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_mc_particle_banks_t=host_mc_particle_banks.host_raw_banks_t,
        host_mc_particle_offsets_t=host_mc_particle_banks.host_raw_offsets_t,
        host_mc_particle_sizes_t=host_mc_particle_banks.host_raw_sizes_t,
        host_mc_pv_banks_t=host_mc_pv_banks.host_raw_banks_t,
        host_mc_pv_offsets_t=host_mc_pv_banks.host_raw_offsets_t,
        host_mc_pv_sizes_t=host_mc_pv_banks.host_raw_sizes_t,
        host_bank_version_t=host_mc_particle_banks.host_raw_bank_version_t,
    )


def rate_validation(lines, groups={}, name="rate_validator"):
    number_of_events = initialize_number_of_events()
    dec_reporter = make_dec_reporter(lines)
    gather_selections = make_gather_selections(lines)

    # Check for configuration error of physics and technical lines
    names_of_active_lines = [line.name for line in lines]
    grouped_line_names = {
        key: [line.name for line in lines] for key, lines in groups.items()
    }
    for key, list_of_names in grouped_line_names.items():
        for line_name in list_of_names:
            if line_name not in names_of_active_lines:
                raise ValueError(
                    f"rate_validator: {key} line with name <{line_name}> is not in the full list of HLT1 lines!"
                )

    # Prefix with json: so that the string don't get parse outside of algorithm
    json_payload = json.dumps(grouped_line_names)
    json_string = f"json:{json_payload}"

    return make_algorithm(
        host_rate_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_names_of_lines_t=gather_selections.host_names_of_active_lines_t,
        host_number_of_active_lines_t=gather_selections.host_number_of_active_lines_t,
        host_dec_reports_t=dec_reporter.host_dec_reports_t,
        json_string=json_string,
    )


def data_quality_validation_long(
    long_tracks, long_track_particles, name="data_quality_validator"
):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        data_quality_validator_long_t,
        name=name,
        enable_tupling=True,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_particle_container_t=long_track_particles[
            "dev_multi_event_basic_particles"
        ],
        dev_offsets_long_tracks_t=long_tracks["dev_offsets_long_tracks"],
        host_number_of_reconstructed_long_tracks_t=long_tracks[
            "host_number_of_reconstructed_scifi_tracks"
        ],
    )


def data_quality_validation_velo(long_tracks, name="data_quality_validator"):
    number_of_events = initialize_number_of_events()

    velo_tracks = long_tracks["velo_tracks"]
    velo_kalman_filter = long_tracks["velo_kalman_filter"]

    return make_algorithm(
        data_quality_validator_velo_t,
        name=name,
        enable_tupling=True,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_offsets_velo_tracks_t=velo_tracks["dev_offsets_all_velo_tracks"],
        dev_offsets_all_velo_tracks_t=velo_tracks["dev_offsets_all_velo_tracks"],
        dev_offsets_velo_track_hit_number_t=velo_tracks[
            "dev_offsets_velo_track_hit_number"
        ],
        dev_velo_track_hits_t=velo_tracks["dev_velo_track_hits"],
        dev_velo_kalman_states_t=velo_kalman_filter["dev_velo_kalman_endvelo_states"],
    )


def data_quality_validation_pv(long_tracks, name="data_quality_validator"):
    number_of_events = initialize_number_of_events()

    velo_tracks = long_tracks["velo_tracks"]
    pvs = make_pvs(velo_tracks)

    return make_algorithm(
        data_quality_validator_pv_t,
        name=name,
        enable_tupling=True,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_multi_fit_vertices_t=pvs["dev_multi_final_vertices"],
        dev_number_of_multi_fit_vertices_t=pvs["dev_number_of_multi_final_vertices"],
    )


def data_quality_validation_occupancy(name="data_quality_validator"):
    number_of_events = initialize_number_of_events()

    decoded_scifi = decode_scifi()
    scifi_xz_seeds = make_seeding_XZ_tracks(decoded_scifi)
    decoded_muon = decode_muon()
    decoded_calo = decode_calo()
    ecal_clusters = make_ecal_clusters(
        decoded_calo, calo_find_clusters_name="calo_find_clusters_dq_validator"
    )

    decoded_velo = decode_velo()
    velo_tracks = make_velo_tracks(decoded_velo)

    return make_algorithm(
        data_quality_validator_occupancy_t,
        name=name,
        enable_tupling=True,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_station_ocurrences_offset_t=decoded_muon["dev_station_ocurrences_offset"],
        dev_velo_offsets_estimated_input_size_t=decoded_velo[
            "dev_offsets_estimated_input_size"
        ],
        dev_offsets_velo_tracks_t=velo_tracks["dev_offsets_all_velo_tracks"],
        dev_scifi_hit_offsets_t=decoded_scifi["dev_scifi_hit_offsets"],
        dev_scifi_seedsXZ_t=scifi_xz_seeds["seed_xz_number_of_tracks"],
        dev_ecal_clusters_offsets_t=ecal_clusters["dev_ecal_cluster_offsets"],
    )


def VertexingValidator(multi_body_svs, name="vertexing_validator"):
    mc_events = mc_data_provider()
    number_of_events = initialize_number_of_events()
    odin = decode_odin()

    return make_algorithm(
        vertexing_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_mc_events_t=mc_events.host_mc_events_t,
        dev_odin_data_t=odin["dev_odin_data"],
        host_number_of_vertices_t=multi_body_svs["host_number_of_svs"],
        dev_offset_vertices_t=multi_body_svs["dev_sv_offsets"],
        dev_multi_event_composites_view_t=multi_body_svs["dev_multi_event_composites"],
    )
