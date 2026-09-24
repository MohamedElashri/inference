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

from AllenCore.configuration_options import is_allen_standalone

from AllenConf.calo_reconstruction import decode_calo, make_ecal_clusters
from AllenConf.muon_reconstruction import decode_muon
from AllenConf.persistency import (
    line_names,
    make_dec_reporter,
    make_gather_selections,
)
from AllenConf.primary_vertex_reconstruction import make_pvs
from AllenConf.scifi_reconstruction import decode_scifi, make_seeding_XZ_tracks
from AllenConf.utils import initialize_number_of_events
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks


def rate_validation(lines, groups={}, name="rate_validator"):
    if is_allen_standalone():
        return None

    number_of_events = initialize_number_of_events()
    dec_reporter = make_dec_reporter(lines)
    gather_selections = make_gather_selections(lines)

    # Check for configuration error of physics and technical lines. The names
    # must follow the gather_selections ordering, as that is what indexes the
    # decisions in the HLT1 dec reports.
    names_of_active_lines = line_names(gather_selections)
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

    # Native Gaudi multi-event algorithm, replacing the Allen host_rate_validator
    # whose RateChecker has been removed.
    from PyConf.Algorithms import RateValidator

    return RateValidator(
        name=name,
        allen_number_of_events=number_of_events["host_number_of_events"],
        allen_names_of_lines=gather_selections.host_names_of_active_lines_t,
        allen_number_of_active_lines=gather_selections.host_number_of_active_lines_t,
        allen_dec_reports=dec_reporter.host_dec_reports_t,
        Hlt1LineNames=names_of_active_lines,
        json_string=json_string,
    )


def rate_validation_nodes(lines, groups={}, name="rate_validator"):
    """Rate validator node(s) to append to a sequence.

    In standalone mode the Gaudi rate validator is not available, so an empty
    list is returned and callers can unconditionally splice the result into a
    CompositeNode children list.
    """
    node = rate_validation(lines, groups, name)
    return [node] if node is not None else []


def data_quality_validation_long(
    long_tracks, long_track_particles, name="data_quality_validator"
):
    from PyConf.Algorithms import ConvertAllenLongTracksDQ, DataQualityValidatorLong

    info = ConvertAllenLongTracksDQ(
        name=name + "_convert",
        dev_particle_container=long_track_particles["dev_multi_event_basic_particles"],
    ).DQLongTrackInfo

    return DataQualityValidatorLong(name=name, DQLongTrackInfo=info)


def data_quality_validation_velo(long_tracks, name="data_quality_validator"):
    from PyConf.Algorithms import DataQualityValidatorVelo

    number_of_events = initialize_number_of_events()

    velo_tracks = long_tracks["velo_tracks"]
    velo_kalman_filter = long_tracks["velo_kalman_filter"]

    return DataQualityValidatorVelo(
        name=name,
        dev_offsets_velo_tracks=velo_tracks["dev_offsets_all_velo_tracks"],
        dev_offsets_all_velo_tracks=velo_tracks["dev_offsets_all_velo_tracks"],
        dev_offsets_velo_track_hit_number=velo_tracks[
            "dev_offsets_velo_track_hit_number"
        ],
        dev_velo_track_hits=velo_tracks["dev_velo_track_hits"],
        dev_velo_kalman_states=velo_kalman_filter["dev_velo_kalman_endvelo_states"],
        host_number_of_events=number_of_events["host_number_of_events"],
    )


def data_quality_validation_pv(long_tracks, name="data_quality_validator"):
    from PyConf.Algorithms import DataQualityValidatorPV

    number_of_events = initialize_number_of_events()

    velo_tracks = long_tracks["velo_tracks"]
    pvs = make_pvs(velo_tracks)

    return DataQualityValidatorPV(
        name=name,
        dev_multi_fit_vertices=pvs["dev_multi_final_vertices"],
        dev_number_of_multi_fit_vertices=pvs["dev_number_of_multi_final_vertices"],
        host_number_of_events=number_of_events["host_number_of_events"],
    )


def data_quality_validation_occupancy(name="data_quality_validator"):
    from PyConf.Algorithms import DataQualityValidatorOccupancy

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

    return DataQualityValidatorOccupancy(
        name=name,
        dev_station_ocurrences_offset=decoded_muon["dev_station_ocurrences_offset"],
        dev_velo_offsets_estimated_input_size=decoded_velo[
            "dev_offsets_estimated_input_size"
        ],
        dev_offsets_velo_tracks=velo_tracks["dev_offsets_all_velo_tracks"],
        dev_scifi_hit_offsets=decoded_scifi["dev_scifi_hit_offsets"],
        dev_scifi_seedsXZ=scifi_xz_seeds["seed_xz_number_of_tracks"],
        dev_ecal_clusters_offsets=ecal_clusters["dev_ecal_cluster_offsets"],
        host_number_of_events=number_of_events["host_number_of_events"],
    )
