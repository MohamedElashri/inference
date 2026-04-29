###############################################################################
# (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "COPYING".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenCore.algorithms import (
    codex_validator_t, codex_prepare_passthrough_t, data_provider_t,
    codex_decode_t, codex_clustering_t, codex_coincidence_t)
from AllenCore.generator import make_algorithm
from AllenConf.utils import initialize_number_of_events


def decode_codex():
    number_of_events = initialize_number_of_events()

    codex_banks = make_algorithm(
        data_provider_t, name="codex_banks", bank_type="CODEX")

    codex_decode = make_algorithm(
        codex_decode_t,
        name="codex_decode",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_codex_raw_input_t=codex_banks.dev_raw_banks_t,
        dev_codex_raw_input_offsets_t=codex_banks.dev_raw_offsets_t,
        dev_codex_raw_input_sizes_t=codex_banks.dev_raw_sizes_t,
        dev_codex_raw_input_types_t=codex_banks.dev_raw_types_t,
        host_raw_bank_version_t=codex_banks.host_raw_bank_version_t)

    return codex_decode


def codex_prepare_passthrough():
    number_of_events = initialize_number_of_events()

    codex_banks = make_algorithm(
        data_provider_t, name="codex_banks", bank_type="CODEX")

    prepare_passthrough = make_algorithm(
        codex_prepare_passthrough_t,
        name="prepare_passthrough",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_codex_raw_input_t=codex_banks.dev_raw_banks_t,
        dev_codex_raw_input_offsets_t=codex_banks.dev_raw_offsets_t,
        dev_codex_raw_input_sizes_t=codex_banks.dev_raw_sizes_t,
        dev_codex_raw_input_types_t=codex_banks.dev_raw_types_t,
        host_raw_bank_version_t=codex_banks.host_raw_bank_version_t)

    return prepare_passthrough


def codex_clustering():

    number_of_events = initialize_number_of_events()
    hits_info = decode_codex()

    return make_algorithm(
        codex_clustering_t,
        name="codex_cluster_hits",
        dev_codex_hits_t=hits_info.dev_codex_hits_t,
        dev_codex_hits_size_t=hits_info.dev_codex_all_hits_size_t,
        dev_codex_singlet_offsets_t=hits_info.dev_codex_singlet_offsets_t,
        dev_codex_hits_permutations_t=hits_info.dev_codex_hits_permutations_t,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_codex_num_hits_t=hits_info.host_codex_num_hits_t,
    )


def codex_coincidence():

    number_of_events = initialize_number_of_events()

    clusters_info = codex_clustering()

    return make_algorithm(
        codex_coincidence_t,
        name="codex_find_coincidences",
        dev_codex_clusters_t=clusters_info.dev_codex_clusters_t,
        dev_codex_cluster_size_t=clusters_info.dev_codex_cluster_size_t,
        host_codex_num_clusters_t=clusters_info.host_codex_num_clusters_t,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
    )


def codex_validate_event():

    number_of_events = initialize_number_of_events()

    hits_info = decode_codex()
    clusters_info = codex_clustering()
    coincidence_info = codex_coincidence()

    return make_algorithm(
        codex_validator_t,
        name="codex_validate_event",
        dev_codex_hits_t=hits_info.dev_codex_hits_t,
        dev_codex_hits_size_t=hits_info.dev_codex_all_hits_size_t,
        dev_codex_singlet_offsets_t=hits_info.dev_codex_singlet_offsets_t,
        dev_codex_hits_permutations_t=hits_info.dev_codex_hits_permutations_t,
        dev_codex_clusters_t=clusters_info.dev_codex_clusters_t,
        dev_codex_cluster_size_t=clusters_info.dev_codex_cluster_size_t,
        dev_codex_coincidences_t=coincidence_info.dev_codex_coincidences_t,
        dev_codex_created_coincidences_size_t=coincidence_info.
        dev_codex_created_coincidences_size_t,
        dev_codex_double_coincidences_size_t=coincidence_info.
        dev_codex_double_coincidences_size_t,
        dev_codex_triple_coincidences_size_t=coincidence_info.
        dev_codex_triple_coincidences_size_t,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
    )
