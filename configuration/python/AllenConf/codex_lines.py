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
    mc_data_provider_t, codex_validator_t, host_data_provider_t,
    data_provider_t, codex_decode_t, codex_clustering_t, codex_coincidence_t,
    codex_passthrough_line_t, codex_coincidence_line_t)
from AllenCore.generator import make_algorithm
from AllenConf.utils import initialize_number_of_events
from PyConf.tonic import configurable
from AllenConf.codex_b import codex_coincidence, codex_prepare_passthrough


@configurable
def make_codex_passthrough_line(name="Hlt1CodexPassthrough",
                                pre_scaler=1,
                                pre_scaler_hash_string=None,
                                post_scaler_hash_string=None):

    number_of_events = initialize_number_of_events()

    prepare_decision = codex_prepare_passthrough()

    return make_algorithm(
        codex_passthrough_line_t,
        name=name,
        pre_scaler=pre_scaler,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_codex_passthrough_decisions_t=prepare_decision.
        dev_codex_passthrough_decisions_t,
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + '_pre',
        post_scaler_hash_string=post_scaler_hash_string or name + '_post')


@configurable
def make_codex_coincidence_line(name="Hlt1CodexCoincidence",
                                pre_scaler=1,
                                pre_scaler_hash_string=None,
                                post_scaler_hash_string=None):

    number_of_events = initialize_number_of_events()
    coincidence_info = codex_coincidence()

    return make_algorithm(
        codex_coincidence_line_t,
        name=name,
        pre_scaler=pre_scaler,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + '_pre',
        post_scaler_hash_string=post_scaler_hash_string or name + '_post',
        dev_codex_double_coincidences_size_t=coincidence_info.
        dev_codex_double_coincidences_size_t,
        dev_codex_triple_coincidences_size_t=coincidence_info.
        dev_codex_triple_coincidences_size_t)
