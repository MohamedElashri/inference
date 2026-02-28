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
from AllenCore.algorithms import (data_provider_t, rich_decoding_t,
                                  rich_make_pixels_t)
from AllenConf.utils import initialize_number_of_events
from AllenCore.generator import make_algorithm

RICH_1 = 1
RICH_2 = 2

VALID_RICHS = [RICH_1, RICH_2]


def decode_rich(rich=RICH_1):
    if rich not in VALID_RICHS:
        raise ValueError(f"rich must be one of {VALID_RICHS}")

    number_of_events = initialize_number_of_events()

    rich_banks = make_algorithm(
        data_provider_t, name=f"rich{rich}_banks", bank_type=f"Rich{rich}")

    rich_decoding = make_algorithm(
        rich_decoding_t,
        name=f"rich{rich}_decoding",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_raw_bank_version_t=rich_banks.host_raw_bank_version_t,
        dev_rich_raw_input_t=rich_banks.dev_raw_banks_t,
        dev_rich_raw_input_offsets_t=rich_banks.dev_raw_offsets_t,
        dev_rich_raw_input_sizes_t=rich_banks.dev_raw_sizes_t,
        dev_rich_raw_input_types_t=rich_banks.dev_raw_types_t)

    return {
        "dev_smart_ids":
        rich_decoding.dev_smart_ids_t,
        "dev_rich_hit_offsets":
        rich_decoding.dev_rich_hit_offsets_t,
        "host_rich_total_number_of_hits":
        rich_decoding.host_rich_total_number_of_hits_t
    }


def make_pixels(decoded_rich=None, rich=RICH_1):
    if rich not in VALID_RICHS:
        raise ValueError(f"rich must be one of {VALID_RICHS}")

    if decoded_rich is None:
        decoded_rich = decode_rich(rich)

    number_of_hits = decoded_rich["host_rich_total_number_of_hits"]
    smart_ids = decoded_rich["dev_smart_ids"]
    hit_offsets = decoded_rich["dev_rich_hit_offsets"]

    rich_pixels = make_algorithm(
        rich_make_pixels_t,
        name=f"rich{rich}_make_pixels",
        host_rich_total_number_of_hits_t=number_of_hits,
        dev_smart_ids_t=smart_ids,
        dev_rich_hit_offsets_t=hit_offsets,
        current_rich=rich)

    return {"dev_rich_pixels": rich_pixels.dev_rich_pixels_t}
