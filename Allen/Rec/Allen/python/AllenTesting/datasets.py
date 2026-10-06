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
"""Dataset metadata shared by producer and consumer test fixtures."""

from typing import NamedTuple


class DatasetMetadata(NamedTuple):
    test_file_db_key: str
    file_name: str
    input_type: str


TEST_DATASETS = {
    "allen.large_event_passthrough": DatasetMetadata(
        test_file_db_key="upgrade_Sept2022_minbias_0fb_md_mdf",
        file_name="large_event_passthrough.mdf",
        input_type="MDF",
    )
}
