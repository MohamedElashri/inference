###############################################################################
# (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
import pytest
from AllenTesting.datasets import TEST_DATASETS
from AllenTesting.preprocessors import preprocessor as AllenPreprocessor
from LHCbTesting import LHCbExeTest

large_event_dataset = TEST_DATASETS["allen.large_event_passthrough"]


@pytest.mark.ctest_fixture_setup("allen.large_event_passthrough")
@pytest.mark.shared_cwd("Allen")
class Test(LHCbExeTest):
    command = [
        "python",
        "../../../../Dumpers/BinaryDumpers/options/allen.py",
        "--test-file-db-key",
        large_event_dataset.test_file_db_key,
        "--sequence",
        "${ALLEN_INSTALL_DIR}/constants/hlt1_pp_matching.json",
        "--events-per-slice",
        "500",
        "-m",
        "0",
        "-s",
        "1",
        "-t",
        "1",
        "-n",
        "2",
        "--output-file",
        large_event_dataset.file_name,
    ]
    reference = "../refs/allen_large_event_passthrough.yaml"
    timeout = 600

    preprocessor = AllenPreprocessor
