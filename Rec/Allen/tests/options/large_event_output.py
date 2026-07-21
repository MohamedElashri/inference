###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenTesting.datasets import TEST_DATASETS
from PyConf.application import ApplicationOptions

large_event_dataset = TEST_DATASETS["allen.large_event_passthrough"]

options = ApplicationOptions(_enabled=False)

# Start from the same dataset used to produce the passthrough file. This keeps
# its simulation, geometry, and conditions metadata consistent with the input.
options.set_input_and_conds_from_testfiledb(large_event_dataset.test_file_db_key)

# Read the MDF produced by the test fixture instead of the original dataset
# files.
options.input_files = [large_event_dataset.file_name]
options.input_type = large_event_dataset.input_type
