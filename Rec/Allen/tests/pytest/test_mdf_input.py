###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenTesting.preprocessors import preprocessor as AllenPreprocessor
from LHCbTesting import LHCbExeTest


class Test(LHCbExeTest):
    command = [
        "lbexec",
        "../options/run_hlt1_pp_matching.py:main",
        "../options/mdf_input.yaml",
    ]
    reference = "../refs/allen_event_loop.yaml"
    timeout = 600

    preprocessor = AllenPreprocessor
