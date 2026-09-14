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
from Allen.config import AllenTestOptions, run_allen
from AllenCore.generator import allen_runtime_options


def main(options: AllenTestOptions):
    with allen_runtime_options.bind(filename="allen_odqv_pytest.root"):
        return run_allen(options, sequence="hlt1_pp_odqv")
