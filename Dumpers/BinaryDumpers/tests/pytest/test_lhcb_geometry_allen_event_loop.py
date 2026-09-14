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
import re

from LHCbTesting import LHCbExeTest


class Test(LHCbExeTest):
    command = [
        "lbexec",
        "../options/run_hlt1_pp_default.py:main",
        "../options/lhcb_geometry_allen_event_loop.yaml",
    ]
    timeout = 600

    reference = {"messages_count": {"FATAL": 0, "ERROR": 0, "WARNING": 0}}

    def test_throughput_output(self, stdout: bytes):
        pattern = re.compile(
            r"^.*Execution time: (\d+) ms. Throughput: (\d+\.\d+) events/s"
        )

        throughput = None
        runtime = None

        for line in stdout.decode().split("\n"):
            m = pattern.match(line)
            if m:
                runtime = float(m.group(1))
                throughput = float(m.group(2))
        assert throughput is not None, "could not parse throughput from stdout"
        assert runtime is not None, "could not parse runtime from stdout"
