#!/bin/bash
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

rm -fv *.log *.root *.csv *.sqlite *.nsys-rep

# Check current status of the GPUs
nvidia-smi 2>&1
nvidia-smi topo -m 2>&1

# Run the test
numactl -N 0 -m 0 -- lbexec $ALLEN_INSTALL_DIR/python/tests/options/run_hlt1_pp_default.py:main $ALLEN_INSTALL_DIR/python/tests/options/throughput_data_2025.yaml 2>&1 | tee throughput.log

# Run the profile
numactl -N 0 -m 0 -- nsys profile -o allen_report --force-overwrite true lbexec $ALLEN_INSTALL_DIR/python/tests/options/run_hlt1_pp_default.py:main $ALLEN_INSTALL_DIR/python/tests/options/profile_data_2025.yaml 2>&1 | tee profile.log
nsys stats allen_report.nsys-rep --format csv --report cuda_gpu_kern_sum -o allen_report --force-overwrite true

# force 0 return code so the handler runs even for failed jobs
exit 0
