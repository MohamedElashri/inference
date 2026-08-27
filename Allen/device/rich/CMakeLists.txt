###############################################################################
# (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################

file(GLOB rich_decoding       "decoding/src/*cu")
file(GLOB rich_reconstruction "reconstruction/src/*cu")
file(GLOB rich_pid            "pid/src/*cu")

allen_add_device_library(Rich OBJECT
  ${rich_decoding}
  ${rich_reconstruction}
  ${rich_pid}
)

target_link_libraries(Rich PRIVATE Backend HostEventModel EventModel Utils Kalman)

target_include_directories(Rich PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/decoding/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pid/include>
)
