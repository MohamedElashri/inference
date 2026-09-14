###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
file(GLOB host_output_handler "src/*cpp")

allen_add_host_library(HostOutputHandler OBJECT
  ${host_output_handler}
)
target_link_libraries(HostOutputHandler PUBLIC Boost::headers HostEventModel EventModel Gear AllenCommon Backend mdf HostRoutingBits)

target_include_directories(HostOutputHandler PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
