###############################################################################
# (c) Copyright 2018-2021 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
add_library(HostEventModel INTERFACE)
install(TARGETS HostEventModel EXPORT Allen)
add_library(Allen::HostEventModel ALIAS HostEventModel)

target_include_directories(HostEventModel INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/routing_bits/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tae/include>)

target_link_libraries(HostEventModel INTERFACE
  $<LINK_ONLY:EventModel>
  $<LINK_ONLY:Gear>
  $<LINK_ONLY:AllenCommon>
  $<LINK_ONLY:Backend>)
