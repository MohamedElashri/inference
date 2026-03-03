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
file(GLOB ttrack_test "test/src/*.cu")
file(GLOB ttrack_prepare_tracks "prepare_tracks/src/*.cu")
file(GLOB ttrack_make_vertices "make_vertices/src/*.cu")

allen_add_device_library(TTrack STATIC
  ${ttrack_test}
  ${ttrack_prepare_tracks}
  ${ttrack_make_vertices}
)

target_link_libraries(TTrack PRIVATE Backend HostEventModel EventModel Utils Kalman TrackChecking)

target_include_directories(TTrack PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prepare_tracks/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/make_vertices/include>)
  
  target_include_directories(WrapperInterface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prepare_tracks/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/make_vertices/include>)



