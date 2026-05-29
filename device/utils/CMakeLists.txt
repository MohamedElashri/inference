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
file(GLOB prefix_sum "prefix_sum/src/*cu")
file(GLOB sorting "sorting/src/*cu")
file(GLOB mva_models "mva_models/src/*cu")
file(GLOB reduction "reduction/src/*cu")

allen_add_device_library(Utils OBJECT
  ${prefix_sum}
  ${sorting}
  ${mva_models}
  ${reduction}
  ${track_trajectory}
)

target_link_libraries(Utils
    PUBLIC
      Backend
      EventModel
      HostEventModel)

target_include_directories(Utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/binary_search/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sorting/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/float_operations/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/memory/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/scaler/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mva_models/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/linear_algebra/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/warp/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prefix_sum/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/scan_reduce/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/reduction/include>)

allen_add_executable(prefix_sum_test prefix_sum/test/prefix_sum_test.cu)
target_link_libraries(prefix_sum_test PRIVATE AllenLib Utils AllenRuntime)

allen_add_executable(reduction_test reduction/test/reduction_test.cu)
target_link_libraries(reduction_test PRIVATE AllenLib Utils AllenRuntime)
