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
# Gear interface library
add_library(Gear INTERFACE)
target_include_directories(Gear INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gear/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/store/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sequence/include>
  $<INSTALL_INTERFACE:include/Gear/gear>
  $<INSTALL_INTERFACE:include/Gear/store>
  $<INSTALL_INTERFACE:include/Gear/sequence>)
target_link_libraries(Gear INTERFACE Boost::boost)
install(TARGETS Gear EXPORT Allen)

install(DIRECTORY gear/include/ DESTINATION include/Gear/gear)
install(DIRECTORY store/include/ DESTINATION include/Gear/store)
install(DIRECTORY sequence/include/ DESTINATION include/Gear/sequence)

include(GenerateConfiguration)

file(GLOB stream_src "sequence/src/*cpp")
file(GLOB gear_src "gear/src/*cpp")

allen_add_host_library(Stream OBJECT ${stream_src} ${gear_src})

target_link_libraries(Stream
  PRIVATE
    HostEventModel
    EventModel
    Backend
    Calo
    Combiners
    DeviceValidators
    Examples
    Kalman
    Lumi
    Muon
    PV_beamline
    Plume
    SciFi
    UT
    Downstream
    Validators
    Velo
    VertexFitter
    AllenCommon
    Gear
    track_matching
    MuonCommon
    Rich
    TTrack
  PUBLIC
    Utils
    Selections
    SelectionsHeaders)

if(STANDALONE)
  add_dependencies(Stream checkout_lhcb checkout_gaudi)
endif()

foreach(sequence ${BUILD_SEQUENCES})
  generate_sequence(${sequence})
endforeach()
