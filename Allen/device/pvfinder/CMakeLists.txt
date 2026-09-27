###############################################################################
# (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
# The headers alone (peak finding, model file), for tests.
add_library(PVFinderHeaders INTERFACE)
target_link_libraries(PVFinderHeaders INTERFACE Backend EventModel TensorModel)
target_include_directories(PVFinderHeaders INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

allen_add_device_library(PVFinder
    src/PVFinderFCAggregation.cu
    src/PVFinderUNet.cu
    src/PVFinderPeak.cu
    src/PVFinderVertexDump.cpp
)

target_link_libraries(PVFinder PRIVATE PVFinderHeaders AllenCommon Backend HostEventModel EventModel Utils PV_beamline)

if(WITH_CUDNN AND TARGET_DEVICE STREQUAL "CUDA")
    target_link_libraries(PVFinder PRIVATE AllenCuDNN)
endif()

# The network's widths are compile-time constants and must match the model
# file (the algorithms check every tensor's shape in init()). The defaults are
# those of the default model, unet16_lc4_scnone_asym5_final; the bfloat16 path
# supports only these.
set(PVFINDER_UNET_N_FEAT 16 CACHE STRING "PVFinder UNet feature maps (N_FEAT)")
set(PVFINDER_UNET_N_BATCH_CHANNELS 4 CACHE STRING
    "PVFinder latent channels: FC outputs per bin and UNet input channels (latentChannels)")
target_compile_definitions(PVFinder PRIVATE
    PVFINDER_UNET_N_FEAT=${PVFINDER_UNET_N_FEAT}
    PVFINDER_UNET_N_BATCH_CHANNELS=${PVFINDER_UNET_N_BATCH_CHANNELS})
message(STATUS "PVFinder: UNet N_FEAT ${PVFINDER_UNET_N_FEAT}, latent channels ${PVFINDER_UNET_N_BATCH_CHANNELS}")


target_include_directories(PVFinder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
