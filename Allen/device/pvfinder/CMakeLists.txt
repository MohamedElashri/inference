allen_add_device_library(PVFinder
    src/VeloFeatureExtraction.cu
    src/PVFinderFCEngine.cu
    src/PVFinderWeightRegistry.cpp
    src/PVFinderTrackAggregation.cu
    src/PVFinderNCWLayout.cu
    src/PVFinderUNet.cu
)

target_link_libraries(PVFinder PRIVATE AllenCommon Backend HostEventModel EventModel Utils PV_beamline)

if(WITH_CUDNN AND TARGET_DEVICE STREQUAL "CUDA")
    target_link_libraries(PVFinder PRIVATE AllenCuDNN)
endif()

target_include_directories(PVFinder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
