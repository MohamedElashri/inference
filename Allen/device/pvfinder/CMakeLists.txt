allen_add_device_library(PVFinder
    src/VeloFeatureExtraction.cu
    src/PVFinderWeightRegistry.cpp
    src/PVFinderFCAggregation.cu
    src/PVFinderNCWLayout.cu
    src/PVFinderUNet.cu
)

target_link_libraries(PVFinder PRIVATE AllenCommon Backend HostEventModel EventModel Utils PV_beamline)

if(WITH_CUDNN AND TARGET_DEVICE STREQUAL "CUDA")
    target_link_libraries(PVFinder PRIVATE AllenCuDNN)
    target_compile_definitions(PVFinder PRIVATE ALLEN_CUDNN_BACKEND_CUDA)
endif()

target_include_directories(PVFinder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
