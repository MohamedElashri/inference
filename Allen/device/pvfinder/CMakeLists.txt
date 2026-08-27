allen_add_device_library(PVFinder
    src/VeloFeatureExtraction.cu
    src/PVFinderWeightRegistry.cpp
    src/PVFinderFCAggregation.cu
    src/PVFinderUNet.cu
)

target_link_libraries(PVFinder PRIVATE AllenCommon Backend HostEventModel EventModel Utils PV_beamline)

if(WITH_CUDNN AND TARGET_DEVICE STREQUAL "CUDA")
    target_link_libraries(PVFinder PRIVATE AllenCuDNN)
    target_compile_definitions(PVFinder PRIVATE ALLEN_CUDNN_BACKEND_CUDA)
endif()

if(PVFINDER_UNET_N_FEAT)
    target_compile_definitions(PVFinder PRIVATE PVFINDER_UNET_N_FEAT=${PVFINDER_UNET_N_FEAT})
    message(STATUS "PVFinder: UNet N_FEAT overridden to ${PVFINDER_UNET_N_FEAT}")
endif()

if(PVFINDER_UNET_N_BATCH_CHANNELS)
    target_compile_definitions(PVFinder PRIVATE PVFINDER_UNET_N_BATCH_CHANNELS=${PVFINDER_UNET_N_BATCH_CHANNELS})
    message(STATUS "PVFinder: UNet N_BATCH_CHANNELS (latentChannels) overridden to ${PVFINDER_UNET_N_BATCH_CHANNELS}")
endif()

if(WITH_CUBLAS AND TARGET_DEVICE STREQUAL "CUDA")
    # cuBLAS ships with every CUDA toolkit — no extra find_package required.
    target_link_libraries(PVFinder PRIVATE CUDA::cublas)
    target_compile_definitions(PVFinder PRIVATE ALLEN_WITH_CUBLAS)
    message(STATUS "PVFinder: cuBLAS GEMM enabled for L6A layer")
endif()

target_include_directories(PVFinder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
