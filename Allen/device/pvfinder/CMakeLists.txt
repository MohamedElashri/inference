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

if(PVFINDER_UNET_N_FEAT)
    target_compile_definitions(PVFinder PRIVATE PVFINDER_UNET_N_FEAT=${PVFINDER_UNET_N_FEAT})
    message(STATUS "PVFinder: UNet N_FEAT overridden to ${PVFINDER_UNET_N_FEAT}")
endif()

if(PVFINDER_FC_HIDDEN_DIM)
    target_compile_definitions(PVFinder PRIVATE PVFINDER_FC_HIDDEN_DIM=${PVFINDER_FC_HIDDEN_DIM})
    message(STATUS "PVFinder: FC hidden dimension overridden to ${PVFINDER_FC_HIDDEN_DIM}")
endif()

if(PVFINDER_FC_HIDDEN_LAYERS)
    target_compile_definitions(PVFinder PRIVATE PVFINDER_FC_HIDDEN_LAYERS=${PVFINDER_FC_HIDDEN_LAYERS})
    message(STATUS "PVFinder: FC hidden layer count overridden to ${PVFINDER_FC_HIDDEN_LAYERS}")
endif()

if(PVFINDER_FC_OUTPUT_DIM)
    target_compile_definitions(PVFinder PRIVATE PVFINDER_FC_OUTPUT_DIM=${PVFINDER_FC_OUTPUT_DIM})
    message(STATUS "PVFinder: FC output dimension overridden to ${PVFINDER_FC_OUTPUT_DIM}")
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
