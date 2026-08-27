###############################################################################
# (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
# Apache-2.0 License
###############################################################################

# AllenCuDNN: translational layer between Allen and cuDNN.
# Only built when WITH_CUDNN=ON and TARGET_DEVICE=CUDA.

allen_add_host_library(AllenCuDNN SHARED
  src/CuDNNWeightRegistry.cpp
  src/CuDNNHandle.cpp)

target_include_directories(AllenCuDNN
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Allen/cudnn>)

target_link_libraries(AllenCuDNN
  PUBLIC
    CuDNN::CuDNN          # the imported target from FindCuDNN.cmake
    AllenCommon           # for Logger.h, BackendCommon.h
    Backend)              # for Allen::malloc_host etc.

target_compile_definitions(AllenCuDNN PUBLIC ALLEN_WITH_CUDNN)

# Propagate the cuDNN lib directory so executables that transitively link
# libAllenLib.so -> libAllenCuDNN.so can resolve libcudnn.so.9 at link-time
# and run-time.  Both rpath (run-time) and rpath-link (link-time transitive)
# are needed because libcudnn.so is a private dependency of AllenCuDNN.
get_filename_component(_cudnn_libdir "${CUDNN_LIBRARY}" DIRECTORY)
set_target_properties(AllenCuDNN PROPERTIES
    BUILD_RPATH   "${_cudnn_libdir}"
    INSTALL_RPATH "${_cudnn_libdir}")
target_link_options(AllenCuDNN
    PUBLIC
        "-Wl,-rpath,${_cudnn_libdir}"
        "-Wl,-rpath-link,${_cudnn_libdir}")

# Guard the alias to avoid duplicate target errors on incremental CMake runs
if(NOT TARGET Allen::AllenCuDNN)
  add_library(Allen::AllenCuDNN ALIAS AllenCuDNN)
endif()

install(TARGETS AllenCuDNN
  EXPORT Allen
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Allen/cudnn
  FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")

# NOTE: AllenLib is only available in Gaudi-mode builds, not STANDALONE.
# Algorithm-level CMakeLists must link AllenCuDNN explicitly where needed.
# The PVFinder CMakeLists will add the dependency when WITH_CUDNN is ON.
