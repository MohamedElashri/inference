if(TARGET CuDNN::CuDNN)
  return()
endif()

find_path(CUDNN_INCLUDE_DIR
  NAMES cudnn.h
  HINTS
    ${CUDNN_ROOT}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    /usr/local/cuda/include
    /usr/include
)

find_library(CUDNN_LIBRARY
  NAMES cudnn
  HINTS
    ${CUDNN_ROOT}/lib64
    ${CUDNN_ROOT}/lib
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CuDNN
  REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR)

if(CuDNN_FOUND AND NOT TARGET CuDNN::CuDNN)
  add_library(CuDNN::CuDNN UNKNOWN IMPORTED GLOBAL)
  set_target_properties(CuDNN::CuDNN PROPERTIES
    IMPORTED_LOCATION "${CUDNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")

  # Read version from header (cudnn_version.h or cudnn.h for older installs)
  set(_cudnn_header "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
  if(NOT EXISTS "${_cudnn_header}")
    set(_cudnn_header "${CUDNN_INCLUDE_DIR}/cudnn.h")
  endif()

  file(READ "${_cudnn_header}" _cudnn_header_contents)
  string(REGEX MATCH "define CUDNN_MAJOR[ \t]+([0-9]+)" _ "${_cudnn_header_contents}")
  set(_cudnn_major "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define CUDNN_MINOR[ \t]+([0-9]+)" _ "${_cudnn_header_contents}")
  set(_cudnn_minor "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL[ \t]+([0-9]+)" _ "${_cudnn_header_contents}")
  set(_cudnn_patch "${CMAKE_MATCH_1}")

  set(CUDNN_VERSION "${_cudnn_major}.${_cudnn_minor}.${_cudnn_patch}"
    CACHE STRING "cuDNN version" FORCE)
  message(STATUS "Found cuDNN ${CUDNN_VERSION} at ${CUDNN_LIBRARY}")
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
