###############################################################################
# (c) Copyright 2000-2026 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################

####
# Some files are links to cvmfs (not necessarily available in standalone mode)
# If necessary, download those files as part of the build configuration
####


# Configure time: Figure out if any files are not available locally (e.g. on cvmfs)

message(STATUS "Searching for magfield files")
set(GEOMETRY_DIR "${CMAKE_SOURCE_DIR}/input/allen_geometries")
set(LOCAL_MAGFIELDS_DIR "${CMAKE_BINARY_DIR}/local_magfields")

# Find all the unique mag fields here (currently all duplicates)
file(GLOB MAGFIELD_LINKS
     "${GEOMETRY_DIR}/*/magfield.bin")

set(MAGFIELD_SYMLINKS_FOR_DOWNLOAD)
set(MAGFIELD_SYMLINKS_FOUND)
foreach(f ${MAGFIELD_LINKS})
  file(REAL_PATH "${f}" real_f)
  if(NOT EXISTS "${real_f}")
    list(APPEND MAGFIELD_SYMLINKS_FOR_DOWNLOAD "${real_f}")
  elseif(NOT real_f IN_LIST MAGFIELD_SYMLINKS_FOUND)
    message(STATUS "  Found: ${real_f}")
    list(APPEND MAGFIELD_SYMLINKS_FOUND ${real_f})
  endif()
endforeach()

if(MAGFIELD_SYMLINKS_FOR_DOWNLOAD)
  # Force INSTALL_GEOMETRY flag true
  if(NOT INSTALL_GEOMETRY)
    message(WARNING "Forcing Geometry to be installed with local mag fields.
This is necessary due to missing magnetic field .bin files")
    set(INSTALL_GEOMETRY ON CACHE BOOL "" FORCE)
  endif()
  # Loop through the broken symlinks to find the versions and polarities
  set(VERSION_AND_POLARITY) # The list to be downloaded

  # Extract file info for download
  foreach(f_symlink ${MAGFIELD_SYMLINKS_FOR_DOWNLOAD})
    file(READ_SYMLINK "${f_symlink}" f) # This loop is 100% broken symlinks, so this is safe
    get_filename_component(FILE_NAME "${f}" NAME)

    # Identify the field version and polarity
    string(
    REGEX MATCH
    "^field\\.([^.]+)\\.(up|down)\\.bin$"
    _match
    "${FILE_NAME}"
    )

    if(NOT _match)
      message(FATAL_ERROR "Unexpected field filename: ${FILE_NAME}")
    endif()

    set(FIELD_VERSION "${CMAKE_MATCH_1}")
    set(FIELD_POLARITY "${CMAKE_MATCH_2}")

    list(APPEND VERSION_AND_POLARITY "${FIELD_VERSION}|${FIELD_POLARITY}")
  endforeach()

  list(REMOVE_DUPLICATES VERSION_AND_POLARITY)
endif()


# Build time: build the necessary magfiles if they're not available

# Custom script to download files at build time
set(DOWNLOAD_SCRIPT "${CMAKE_BINARY_DIR}/cmake_download.cmake")

file(WRITE "${DOWNLOAD_SCRIPT}" "
# cmake_download.cmake
file(DOWNLOAD \${URL} \${OUT} STATUS status LOG log)
list(GET status 0 code)
list(GET status 1 msg)
if(NOT code EQUAL 0)
  message(FATAL_ERROR
    \"Download failed!\\n\"
    \"  URL: \${URL}\\n\"
    \"  Code: \${code}\\n\"
    \"  Message: \${msg}\\n\"
    \"  Log:\\n\${log}\\n\"
  )
endif()
")


# If there are magfiles to build do that
if(MAGFIELD_SYMLINKS_FOR_DOWNLOAD)
  # We will need python to build the mag fields
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  # Download the python file that converts CDF files to bin
  set(PY_SCRIPT "${LOCAL_MAGFIELDS_DIR}/python/cdf2bin.py")

  add_custom_command(
    OUTPUT "${PY_SCRIPT}"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${PY_SCRIPT}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LOCAL_MAGFIELDS_DIR}/python"
    COMMAND ${CMAKE_COMMAND}
          -DURL=https://gitlab.cern.ch/lhcb-datapkg/FieldMap/-/raw/master/scripts/cdf2bin.py
          -DOUT=${PY_SCRIPT}
          -P "${DOWNLOAD_SCRIPT}"
    COMMENT "Downloading cdf2bin.py"
    VERBATIM
  )

  set(GENERATED_MAGFIELDS)
  foreach(vp_list IN LISTS VERSION_AND_POLARITY)
    string(REPLACE "|" ";" field "${vp_list}")
    list(LENGTH field _len)
    if(NOT _len EQUAL 2)
      message(FATAL_ERROR
        "Internal error: expected 'version;polarity', got '${field}'")
    endif()
    list(GET field 0 FIELD_VERSION)
    list(GET field 1 FIELD_POLARITY)

    message(STATUS "  will rebuild field with version: ${FIELD_VERSION} and polarity: ${FIELD_POLARITY}")

    set(OUT_BIN
      "${LOCAL_MAGFIELDS_DIR}/magfield.${FIELD_VERSION}.${FIELD_POLARITY}.bin")

    # Download the CDF files necessary
    set(CDF_FILES)
    foreach(c RANGE 1 4)
      set(CDF_LOCAL
        "${LOCAL_MAGFIELDS_DIR}/cdf/field.${FIELD_VERSION}.c${c}.${FIELD_POLARITY}.cdf")

      add_custom_command(
        OUTPUT "${CDF_LOCAL}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LOCAL_MAGFIELDS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LOCAL_MAGFIELDS_DIR}/cdf"
        COMMAND ${CMAKE_COMMAND} -E remove -f "${CDF_LOCAL}"
        COMMAND ${CMAKE_COMMAND}
              -DURL=https://gitlab.cern.ch/lhcb-datapkg/FieldMap/-/raw/${FIELD_VERSION}/cdf/field.${FIELD_VERSION}.c${c}.${FIELD_POLARITY}.cdf
              -DOUT=${CDF_LOCAL}
              -P "${DOWNLOAD_SCRIPT}"
        COMMENT "Downloading CDF c${c} for field.${FIELD_VERSION}.${FIELD_POLARITY}"
        VERBATIM
      )

      list(APPEND CDF_FILES "${CDF_LOCAL}")
    endforeach()


    # Build the bin files from the CDF and py
    add_custom_command(
      OUTPUT "${OUT_BIN}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${LOCAL_MAGFIELDS_DIR}"
      COMMAND "${Python3_EXECUTABLE}" "${PY_SCRIPT}"
              --output "${OUT_BIN}"
              --input ${CDF_FILES}
      DEPENDS ${CDF_FILES} "${PY_SCRIPT}"
      COMMENT "Generating magfield.${FIELD_VERSION}.${FIELD_POLARITY}.bin"
      VERBATIM
    )

    list(APPEND GENERATED_MAGFIELDS "${OUT_BIN}")
  endforeach()

  message(STATUS "Generated magfields: ${GENERATED_MAGFIELDS}")

  add_custom_target(download_cdf2bin
    DEPENDS ${PY_SCRIPT}
  )
  # Make the cmake target
  add_custom_target(generate_magfields ALL
    DEPENDS ${GENERATED_MAGFIELDS} download_cdf2bin
  )
else()
  message(STATUS "  All necessary magfields found!")
endif()

# Now install the geometry
if(INSTALL_GEOMETRY)
  add_custom_target(install_geometry ALL
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/cmake/utils/copy_geometry.py"
              "${GEOMETRY_DIR}" "${CMAKE_BINARY_DIR}/allen_geometries" "${LOCAL_MAGFIELDS_DIR}"
      COMMENT "Installing geometry to build tree"
  )
  if(MAGFIELD_SYMLINKS_FOR_DOWNLOAD)
    add_dependencies(install_geometry generate_magfields)
  endif()
endif()
