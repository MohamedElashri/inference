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

# Deal with configuration generation machinery
# * If clang is available, we can and will generate the configuration files
# * Otherwise, fail and message that it is not possible to generate configurations
set(CODE_GENERATION_DIR ${PROJECT_BINARY_DIR}/code_generation)
set(PROJECT_SEQUENCE_DIR ${CODE_GENERATION_DIR}/sequences)
set(SEQUENCE_DEFINITION_DIR ${PROJECT_SEQUENCE_DIR}/AllenConf)
set(ALLEN_ALGORITHMS_DIR ${PROJECT_SEQUENCE_DIR}/AllenAlgorithms)
set(ALLEN_GENERATED_INCLUDE_FILES_DIR ${PROJECT_SEQUENCE_DIR}/include)
set(ALLEN_CORE_DIR ${PROJECT_SEQUENCE_DIR}/AllenCore)
set(ALLEN_SEQUENCE_DIR ${PROJECT_SEQUENCE_DIR}/AllenSequences)
set(ALLEN_PARSER_DIR ${PROJECT_SEQUENCE_DIR}/parser)
set(ALGORITHMS_OUTPUTFILE ${ALLEN_ALGORITHMS_DIR}/allen_standalone_algorithms.py)
set(ALGORITHMS_GENERATION_SCRIPT ${PROJECT_SOURCE_DIR}/configuration/parser/ParseAlgorithms.py)
set(DEFAULT_PROPERTIES_SRC ${PROJECT_SOURCE_DIR}/configuration/src/default_properties.cpp)

include_guard(GLOBAL)

file(MAKE_DIRECTORY ${CODE_GENERATION_DIR})
file(MAKE_DIRECTORY ${ALLEN_PARSER_DIR})
file(MAKE_DIRECTORY ${ALLEN_GENERATED_INCLUDE_FILES_DIR})
file(MAKE_DIRECTORY ${ALLEN_ALGORITHMS_DIR})

# Symlink Allen build directories
file(RELATIVE_PATH PROJECT_SOURCE_DIR_RELPATH ${PROJECT_SEQUENCE_DIR} ${PROJECT_SOURCE_DIR})
message(STATUS "Set project source dir to: ${PROJECT_SOURCE_DIR_RELPATH}")
add_custom_command(
  OUTPUT "${SEQUENCE_DEFINITION_DIR}" "${ALLEN_CORE_DIR}" "${ALLEN_SEQUENCE_DIR}"
  COMMENT "Making symlink of sequence definitions and configuration utilities"
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR_RELPATH}/configuration/python/AllenConf" "${SEQUENCE_DEFINITION_DIR}" &&
    ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR_RELPATH}/configuration/python/AllenCore" "${ALLEN_CORE_DIR}" &&
    ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR_RELPATH}/configuration/python/AllenSequences" "${ALLEN_SEQUENCE_DIR}"
  DEPENDS "${PROJECT_SOURCE_DIR}/configuration/python/AllenConf" "${PROJECT_SOURCE_DIR}/configuration/python/AllenCore" "${PROJECT_SOURCE_DIR}/configuration/python/AllenSequences")
add_custom_target(generate_conf_core DEPENDS "${SEQUENCE_DEFINITION_DIR}" "${ALLEN_CORE_DIR}" "${ALLEN_SEQUENCE_DIR}")

add_executable(default_properties ${DEFAULT_PROPERTIES_SRC})
target_link_libraries(default_properties PRIVATE AllenLib HostEventModel EventModel Gear ${ALLEN_ALGORITHM_LIB})
if (NOT STANDALONE)
  target_link_libraries(default_properties PRIVATE LHCb::DetDescLib)
endif()

set(PARSER_ENV PYTHONPATH=$ENV{PYTHONPATH} LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH})

# Generate allen standalone algorithms file
add_custom_command(
  OUTPUT "${ALGORITHMS_OUTPUTFILE}"
  COMMAND
    ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate views --filename "${ALGORITHMS_OUTPUTFILE}" --default_properties $<TARGET_FILE:default_properties> --prefix_project_folder "${PROJECT_SOURCE_DIR}" &&
    ${CMAKE_COMMAND} -E touch ${ALLEN_ALGORITHMS_DIR}/__init__.py
  WORKING_DIRECTORY ${ALLEN_PARSER_DIR}
  DEPENDS generate_conf_core default_properties)
add_custom_target(generate_algorithms_view DEPENDS "${ALGORITHMS_OUTPUTFILE}")
install(FILES "${ALGORITHMS_OUTPUTFILE}" DESTINATION python/AllenAlgorithms)

# Target that the generation of the sequences can depend on
add_custom_target(Sequences DEPENDS generate_algorithms_view)

# Make ExternLines.cuh
if(SEPARABLE_COMPILATION)
  add_custom_command(
    OUTPUT "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh"
    COMMAND
      ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate extern_lines --filename "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh" --prefix_project_folder "${PROJECT_SOURCE_DIR}"
    WORKING_DIRECTORY ${ALLEN_PARSER_DIR})
else()
  add_custom_command(
    OUTPUT "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh"
    COMMAND
      ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate extern_lines_nosepcomp --filename "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh" --prefix_project_folder "${PROJECT_SOURCE_DIR}"
    WORKING_DIRECTORY ${ALLEN_PARSER_DIR})
endif()
add_custom_target(extern_lines_generation DEPENDS "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh")
add_library(extern_lines INTERFACE)
add_dependencies(extern_lines extern_lines_generation)
target_include_directories(extern_lines INTERFACE $<BUILD_INTERFACE:${ALLEN_GENERATED_INCLUDE_FILES_DIR}>)
install(TARGETS extern_lines
      EXPORT Allen
      LIBRARY DESTINATION lib)

if(STANDALONE)
  if (DEFINED ENV{LHCBROOT})
    set(LHCBROOT $ENV{LHCBROOT} CACHE STRING "LHCB root directory")
    set(LHCBOUTPUTS
      "${LHCBROOT}/Event/DAQEvent/src/RawBank.cpp"
      "${LHCBROOT}/Event/DAQEvent/src/ODIN.cpp")

    add_custom_command(
      OUTPUT "${PROJECT_SEQUENCE_DIR}/PyConf"
      BYPRODUCTS ${LHCBOUTPUTS}
      COMMENT "Selecting user-specified LHCBROOT"
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${LHCBROOT}/PyConf/python/PyConf ${PROJECT_SEQUENCE_DIR}/PyConf)
    add_custom_target(checkout_lhcb DEPENDS "${PROJECT_SEQUENCE_DIR}/PyConf")
    message(STATUS "LHCBROOT set to ${LHCBROOT}")
  else()
    find_package(Git REQUIRED)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/external/LHCb")
    set(LHCBROOT "${PROJECT_BINARY_DIR}/external/LHCb" CACHE STRING "LHCB root directory")
    file(RELATIVE_PATH LHCBROOT_RELBIN ${PROJECT_BINARY_DIR} ${LHCBROOT})
    file(RELATIVE_PATH LHCBROOT_RELSEQ ${PROJECT_SEQUENCE_DIR} ${LHCBROOT})
    set(LHCBOUTPUTS
      "${LHCBROOT}/Event/DAQEvent/src/RawBank.cpp"
      "${LHCBROOT}/Event/DAQEvent/src/ODIN.cpp")

    if(NOT LHCB_TARGET_BRANCH)
      set(LHCB_TARGET_BRANCH "master" CACHE STRING "LHCB target branch")
    endif()

    add_custom_command(
      OUTPUT "${PROJECT_SEQUENCE_DIR}/PyConf"
      BYPRODUCTS ${LHCBOUTPUTS}
      COMMENT "Checking out LHCb project from the LHCb stack"
      COMMAND
        ${CMAKE_COMMAND} -E rm -rf ${PROJECT_BINARY_DIR}/external/LHCb/Event &&
        ${CMAKE_COMMAND} -E env ${GIT_EXECUTABLE} clone -b ${LHCB_TARGET_BRANCH} https://gitlab.cern.ch/lhcb/LHCb.git ${PROJECT_BINARY_DIR}/external/LHCb &&
        ${CMAKE_COMMAND} -E create_symlink ${LHCBROOT_RELSEQ}/PyConf/python/PyConf ${PROJECT_SEQUENCE_DIR}/PyConf)
    add_custom_target(checkout_lhcb DEPENDS "${PROJECT_SEQUENCE_DIR}/PyConf" ${LHCBOUTPUTS})
    message(STATUS "LHCBROOT set to ${LHCBROOT}")
    message(STATUS "LHCB_TARGET_BRANCH set to ${LHCB_TARGET_BRANCH}")
  endif()

  if (DEFINED ENV{GAUDIROOT})
    set(GAUDIROOT $ENV{GAUDIROOT} CACHE STRING "GAUDI root directory")
    add_custom_command(
      OUTPUT "${PROJECT_SEQUENCE_DIR}/GaudiKernel"
      COMMENT "Selecting user-specified GAUDIROOT"
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${GAUDIROOT}/GaudiKernel/python/GaudiKernel ${PROJECT_SEQUENCE_DIR}/GaudiKernel)
    add_custom_target(checkout_gaudi DEPENDS "${PROJECT_SEQUENCE_DIR}/GaudiKernel")
    message(STATUS "GAUDIROOT set to ${GAUDIROOT}")
  else()
    find_package(Git REQUIRED)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/external/Gaudi")
    set(GAUDIROOT "${PROJECT_BINARY_DIR}/external/Gaudi" CACHE STRING "GAUDI root directory")
    file(RELATIVE_PATH GAUDIROOT_RELPATH ${PROJECT_SEQUENCE_DIR} ${GAUDIROOT})

    if(NOT GAUDI_TARGET_BRANCH)
      set(GAUDI_TARGET_BRANCH "master" CACHE STRING "Gaudi target branch")
    endif()

    add_custom_command(
      OUTPUT "${PROJECT_SEQUENCE_DIR}/GaudiKernel"
      COMMENT "Checking out Gaudi project from the LHCb stack"
      COMMAND
        ${CMAKE_COMMAND} -E env ${GIT_EXECUTABLE} clone -b ${GAUDI_TARGET_BRANCH} https://gitlab.cern.ch/gaudi/Gaudi.git ${PROJECT_BINARY_DIR}/external/Gaudi &&
        ${CMAKE_COMMAND} -E create_symlink ${GAUDIROOT_RELPATH}/GaudiKernel/python/GaudiKernel ${PROJECT_SEQUENCE_DIR}/GaudiKernel)
    add_custom_target(checkout_gaudi DEPENDS "${PROJECT_SEQUENCE_DIR}/GaudiKernel")
    message(STATUS "GAUDIROOT set to ${GAUDIROOT}")
    message(STATUS "GAUDI_TARGET_BRANCH set to ${GAUDI_TARGET_BRANCH}")
  endif()

  # Unfortunately this has to be defined here to make the generated
  # files work. CMake doesn't support doing this in another
  # directory...
  add_library(LHCbEvent STATIC ${LHCBOUTPUTS})
  if(STANDALONE)
    target_compile_definitions(LHCbEvent PUBLIC ODIN_WITHOUT_GAUDI)
  endif()
  add_dependencies(LHCbEvent checkout_lhcb checkout_gaudi)
  target_link_libraries(LHCbEvent PUBLIC ROOT::Core ROOT::MathCore Boost::headers)
  target_include_directories(
    LHCbEvent
    PUBLIC
    $<BUILD_INTERFACE:${GAUDIROOT}/GaudiKernel/include>
    $<BUILD_INTERFACE:${LHCBROOT}/Event/DAQEvent/include>
    $<BUILD_INTERFACE:${LHCBROOT}/Kernel/LHCbMath/include>)
endif()

file(GLOB python_allen_conf "${PROJECT_SOURCE_DIR}/configuration/python/AllenConf/*py")
file(GLOB python_allen_core "${PROJECT_SOURCE_DIR}/configuration/python/AllenCore/*py")
function(generate_sequence sequence)
  set(sequence_dir ${PROJECT_SEQUENCE_DIR}/${sequence})
  file(MAKE_DIRECTORY ${sequence_dir})
  if(NOT STANDALONE)
    configure_file(${PROJECT_SOURCE_DIR}/scripts/generate_script.sh.in ${sequence_dir}/generate_${sequence}.sh @ONLY)
    add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/${sequence}.json"
      COMMAND
        ${CMAKE_BINARY_DIR}/run bash ${sequence_dir}/generate_${sequence}.sh &&
        ${CMAKE_COMMAND} -E rename "${sequence_dir}/Sequence.json" "${PROJECT_BINARY_DIR}/${sequence}.json"
      DEPENDS generate_conf_core "${PROJECT_SOURCE_DIR}/configuration/python/AllenSequences/${sequence}.py" "${ALGORITHMS_OUTPUTFILE}" ${python_allen_conf} ${python_allen_core}
      WORKING_DIRECTORY ${sequence_dir})
  else()
    add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/${sequence}.json"
      COMMAND
        ${CMAKE_COMMAND} -E env "${LIBRARY_PATH_VARNAME}=$ENV{LD_LIBRARY_PATH}" "PYTHONPATH=${PROJECT_SEQUENCE_DIR}:$ENV{PYTHONPATH}" "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/configuration/python/AllenCore/gen_allen_json.py" "--seqpath" "${PROJECT_SOURCE_DIR}/configuration/python/AllenSequences/${sequence}.py" "--no-register-keys" &&
        ${CMAKE_COMMAND} -E rename "${sequence_dir}/Sequence.json" "${PROJECT_BINARY_DIR}/${sequence}.json"
      DEPENDS generate_conf_core "${PROJECT_SOURCE_DIR}/configuration/python/AllenSequences/${sequence}.py" "${ALGORITHMS_OUTPUTFILE}" "${PROJECT_SEQUENCE_DIR}/GaudiKernel" "${PROJECT_SEQUENCE_DIR}/PyConf" ${python_allen_conf} ${python_allen_core}
      WORKING_DIRECTORY ${sequence_dir})
  endif()
  add_custom_target(sequence_${sequence} DEPENDS "${PROJECT_BINARY_DIR}/${sequence}.json")
  add_dependencies(Sequences sequence_${sequence})
  install(FILES "${PROJECT_BINARY_DIR}/${sequence}.json" DESTINATION constants)
endfunction()
