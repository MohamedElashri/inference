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
#[========================================[.rst:
Rec/Allen
------------------
Utility algorithms to run the Allen
sequence as part of a Gaudi/LHCb
application
#]========================================]

include(GenerateConfiguration)

file(GLOB ALLEN_WRAPPER_SOURCES "src/*.cpp")

gaudi_add_module(AllenWrapper
                 SOURCES
                   ${ALLEN_WRAPPER_SOURCES}
                 LINK
                   AllenLib
                   Backend
                   EventModel
                   HostEventModel
                   Stream

                   Gaudi::GaudiAlgLib
                   LHCb::DAQEventLib
                   LHCb::DetDescLib
                   LHCb::RecEvent
                   LHCb::VPDetLib
                   LHCb::UTDetLib
                   LHCb::UTKernelLib
                   LHCb::CaloDetLib
                   LHCb::FTDetLib
                   LHCb::MCEvent
                   LHCb::RecEvent
                   LHCb::RichFutureUtils
                   Rec::PrKernel)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_definitions(AllenWrapper PRIVATE ALLEN_DEBUG)
endif()
target_compile_definitions(AllenWrapper PRIVATE -D${TARGET_DEFINITION})

gaudi_add_module(AllenAlgorithms
                 LINK
                   ${ALLEN_ALGORITHM_LIB}
                   AllenLib
                   Backend
                   EventModel
                   HostEventModel
                   NonEventData
                   Stream
                   Gaudi::GaudiAlgLib
                   LHCb::DAQEventLib
                   LHCb::DetDescLib
                   LHCb::VPDetLib
                   LHCb::UTDetLib
                   LHCb::UTKernelLib
                   LHCb::CaloDetLib
                   LHCb::FTDetLib
                   LHCb::MCEvent
                   Rec::PrKernel)

add_custom_target(gaudi_wrappers_of_allen_algorithms DEPENDS ${ALLEN_ALGORITHM_LIB})
add_dependencies(AllenAlgorithms gaudi_wrappers_of_allen_algorithms)
target_compile_definitions(AllenAlgorithms PRIVATE -D${TARGET_DEFINITION})
# We cannot fix use of lambda capture `[=]` (C++17) or `[=,this]` (C++20) at the same time
# so we hide the warning when we use C++17.
target_compile_options(AllenAlgorithms PRIVATE -Wno-c++20-extensions)

gaudi_install(PYTHON)

gaudi_add_pytest(tests/pytest)

gaudi_add_executable(test_mass_definitions
    SOURCES
        tests/src/test_mass_definitions.cpp
    LINK
        Boost::unit_test_framework
        AllenLib
        LHCb::LHCbKernel
        LHCb::PartPropLib
    TEST
)

if(BUILD_TESTING AND NOT USE_DD4HEP)
    # Disable test on data for detdesc builds
    list(APPEND tests_to_disable
        Allen.pytest.test_mdf_input_2025_data
        Allen.pytest.test_mdf_input_2026_data_tae
    )
endif()


if(tests_to_disable)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/disable_tests.cmake
        "set_tests_properties(${tests_to_disable} PROPERTIES DISABLED TRUE)"
    )
    set_property(DIRECTORY APPEND PROPERTY
        TEST_INCLUDE_FILES ${CMAKE_CURRENT_BINARY_DIR}/disable_tests.cmake)
endif()
