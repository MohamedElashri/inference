/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "BackendCommon.h"

#ifdef __linux__
#include <fstream>
#include <regex>
#endif

thread_local GridDimensions gridDim;
thread_local BlockIndices blockIdx;

namespace Allen {
  std::tuple<bool, std::string, unsigned, unsigned> set_device(int id, size_t)
  {
#ifdef __linux__
    std::ifstream cpuinfo {"/proc/cpuinfo"};
    std::string processor_name;

    for (std::string line; std::getline(cpuinfo, line);) {
      if (!line.starts_with("model name")) continue;

      const auto colon = line.find(':');
      if (colon != std::string::npos) {
        processor_name = line.substr(colon + 1);
        processor_name.erase(0, processor_name.find_first_not_of(" \t"));
      }
      break;
    }

    if (processor_name.empty()) processor_name = "CPU";

    // Clean the string
    const std::regex regex_to_remove {"(\\(R\\))|(CPU )|( @.*)|(\\(TM\\))|( Processor)"};
    processor_name = std::regex_replace(processor_name, regex_to_remove, std::string {});

    return {true, processor_name, cpu_alignment, id};
#else
    return {true, "CPU", cpu_alignment, id};
#endif // linux-dependent CPU detection
  }
} // namespace Allen
