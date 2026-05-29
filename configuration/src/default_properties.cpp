/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <Algorithm.cuh>

int main()
{
  nlohmann::json default_properties;

  // Loop over the algorithms, instantiate each algorithm and get its
  // (default valued) properties.
  for (const auto& [id, alg] : Allen::AlgorithmDB::get()->all_algorithms()) {
    // std::cerr << " Generating: " << id << std::endl;
    auto infos = alg.get_algorithm_infos();
    infos["name"] = id;
    default_properties[id] = infos;
  }
  // std::cerr << std::setw(4) << default_properties;
  std::cout << std::setw(4) << default_properties;
}
