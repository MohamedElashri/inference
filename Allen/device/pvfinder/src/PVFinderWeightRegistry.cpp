/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "PVFinderWeightRegistry.h"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <BackendCommon.h>

namespace PVFinder {

  void WeightRegistry::load(const std::string& key, const std::string& file_path) {
    if (contains(key)) return; // Already loaded

    std::ifstream f(file_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
      throw StrException("WeightRegistry: cannot open " + file_path);
    }
    const size_t bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> host_buf(bytes);
    f.read(host_buf.data(), static_cast<std::streamsize>(bytes));
    load_from_buffer(key, host_buf.data(), bytes);
  }

  void WeightRegistry::load_from_buffer(const std::string& key,
                                         const void* host_data,
                                         size_t bytes) {
    if (contains(key)) {
      throw StrException("WeightRegistry: key already registered: " + key);
    }

    void* dev_ptr = nullptr;
    // Use Allen's backend agnostic allocation
    Allen::malloc(&dev_ptr, bytes);
    Allen::memcpy(dev_ptr, host_data, bytes, Allen::memcpyHostToDevice);
    
    m_registry.push_back(Entry{key, dev_ptr, bytes});
  }

} // namespace PVFinder
