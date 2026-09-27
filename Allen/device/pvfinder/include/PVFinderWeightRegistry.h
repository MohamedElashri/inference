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
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include "Common.h"
#include <cstddef>

namespace PVFinder {

  /**
   * @brief Singleton registry mapping string keys to device-side weight tensors.
   *
   * Weights are allocated outside Allen's pool using a direct cudaMalloc or hipMalloc
   * and persist for the lifetime of the process.
   */
  class WeightRegistry {
  public:
    static WeightRegistry& instance() {
      static WeightRegistry s_instance;
      return s_instance;
    }

    /**
     * @brief Load weights from a flat binary file into device memory.
     */
    void load(const std::string& key, const std::string& file_path);

    /**
     * @brief Load weights from a host-side buffer.
     */
    void load_from_buffer(const std::string& key, const void* host_data, size_t bytes);

    /**
     * @brief Returns typed const pointer into device memory.
     */
    template<typename T>
    const T* get(const std::string& key) const {
      for (const auto& entry : m_registry) {
        if (entry.key == key) {
          return static_cast<const T*>(entry.dev_ptr);
        }
      }
      throw StrException("WeightRegistry: key not found: " + key);
    }

    size_t size_bytes(const std::string& key) const {
      for (const auto& entry : m_registry) {
        if (entry.key == key) {
          return entry.bytes;
        }
      }
      return 0;
    }

    bool contains(const std::string& key) const {
      for (const auto& entry : m_registry) {
        if (entry.key == key) {
          return true;
        }
      }
      return false;
    }

    // Prevent accidental copying
    WeightRegistry(const WeightRegistry&) = delete;
    WeightRegistry& operator=(const WeightRegistry&) = delete;

  private:
    WeightRegistry() = default;

    struct Entry {
      std::string key;
      void*  dev_ptr = nullptr;
      size_t bytes   = 0;
    };

    std::vector<Entry> m_registry;
  };

} // namespace PVFinder
