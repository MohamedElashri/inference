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

#include "BackendCommon.h"
#include <type_traits>
#include <utility>

namespace Allen::Conditions {
  template<typename T>
  class HostDeviceCondition {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    static_assert(std::is_trivially_destructible_v<T>, "T must be trivially destructible");

  public:
    HostDeviceCondition() = default;

    HostDeviceCondition(T&& data)
    {
      Allen::malloc_host((void**) &m_host_ptr, sizeof(T));
      Allen::malloc((void**) &m_dev_ptr, sizeof(T));

      *m_host_ptr = std::move(data);
      Allen::memcpy(m_dev_ptr, m_host_ptr, sizeof(T), Allen::memcpyHostToDevice);
    }

    HostDeviceCondition(const std::vector<char>& data) :
      HostDeviceCondition<T>([&data] {
        assert(data.size() == sizeof(T));
        T geom {};
        std::memcpy(&geom, data.data(), data.size());
        return geom;
      }())
    {}

    // Move constructor
    HostDeviceCondition(HostDeviceCondition&& other) noexcept :
      m_host_ptr(std::exchange(other.m_host_ptr, nullptr)), m_dev_ptr(std::exchange(other.m_dev_ptr, nullptr))
    {}

    // Move assignment
    HostDeviceCondition& operator=(HostDeviceCondition&& other) noexcept
    {
      if (this != &other) {
        cleanup();
        m_host_ptr = std::exchange(other.m_host_ptr, nullptr);
        m_dev_ptr = std::exchange(other.m_dev_ptr, nullptr);
      }
      return *this;
    }

    // Delete copy operations
    HostDeviceCondition(const HostDeviceCondition&) = delete;
    HostDeviceCondition& operator=(const HostDeviceCondition&) = delete;

    ~HostDeviceCondition() { cleanup(); }

    T* host() const noexcept { return m_host_ptr; }
    T* device() const noexcept { return m_dev_ptr; }

    std::span<const char> data() const noexcept { return {reinterpret_cast<const char*>(m_host_ptr), sizeof(T)}; }

  private:
    void cleanup()
    {
      if (m_host_ptr) {
        Allen::free_host(m_host_ptr);
        m_host_ptr = nullptr;
      }

      if (m_dev_ptr) {
        Allen::free(m_dev_ptr);
        m_dev_ptr = nullptr;
      }
    }

    T* m_host_ptr {nullptr};
    T* m_dev_ptr {nullptr};
  };
} // namespace Allen::Conditions
