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
#pragma once

#include <list>
#include <algorithm>
#include "Common.h"
#include "Logger.h"
#include "Argument.cuh"
#include "BackendCommon.h"
#include "MemoryReport.h"
#include "SlabAllocator.h"

// forward declarations
namespace Allen::details {
  struct shared_buffer_metadata;
  struct type_erased_dependency;
} // namespace Allen::details

namespace Allen::Store {
  // Distinguish between single and multi alloc memory managers
  enum struct AllocPolicy { SingleAlloc, MultiAlloc };

  template<Scope S>
  struct MemoryManagerAllocator {
    constexpr static auto scope = S;
    static_assert((S == Scope::Host || S == Scope::Device) && "memory manager allocator scope must be supported");

    static void free(void* ptr)
    {
      if constexpr (S == Scope::Host) {
        Allen::free_host(ptr);
      }
      else {
        Allen::free(ptr);
      }
    }

    static void malloc(void** ptr, size_t s)
    {
      if constexpr (S == Scope::Host) {
        Allen::malloc_host(ptr, s);
      }
      else {
        Allen::malloc(ptr, s);
      }
    }
  };

  template<Scope S, AllocPolicy P>
  struct MemoryManager;

  /**
   * @brief This memory manager allocates a single chunk of memory at the beginning,
   *        and reserves / frees using this memory chunk from that point onwards.
   */
  template<Scope S>
  struct MemoryManager<S, AllocPolicy::SingleAlloc> : MemoryManagerAllocator<S> {
  private:
    std::string m_name = "Memory manager";
    size_t m_max_available_memory = 0;
    unsigned m_guaranteed_alignment = 512;
    char* m_base_pointer = nullptr;
    AllocationReport m_report;

    /**
     * @brief A memory segment is composed of a start
     *        and size, both referencing bytes.
     */
    struct MemorySegment {
      unsigned start;
      size_t size;
      bool used;
    };
    std::list<MemorySegment, SlabAllocator<MemorySegment>> m_memory_segments {{0, m_max_available_memory, false}};

  public:
    MemoryManager() = default;
    MemoryManager(const std::string& name) : m_name {name} {}
    MemoryManager(const std::string& name, const size_t memory_size, const unsigned memory_alignment) :
      m_name {name}, m_max_available_memory {memory_size}, m_guaranteed_alignment {memory_alignment}
    {
      if (m_base_pointer) MemoryManagerAllocator<S>::free(m_base_pointer);
      MemoryManagerAllocator<S>::malloc(reinterpret_cast<void**>(&m_base_pointer), memory_size);
      free_all();
    }

    /**
     * @brief Sets the m_max_available_memory of this manager.
     *        Note: This triggers a free_all to restore the m_memory_segments
     *        to a valid state. This operation is very disruptive.
     */
    void reserve_memory(size_t memory_size, const unsigned memory_alignment)
    {
      if (m_base_pointer) MemoryManagerAllocator<S>::free(m_base_pointer);
      MemoryManagerAllocator<S>::malloc(reinterpret_cast<void**>(&m_base_pointer), memory_size);

      m_guaranteed_alignment = memory_alignment;
      m_max_available_memory = memory_size;
      free_all();
    }

    char* reserve(size_t requested_size)
    {
      // Size requested should be greater than zero
      if (requested_size == 0) {
        constexpr int zero_size_message_verbosity = logger::debug;
        if (logger::verbosity() >= zero_size_message_verbosity) {
          debug_cout << "MemoryManager: Requested to reserve zero bytes."
                     << " Did you forget to set_size?" << std::endl;
        }
        requested_size = 1;
      }

      // Aligned requested size
      const size_t aligned_request = requested_size + m_guaranteed_alignment - 1 -
                                     ((requested_size + m_guaranteed_alignment - 1) % m_guaranteed_alignment);

      if (logger::verbosity() >= 5) {
        verbose_cout << "MemoryManager: Requested to reserve " << requested_size << " B (" << aligned_request
                     << " B aligned)" << std::endl;
      }

      // Finds first free segment providing sufficient space
      auto it = std::find_if(m_memory_segments.begin(), m_memory_segments.end(), [&](const auto& ms) {
        return ms.used == false && ms.size >= aligned_request;
      });

      // Complain if no space was available
      if (it == m_memory_segments.end()) {
        warning_cout << "Reserve: Requested size could not be met (" +
                          std::to_string(static_cast<float>(aligned_request) / (1000.f * 1000.f)) + " MB)\n";
        print();
        throw MemoryException("not enough memory to meet request");
      }

      // Start of allocation
      const auto start = it->start;

      // Update current segment
      it->start += aligned_request;
      it->size -= aligned_request;
      if (it->size == 0) {
        it = m_memory_segments.erase(it);
      }

      // Insert an occupied segment
      auto segment = MemorySegment {start, aligned_request, true};
      m_memory_segments.insert(it, segment);

      // Update total memory required
      // Note: This can be done accesing the last element in m_memory_segments
      //       upon every reserve, and keeping the maximum used memory
      m_report.report_allocation(
        aligned_request); // TODO: virtual peak: m_max_available_memory - m_memory_segments.back().size

      return m_base_pointer + start;
    }

    /**
     * @brief Reserves a memory request of size requested_size, implementation.
     *        Finds the first available segment.
     *        If there are no available segments of the requested size,
     *        it throws an exception.
     */
    void reserve(BaseArgument& argument) { argument.set_pointer(reserve(argument.size_bytes())); }

    void free(char* ptr)
    {
      unsigned start = ptr - m_base_pointer;
      auto it =
        std::find_if(m_memory_segments.begin(), m_memory_segments.end(), [&start](const MemorySegment& segment) {
          return segment.start == start;
        });

      if (it == m_memory_segments.end()) {
        throw std::runtime_error("MemoryManager free: Requested segment could not be found");
      }

      // Free found segment
      it->used = false;

      m_report.report_free(it->size);

      // Check if previous segment is free, in which case, join
      if (it != m_memory_segments.begin()) {
        auto previous_it = std::prev(it);
        if (previous_it->used == false) {
          previous_it->size += it->size;
          // Remove current element, and point to previous one
          it = std::prev(m_memory_segments.erase(it));
        }
      }

      // Check if next segment is free, in which case, join
      if (std::next(it) != m_memory_segments.end()) {
        auto next_it = std::next(it);
        if (next_it->used == false) {
          it->size += next_it->size;
          // Remove next segment
          m_memory_segments.erase(next_it);
        }
      }
    }

    /**
     * @brief Recursive free, implementation for Argument.
     */
    void free(BaseArgument& argument) { free(reinterpret_cast<char*>(argument.pointer())); }

    void test_alignment()
    {
      for (const auto it : m_memory_segments) {
        if (it.used) {
          // Note: Do an assert
          if (!((it.start % m_guaranteed_alignment) == 0)) {
            info_cout << "Found misaligned entry: " << it.start << "\n";
            print();
          }
        }
      }
    }

    /**
     * @brief Frees all memory segments, effectively resetting the
     *        available space.
     */
    void free_all()
    {
      m_memory_segments.clear();
      m_memory_segments.emplace_front(MemorySegment {0, m_max_available_memory, false});
    }

    const auto& report() const { return m_report; }

    /**
     * @brief Prints the current state of the memory segments.
     */
    void print() const
    {
      info_cout << m_name << " segments (MB):" << std::endl;
      for (auto& segment : m_memory_segments) {
        std::string name = segment.used ? "used" : "unused";
        info_cout << name << " (" << segment.start << ", " << static_cast<float>(segment.size) / (1024.f * 1024.f)
                  << "), ";
      }
      info_cout << "\nMax memory required: " << (static_cast<float>(m_report.current_bytes_in_use) / (1024.f * 1024.f))
                << " MB"
                << "\n\n";
    }
  };

  /**
   * @brief This memory manager allocates / frees using the backend calls (eg. Allen::malloc, Allen::free).
   *        It is slower but better at debugging out-of-bound accesses.
   */
  template<Scope S>
  struct MemoryManager<S, AllocPolicy::MultiAlloc> : MemoryManagerAllocator<S> {
  private:
    std::string m_name = "Memory manager";
    AllocationReport m_report;

    /**
     * @brief A memory segment, in the case of MultiAlloc policy,
     *        consists of a pointer to segment association.
     */
    struct MemorySegment {
      char* pointer;
      size_t size;
    };
    std::unordered_map<char*, MemorySegment> m_memory_segments {};

  public:
    MemoryManager() = default;
    MemoryManager(const std::string& name) : m_name {name} {}
    MemoryManager(const std::string& name, const size_t, const unsigned) : m_name {name} {}

    /**
     * @brief This MultiAlloc MemoryManager does not reserve memory upon startup.
     */
    void reserve_memory(size_t, const unsigned) {}

    char* reserve(size_t requested_size)
    {
      /// Size requested should be greater than zero
      if (requested_size == 0) {
        warning_cout << "Warning: MemoryManager: Requested to reserve zero bytes for argument "
                     << ". Did you forget to set_size?" << std::endl;
        requested_size = 1;
      }

      // We will allocate in a char*
      char* memory_pointer;

      MemoryManagerAllocator<S>::malloc(reinterpret_cast<void**>(&memory_pointer), requested_size);

      // Add the pointer to the memory segments map
      m_memory_segments[memory_pointer] = MemorySegment {memory_pointer, requested_size};

      m_report.report_allocation(requested_size);

      return memory_pointer;
    }

    /**
     * @brief Allocates a segment of the requested size.
     */
    void reserve(BaseArgument& argument) { argument.set_pointer(reserve(argument.size_bytes())); }

    void free(char* ptr)
    {
      // Verify the pointer existed in the memory segments map
      const auto it = m_memory_segments.find(ptr);
      if (it == m_memory_segments.end()) {
        print();
        throw MemoryException(
          "MemoryManager free: Requested to free segment but it was not allocated with this MemoryManager");
      }

      MemoryManagerAllocator<S>::free(it->second.pointer);

      m_report.report_free(it->second.size);

      m_memory_segments.erase(ptr);
    }

    /**
     * @brief Frees the requested argument.
     */
    void free(BaseArgument& argument) { free(reinterpret_cast<char*>(argument.pointer())); }

    /**
     * @brief Frees all memory segments, effectively resetting the
     *        available space.
     */
    void free_all()
    {
      for (const auto& it : m_memory_segments) {
        MemoryManagerAllocator<S>::free(it.second.pointer);
      }
      m_memory_segments.clear();
    }

    void test_alignment() {}

    const auto& report() const { return m_report; }

    /**
     * @brief Prints the current state of the memory segments.
     */
    void print() const
    {
      info_cout << m_name << " segments (MB):" << std::endl;
      for (auto const& [ptr, segment] : m_memory_segments) {
        info_cout << static_cast<const void*>(ptr) << " (" << static_cast<float>(segment.size) / (1024.f * 1024.f)
                  << "), ";
      }
      info_cout << "\nMax memory required: " << (static_cast<float>(m_report.current_bytes_in_use) / (1024.f * 1024.f))
                << " MB"
                << "\n\n";
    }
  };

#ifdef MEMORY_MANAGER_MULTI_ALLOC
  template<Scope S>
  using memory_manager_t = MemoryManager<S, AllocPolicy::MultiAlloc>;
#else
  template<Scope S>
  using memory_manager_t = MemoryManager<S, AllocPolicy::SingleAlloc>;
#endif

  using host_memory_manager_t = memory_manager_t<Scope::Host>;
  using device_memory_manager_t = memory_manager_t<Scope::Device>;

  struct memory_managers_t {
    template<Store::Scope S, typename T>
    T* reserve(size_t size) const
    {
      if (size == 0) {
        size = 1;
      }
      if constexpr (S == Store::Scope::Host) {
        if (host_allocator) {
          return reinterpret_cast<T*>(host_allocator->reserve(size * sizeof(T)));
        }
      }
      else if constexpr (S == Store::Scope::Device) {
        if (device_allocator) {
          return reinterpret_cast<T*>(device_allocator->reserve(size * sizeof(T)));
        }
      }
      return reinterpret_cast<T*>(std::malloc(size * sizeof(T)));
    }
    template<Store::Scope S, typename T>
    void free(T* ptr) const
    {
      if constexpr (S == Store::Scope::Host) {
        if (host_allocator == nullptr)
          std::free(reinterpret_cast<void*>(ptr));
        else
          host_allocator->free(reinterpret_cast<char*>(ptr));
      }
      else if constexpr (S == Store::Scope::Device) {
        if (device_allocator == nullptr)
          std::free(reinterpret_cast<void*>(ptr));
        else
          device_allocator->free(reinterpret_cast<char*>(ptr));
      }
    }
    host_memory_manager_t* host_allocator {nullptr};
    device_memory_manager_t* device_allocator {nullptr};
    SlabAllocator<Allen::details::shared_buffer_metadata>* meta_allocator {nullptr};
    SlabAllocator<Allen::details::type_erased_dependency>* dep_allocator {nullptr};
    Allen::Context context {}; // for convenience, keep a copy of the context
  };
} // namespace Allen::Store
