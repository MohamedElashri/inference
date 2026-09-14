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

#include <atomic>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <span>

namespace {
  constexpr unsigned CACHE_LINE_SIZE = 64;
} // namespace

/**
 * @Brief: Single Producer / Single Consumer Ring Buffer
 */
struct SPSCRingBuffer {
  SPSCRingBuffer(std::size_t capacity)
  {
    // This constructor allocates m_capacity bytes in physical memory, but map it to
    // a contiguous virtual range of 2 * m_capacity, where the first and second halves
    // are aliased. This trick allows to return standard spans that wrap around the buffer

    unsigned page_size = getpagesize();
    if ((capacity < page_size) || (page_size <= 0) || ((capacity % page_size) != 0)) {
      throw std::runtime_error("Requested ring capacity is not a multiple of page size.");
    }

    int memfd = memfd_create("ring", 0);
    if (memfd == -1) {
      throw std::runtime_error("Could not create ring file descriptor");
    }

    if (ftruncate(memfd, capacity) == -1) {
      throw std::runtime_error("Could not set ring size");
    }

    // Have the kernel find a contiguous range of unused address space.
    char* base = (char*) mmap(NULL, 2 * capacity, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (base == MAP_FAILED) {
      throw std::runtime_error("Could not allocate ring buffer");
    }

    // Map that "ring" file 2 times, filling that range exactly.
    for (int i = 0; i < 2; i++) {
      void* p = mmap(base + (i * capacity), capacity, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, memfd, 0);
      if (p == MAP_FAILED) {
        throw std::runtime_error("Could not map ring buffer");
      }
    }

    close(memfd);

    m_reader.base = base;
    m_reader.capacity = capacity;
    m_writer.base = base;
    m_writer.capacity = capacity;
  }

  ~SPSCRingBuffer()
  {
    if (m_reader.base) {
      munmap(m_reader.base, m_reader.capacity * 2);
    }
  }

  std::span<char> reserve_write(std::size_t s)
  {
    m_writer.end =
      ((m_writer.begin + s) > m_writer.capacity) ? (m_writer.begin + s - m_writer.capacity) : (m_writer.begin + s);
    std::size_t available_space;
    do {
      std::size_t rb = m_reader_begin.load(std::memory_order_acquire);
      available_space = (rb >= m_writer.end) ? rb - m_writer.end : rb - m_writer.end + m_writer.capacity;
    } while (available_space < s);
    return {m_writer.base + m_writer.begin, s};
  }

  void commit()
  {
    m_writer.begin = m_writer.end;
    m_writer_begin.store(m_writer.end, std::memory_order_release);
  }

  std::span<char> consume()
  {
    m_reader.end = m_writer_begin.load(std::memory_order_acquire);
    std::size_t size = (m_reader.end >= m_reader.begin) ? m_reader.end - m_reader.begin :
                                                          m_reader.end - m_reader.begin + m_reader.capacity;
    return {m_reader.base + m_reader.begin, size};
  }

  void release()
  {
    m_reader.begin = m_reader.end;
    m_reader_begin.store(m_reader.end, std::memory_order_release);
  }

private:
  // Use dedicated cache lines for reader and writer local and
  // shared states to prevent false sharing
  struct alignas(CACHE_LINE_SIZE) LocalState {
    char* base {nullptr};
    std::size_t capacity {0};
    std::size_t begin {0};
    std::size_t end {0};
  };
  LocalState m_reader {};
  LocalState m_writer {};

  alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> m_reader_begin {0};
  alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> m_writer_begin {0};
};
