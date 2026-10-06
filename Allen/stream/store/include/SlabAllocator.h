/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

template<typename T>
class SlabAllocator {
  static_assert(sizeof(T) >= sizeof(void*), "T must be large enough to store freelist pointer");

  static constexpr size_t BLOCK_SIZE = 4096; // BLOCK_SIZE is bytes per system allocation

  struct Block {
    Block* next;
    char data[BLOCK_SIZE - sizeof(Block*)];
  };

  Block* m_blocks = nullptr;
  T* m_freelist = nullptr;
  size_t m_objects_per_block = (BLOCK_SIZE - sizeof(Block*)) / sizeof(T);

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  SlabAllocator() = default;

  template<typename U>
  SlabAllocator(const SlabAllocator<U>&) noexcept
  {}

  SlabAllocator(const SlabAllocator&) = delete;
  SlabAllocator& operator=(const SlabAllocator&) = delete;

  SlabAllocator(SlabAllocator&& other) noexcept :
    m_blocks(std::exchange(other.m_blocks, nullptr)), m_freelist(std::exchange(other.m_freelist, nullptr))
  {}

  ~SlabAllocator()
  {
    while (m_blocks) {
      Block* next = m_blocks->next;
      delete m_blocks;
      m_blocks = next;
    }
  }

  T* allocate([[maybe_unused]] size_t n = 1)
  {
    assert(n == 1);
    if (!m_freelist) {
      allocate_new_block();
    }
    T* obj = m_freelist;
    m_freelist = *reinterpret_cast<T**>(m_freelist); // Next free object
    return obj;
  }

  void deallocate(T* obj, [[maybe_unused]] size_t n = 1)
  {
    assert(n == 1);
    *reinterpret_cast<T**>(obj) = m_freelist; // Store next pointer in the object
    m_freelist = obj;
  }

  bool operator==(const SlabAllocator&) const { return true; }
  bool operator!=(const SlabAllocator&) const { return false; }

private:
  void allocate_new_block()
  {
    Block* new_block = static_cast<Block*>(::operator new(BLOCK_SIZE));
    new_block->next = m_blocks;
    m_blocks = new_block;

    // Build freelist from this block
    char* start = new_block->data;
    for (size_t i = 0; i < m_objects_per_block; ++i) {
      T* obj = reinterpret_cast<T*>(start + i * sizeof(T));
      deallocate(obj, 1); // Adds to freelist
    }
  }
};
