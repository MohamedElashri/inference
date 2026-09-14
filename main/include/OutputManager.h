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

// ----------------------------------------------------------------------------
// Singleton owning the per-producer SPSC ring buffers used to hand selected
// events from Allen producers to output handlers.
// ----------------------------------------------------------------------------
#pragma once

#include <vector>
#include <SPSCRingBuffer.h>
#include <InputReader.h>
#include <regex>

struct OutputManager {
  static OutputManager* get()
  {
    static OutputManager instance;
    return &instance;
  }

  void init(int n_producers, size_t capacity)
  {
    m_buffers.reserve(n_producers);
    for (int i = 0; i < n_producers; i++) {
      m_buffers.emplace_back(std::make_shared<SPSCRingBuffer>(capacity));
    }
  }

  int n_producers() const { return m_buffers.size(); }

  SPSCRingBuffer* buffer(int producer_id) { return m_buffers[producer_id].get(); }

  std::span<char> reserve_write(int producer_id, std::size_t s) { return m_buffers[producer_id]->reserve_write(s); }

  void commit(int producer_id) { m_buffers[producer_id]->commit(); }

private:
  std::vector<std::shared_ptr<SPSCRingBuffer>> m_buffers;
};
