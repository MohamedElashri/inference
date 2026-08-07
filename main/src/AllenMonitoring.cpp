/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "AllenMonitoring.h"
#include <thread>

namespace Allen::Monitoring {

  AccumulatorBase::~AccumulatorBase()
  {
#ifndef ALLEN_STANDALONE
    // Only remove the entity if this instance actually registered one -- calling
    // Gaudi::svcLocator() unconditionally here lazily creates (and, if none was
    // running to begin with, leaks) a whole ApplicationMgr. See m_registered's
    // declaration for the two ways an AccumulatorBase can reach here unregistered.
    if (m_registered) {
      Gaudi::svcLocator()->monitoringHub().removeEntity(*this);
    }
#endif
  }

  void AccumulatorManager::registerAccumulator(AccumulatorBase* acc)
  {
    // We can't do anything yet with acc (the algorithm name is not initialized at this stage)
    // We save it to process later
    m_registered_accumulators.push_back(acc);
  }

  void AccumulatorManager::initAccumulators(unsigned number_of_streams)
  {
    // Algorithms have finished their initializations for all streams
    // This function will init the memory and communicate back pointers to algorithms

    // Allocate memory for buffer here to avoid segfault in sequences that do not use monitoring.
    // std::atomic<T> is neither copy- nor move-constructible, so resize() would not compile on
    // these vectors -- move-assign freshly-constructed ones instead. Safe without synchronization
    // since this runs once, single-threaded, strictly before any worker/aggregation thread starts.
    m_stream_current_buffer = std::vector<std::atomic<unsigned>>(number_of_streams);
    m_stream_done = std::vector<std::atomic<bool>>(number_of_streams);
    for (auto& done : m_stream_done) {
      done.store(true, std::memory_order_relaxed);
    }

    if (m_registered_accumulators.empty()) return;

    // Resize bins and register CountersHistogram
    m_counters_histogram.resetHistogram();
    for (auto counter : m_counters) {
      m_counters_histogram.addCounter(counter->uniqueName());
    }
    for (auto counter : m_av_counters) {
      m_counters_histogram.addAvCounter(counter->uniqueName());
    }
    m_counters_histogram.registerHistogram();

    // * Group registered allocator by unique name (they should be registered once per stream)
    for (auto acc : m_registered_accumulators) {
      auto key = acc->uniqueName();
      m_accumulators[key].size = acc->size();
      m_accumulators[key].element_size = acc->elementSize();
      m_accumulators[key].owners.push_back(acc);
      acc->setBufferInfos(&m_accumulators[key]);
    }
    m_registered_accumulators.clear(); // not needed anymore

    // * Allocate memory on the device and the host to store the accumulators
    m_buffer_size = 0;
    for (auto& [key, acc] : m_accumulators) {
      acc.offset = m_buffer_size;
      m_buffer_size += acc.size * acc.element_size;
    }
    Allen::malloc_host((void**) &m_host_buffer_ptr, m_buffer_size);
    Allen::malloc((void**) &m_dev_buffer_ptr[0], m_buffer_size * 2);
    m_dev_buffer_ptr[1] = m_dev_buffer_ptr[0] + m_buffer_size;
    debug_cout << "Accumulators Host memory: " << m_buffer_size << " bytes, Device memory: " << (m_buffer_size * 2)
               << " bytes (double buffering)" << std::endl;

    // * Init device memory to 0 (both buffers)
    Allen::memset((void*) m_dev_buffer_ptr[0], 0, m_buffer_size * 2);

    for ([[maybe_unused]] auto& [key, acc] : m_accumulators) {
      // * Register the accumulator
      // we only need to create one as the aggregation is done before
      // but for convenience, we delegate the creation to the Allen::Accumulator
      acc.owners[0]->registerAccumulator();
    }
  }

  void AccumulatorManager::mergeAndReset(bool singlethreaded)
  {
    // This function is called by the monitoring thread

    // * Signal all streams to switch to the 2nd buffer
    auto buf = m_current_buffer.load(std::memory_order_acquire);
    m_current_buffer.store(!buf, std::memory_order_release);

    // * Wait acknowledge from the streams
    if (!singlethreaded) {
      auto const next_buf = !buf;
      for (unsigned i = 0; i < m_stream_current_buffer.size(); i++) {
        while (m_stream_current_buffer[i].load(std::memory_order_acquire) != next_buf &&
               !m_stream_done[i].load(std::memory_order_acquire)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
    }

    // * Copy from device to host
    Allen::memcpy(
      (void*) m_host_buffer_ptr, (const void*) m_dev_buffer_ptr[buf], m_buffer_size, Allen::memcpyDeviceToHost);

    // * Reset device buffer to 0
    Allen::memset((void*) m_dev_buffer_ptr[buf], 0, m_buffer_size);

    // * Update accumulators
    for ([[maybe_unused]] auto& [key, acc] : m_accumulators) {
      acc.owners[0]->fillAccumulator((void*) (m_host_buffer_ptr + acc.offset));
    }

    int counter_index = 0;
    for (auto counter : m_counters) {
      m_counters_histogram.updateBin(counter_index, static_cast<float>(counter->m_entries));
      counter_index++;
    }

    for (auto counter : m_av_counters) {
      m_counters_histogram.updateBin(counter_index, static_cast<float>(counter->m_sum));
      counter_index++;
      m_counters_histogram.updateBin(counter_index, static_cast<float>(counter->m_entries));
      counter_index++;
    }
  }
} // namespace Allen::Monitoring
