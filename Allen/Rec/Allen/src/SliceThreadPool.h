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

// ----------------------------------------------------------------------------
// Worker thread pool with an MPMC queue that executes submitted event slices,
// tracks per-slice reference counts and returns slices to the input provider
// once they have been processed.
// ----------------------------------------------------------------------------
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

#include "EventMask.h"
#include "InputProvider.h"
#include "IOutputWriter.h"

namespace Allen::Scheduler {
  struct EventSlice {
    unsigned slice_index {};
    unsigned start_event {};
    unsigned number_of_events {};

    std::pair<EventSlice, EventSlice> split() const
    {
      unsigned half = number_of_events / 2;
      return {{slice_index, start_event, half}, {slice_index, start_event + half, number_of_events - half}};
    }

    bool can_split() const { return number_of_events > 1; }

    friend std::ostream& operator<<(std::ostream& os, const EventSlice& slice)
    {
      os << "[" << slice.start_event << " + " << slice.number_of_events << " → "
         << (slice.start_event + slice.number_of_events - 1) << "]";
      return os;
    }
  };

  struct WorkerContext {
    unsigned stream_id {};
    Allen::Context context {};
    Allen::Store::host_memory_manager_t host_allocator;
    Allen::Store::device_memory_manager_t device_allocator;
    SlabAllocator<Allen::details::shared_buffer_metadata> meta_allocator;
    SlabAllocator<Allen::details::type_erased_dependency> dep_allocator;
    std::vector<EventMask> input_event_masks;
    std::vector<EventMask> event_masks;
    std::vector<size_t> stores {};
    IInputProvider* input_provider {nullptr};
    IOutputWriter* output_writer {nullptr};
  };

  // Simple MPMC queue with mutex
  template<typename T>
  class MPMCQueue {
  public:
    void push(T value)
    {
      {
        std::lock_guard<std::mutex> lock {m_mutex};
        m_queue.push(std::move(value));
      }
      m_not_empty.notify_one();
    }

    // Block until an item is available or the timeout elapses.
    template<typename Rep, typename Period>
    bool pop_wait_for(T& value, const std::chrono::duration<Rep, Period>& timeout)
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      if (!m_not_empty.wait_for(lock, timeout, [this] { return !m_queue.empty(); })) {
        return false;
      }
      value = std::move(m_queue.front());
      m_queue.pop();
      return true;
    }

  private:
    mutable std::mutex m_mutex;
    std::queue<T> m_queue;
    std::condition_variable m_not_empty;
  };

  class SliceThreadPool {
  public:
    using ProcessFunc = std::function<void(const EventSlice&, WorkerContext&)>;
    using CreateCtxFunc = std::function<WorkerContext(const unsigned i)>;

    SliceThreadPool(const unsigned n_threads, const unsigned n_slices, CreateCtxFunc init, ProcessFunc processor) :
      m_init(std::move(init)), m_processor(std::move(processor)), m_stop(false)
    {
      m_slice_ref_count = std::vector<std::atomic<size_t>>(n_slices);
      m_workers.reserve(n_threads);
      for (unsigned i = 0; i < n_threads; i++) {
        m_workers.emplace_back(&SliceThreadPool::worker_loop, this, i);
      }
    }

    ~SliceThreadPool()
    {
      m_stop = true;
      for (auto& worker : m_workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
    }

    void submit(EventSlice slice)
    {
      m_slice_ref_count[slice.slice_index]++;
      m_pending.fetch_add(1, std::memory_order_acq_rel);
      m_queue.push(std::move(slice));
    }

    void free_slice(EventSlice slice, IInputProvider* input_provider)
    {
      auto count = --m_slice_ref_count[slice.slice_index];
      if (count == 0) {
        input_provider->slice_free(slice.slice_index);
      }
    }

    bool is_complete() const
    {
      // No submitted slices left to process and free.
      return m_pending.load(std::memory_order_acquire) == 0;
    }

    // Block the main thread until all work is complete
    void wait_for_completion()
    {
      while (!is_complete()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }

  private:
    void worker_loop(const unsigned i)
    {
      auto ctx = m_init(i);

      while (!m_stop) {
        EventSlice slice;

        // Wait for work; the timeout lets us periodically re-check m_stop.
        if (!m_queue.pop_wait_for(slice, std::chrono::milliseconds(100))) {
          continue;
        }

        try {
          m_processor(slice, ctx);
        } catch (const MemoryException& e) {
          std::cout << "Insufficient memory to process slice - will sub-divide and retry.\n";
          if (slice.can_split()) {
            // Failed, split and resubmit both halves
            auto [left, right] = slice.split();
            submit(left);
            submit(right);
          }
          else {
            std::cout << "Slice cannot be split further - passthrough.\n";
            ctx.output_writer->singleEventPassthrough()->write(
              slice.slice_index, slice.start_event, ctx.input_provider, ctx.stream_id);
          }
        }
        free_slice(slice, ctx.input_provider);

        m_pending.fetch_sub(1, std::memory_order_acq_rel);
      }
    }

    MPMCQueue<EventSlice> m_queue;
    std::vector<std::thread> m_workers;
    std::vector<std::atomic<size_t>> m_slice_ref_count;
    std::atomic<size_t> m_pending {0};
    CreateCtxFunc m_init;
    ProcessFunc m_processor;
    std::atomic<bool> m_stop;
  };
} // namespace Allen::Scheduler
