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

// ----------------------------------------------------------------------------
// Thread pool transposing batches of prefetched raw events into Allen slices
// (per-bank-type buffers) and exposing the get_slice/slice_free interface.
// ----------------------------------------------------------------------------
#pragma once

#include <thread>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <memory>
#include <tuple>

#include <StandaloneRawEvent.h>
#include <Event/RawBank.h>
#include <Event/ODIN.h>

#include "PinnedVector.h"
#ifndef ALLEN_STANDALONE
#include "IOAlgorithms/InputFileManifest.h"
#endif

namespace {
  bool check_top5(BankTypes bt, LHCb::RawBank const* bank)
  {
    auto const sys = static_cast<SourceIdSys>(SourceId_sys(bank->sourceID()));
    auto it = Allen::subdetectors.find(sys);
    return it != Allen::subdetectors.end() && it->second == bt;
  }

  void prefix_sum(const std::span<unsigned>& vec)
  {
    unsigned sum = 0;
    for (unsigned i = 0; i < vec.size(); i++) {
      unsigned count = vec[i];
      vec[i] = sum;
      sum += count;
    }
  }
} // namespace

namespace Allen {
  /**
   * @brief      A thread pool for transposing raw events into Allen slices
   *
   * @details    This class manages a pool of worker threads that take batches
   *             of prefetched raw events and transpose them into the per-bank-type
   *             slices required by the Allen processing framework. It provides
   *             thread-safe input and output queues with the same interface
   *             pattern as the original MDFProvider (get_slice/slice_free).
   */
  struct TransposeWorkers {
    /**
     * @brief    Input: a batch of prefetched raw events with their memory buffers
     */
    struct PrefetchedEvents {
      std::vector<LHCb::RawEvent> events;
      std::vector<std::shared_ptr<void>> buffers;
      std::vector<LHCb::ODIN> odin_data;
      std::vector<EventID> event_ids;
      std::vector<char> event_mask; // ODIN error bits
#ifndef ALLEN_STANDALONE
      std::vector<std::vector<DataObject*>> branches;
      std::vector<LHCb::IO::InputFileManifest> input_file_manifests;
#endif
      void reset()
      {
        events.clear();
        buffers.clear();
        odin_data.clear();
        event_ids.clear();
        event_mask.clear();
#ifndef ALLEN_STANDALONE
        branches.clear();
        input_file_manifests.clear();
#endif
      }
      size_t size() const { return events.size(); }
      bool empty() const { return events.empty(); }
    };

    struct BankSliceStorage {
      pinned_vector<char> data;
      pinned_vector<unsigned> offsets;
      pinned_vector<unsigned> sizes;
      pinned_vector<unsigned> types;
      int version = -1;

      void reset()
      {
        data.clear();
        offsets.clear();
        sizes.clear();
        types.clear();
        version = -1;
      }

      BanksAndOffsets banks_and_offsets() const
      {
        if (version == -1 || offsets.empty()) {
          BanksAndOffsets bno {};
          bno.version = version;
          return bno;
        }
        std::span<char const> b {reinterpret_cast<const char*>(data.data()), data.size()};
        return {
          {std::move(b)},
          {offsets.data(), offsets.size()},
          data.size(),
          {sizes.data(), sizes.size()},
          {types.data(), types.size()},
          version};
      }
    };

    struct TransposedSlice {
      std::array<BankSliceStorage, NBankTypes> banks;
      PrefetchedEvents batch;
      void reset()
      {
        for (auto& b : banks)
          b.reset();
        batch.reset();
      }
    };

    /**
     * @brief    Configuration for the transpose workers
     */
    struct Config {
      size_t n_threads = 2;
      size_t n_slices = 6;
      bool use_retina = true;
    };

    /**
     * @brief    Constructor
     * @param    config Configuration parameters
     * @param    skip_banks Set of lhcb bank types to skip
     */
    TransposeWorkers(Config config, std::unordered_set<LHCb::RawBank::BankType> skip_banks) :
      m_config(config), m_slices(config.n_slices), m_slice_in_use(config.n_slices, true)
    {
      // TODO: precompute elsewhere
      for (auto const& [lhcb_type, bank_types] : Allen::bank_mapping) {
        if (skip_banks.contains(lhcb_type)) continue;
        for (auto bt : bank_types) {
          m_mapping[bt].insert(lhcb_type);
        }
      }

      // Initialize free slice queue with all slice indices
      for (size_t i = 0; i < m_config.n_slices; ++i) {
        m_free_slices.push_back(i);
      }

      // Start worker threads
      for (size_t i = 0; i < m_config.n_threads; ++i) {
        m_threads.emplace_back(&TransposeWorkers::worker_thread, this, i);
      }
    }

    /**
     * @brief    Destructor - joins all threads
     */
    ~TransposeWorkers()
    {
      m_done = true;

      // Wake up all waiting threads
      m_input_cond.notify_all();
      m_output_cond.notify_all();
      m_free_slices_cond.notify_all();

      // Join all threads
      for (auto& thread : m_threads) {
        if (thread.joinable()) {
          thread.join();
        }
      }
    }

    // Non-copyable, non-movable
    TransposeWorkers(const TransposeWorkers&) = delete;
    TransposeWorkers& operator=(const TransposeWorkers&) = delete;
    TransposeWorkers(TransposeWorkers&&) = delete;
    TransposeWorkers& operator=(TransposeWorkers&&) = delete;

    void set_input_done()
    {
      m_input_done = true;
      m_input_cond.notify_all();
    }

    /**
     * @brief      Submit a batch of prefetched events for transposition
     * @param      events The prefetched events to process
     * @return     true if submission was successful
     */
    bool submit(PrefetchedEvents&& events)
    {
      if (m_done || m_error) {
        return false;
      }
      {
        std::unique_lock<std::mutex> lock(m_input_mutex);
        m_input_queue.emplace_back(std::move(events));
        m_pending.fetch_add(1, std::memory_order_release);
      }
      m_input_cond.notify_one();
      return true;
    }

    /**
     * @brief      Get a completed slice (thread-safe, blocking)
     * @param      timeout Optional timeout in milliseconds
     * @return     Tuple of (success, done, timed_out, slice_index, n_events, odin_data)
     */
    std::tuple<bool, bool, bool, size_t, size_t, std::any> get_slice(std::optional<unsigned int> timeout = std::nullopt)
    {
      bool success = false;
      bool done = false;
      bool timed_out = false;
      size_t slice_index = 0;
      size_t n_events = 0;
      std::any odin_data;
      {
        std::unique_lock<std::mutex> lock(m_output_mutex);

        if (!m_error) {
          // Wait for a completed slice or completion
          auto wakeup = [this] {
            return !m_output_queue.empty() || m_error || m_done ||
                   (m_input_done && m_output_queue.empty() && m_pending.load(std::memory_order_acquire) == 0);
          };

          if (timeout) {
            timed_out = !m_output_cond.wait_for(lock, std::chrono::milliseconds(*timeout), wakeup);
          }
          else {
            m_output_cond.wait(lock, wakeup);
          }

          if (!m_output_queue.empty() && (!timeout || !timed_out)) {
            slice_index = m_output_queue.front();
            m_output_queue.pop_front();

            auto& slice = m_slices[slice_index];
            n_events = slice.batch.events.size();
            odin_data = std::span<unsigned const> {slice.batch.odin_data[0].data};
          }
        }
        done = m_input_done && m_output_queue.empty() && m_pending.load(std::memory_order_acquire) == 0;
      }
      return {success, done, timed_out, slice_index, n_events, odin_data};
    }

    TransposedSlice& slice(size_t slice_index) { return m_slices[slice_index]; }

    /**
     * @brief      Mark a slice as free for reuse (thread-safe)
     * @param      slice_index The slice to free
     */
    void slice_free(size_t slice_index)
    {
      m_slices[slice_index].reset();
      {
        std::unique_lock<std::mutex> lock(m_free_slices_mutex);
        if (m_slice_in_use[slice_index]) {
          m_slice_in_use[slice_index] = false;
          m_free_slices.push_back(slice_index);
        }
      }
      m_free_slices_cond.notify_one();
    }

  private:
    /**
     * @brief      Worker thread function
     * @param      thread_id The ID of this worker thread
     */
    void worker_thread([[maybe_unused]] size_t thread_id)
    {
      while (!m_done && !m_error) {
        // Get work from input queue
        PrefetchedEvents batch;
        {
          std::unique_lock<std::mutex> lock(m_input_mutex);
          m_input_cond.wait(lock, [this] {
            return !m_input_queue.empty() || m_done || m_error || (m_input_done && m_input_queue.empty());
          });
          if (m_done || m_error || (m_input_done && m_input_queue.empty())) return;
          batch = std::move(m_input_queue.front());
          m_input_queue.pop_front();
        }

        // Get a slice to write to
        size_t slice_index = 0;
        {
          std::unique_lock<std::mutex> lock(m_free_slices_mutex);
          m_free_slices_cond.wait(lock, [this] { return !m_free_slices.empty() || m_done || m_error; });
          if (m_done || m_error) return;
          slice_index = m_free_slices.front();
          m_free_slices.pop_front();
          m_slice_in_use[slice_index] = true;
        }

        // Transpose
        TransposedSlice& slice = m_slices[slice_index];
        slice.batch = std::move(batch);
        transpose_slice(slice);

        // Put the completed slice in the output queue and mark it as no longer pending
        {
          std::unique_lock<std::mutex> lock(m_output_mutex);
          m_output_queue.emplace_back(slice_index);
          m_pending.fetch_sub(1, std::memory_order_release);
        }
        m_output_cond.notify_one();
      }
    }

    void transpose_slice(TransposedSlice& slice)
    {
      const unsigned nEvents = slice.batch.events.size();
      // First collect and sort pointers to LHCb::RawBank
      for (auto bt : AllBankTypes) {
        auto& event_offsets = slice.banks[to_integral(bt)].offsets;
        event_offsets.resize(nEvents + 1, 0);

        std::vector<unsigned> bank_metadata_offsets {};
        bank_metadata_offsets.resize(nEvents + 1, 0);

        // (NEvents, NRawBanks)
        std::vector<std::vector<LHCb::RawBank const*>> rawBanks {};
        rawBanks.resize(nEvents);

        for (unsigned iEvt = 0; iEvt < nEvents; iEvt++) {
          const auto& evt = slice.batch.events[iEvt];

          for (const auto lhcb_type : m_mapping.at(bt)) {
            const auto& banks = evt.banks(lhcb_type);

            if (m_config.use_retina && lhcb_type == LHCb::Event::Enum::RawBank::BankType::VP) continue;
            if (!m_config.use_retina && lhcb_type == LHCb::Event::Enum::RawBank::BankType::VPRetinaCluster) continue;

            if (!banks.empty()) {
              // For the Calo and Rich, use the top5 bits of the source ID
              // to determine which banks to add. This will take care of
              // splitting Calo banks into ECal and HCal and Rich into
              // Rich1 and Rich2.
              std::function<bool(BankTypes bt, LHCb::RawBank const* bank)> pred;
              // In a RawBank::View all banks have the same type
              auto lhcb_it = Allen::bank_mapping.find(lhcb_type);
              if (lhcb_it != Allen::bank_mapping.end() && lhcb_it->second.size() > 1) {
                pred = check_top5;
              }
              else {
                pred = [](BankTypes, LHCb::RawBank const*) { return true; };
              }

              rawBanks[iEvt].reserve(banks.size());
              for (auto bank : banks) {
                if (pred(bt, bank)) {
                  rawBanks[iEvt].emplace_back(bank);
                  // SourceID + bankData in bytes
                  size_t paddedSize = bank->totalSize() - bank->hdrSize();
                  event_offsets[iEvt] += sizeof(uint32_t) + paddedSize;
                }
              }
            }
          } // Loop over LHCb::RawBanks

          // per event: nBanks, bankOffsets, bankData
          event_offsets[iEvt] += (2 + rawBanks[iEvt].size()) * sizeof(uint32_t);
          bank_metadata_offsets[iEvt] = rawBanks[iEvt].size();

          // Sort banks first by type and then by source ID to partition in VP and VPRetinaCluster banks.
          std::sort(rawBanks[iEvt].begin(), rawBanks[iEvt].end(), [](LHCb::RawBank const* a, LHCb::RawBank const* b) {
            return a->type() == b->type() ? (a->sourceID() < b->sourceID()) : (a->type() < b->type());
          });
        } // loop over events

        // Turn counts into offsets:
        prefix_sum(event_offsets);
        prefix_sum(bank_metadata_offsets);

        // Fill Output
        auto& bankSizes = slice.banks[to_integral(bt)].sizes;
        bankSizes.resize(2 + bank_metadata_offsets[nEvents] / 2 + nEvents);
        // The offset count uint16_t and are uint32_t
        for (unsigned iEvt = 0; iEvt < nEvents; iEvt++) {
          bankSizes[iEvt] = nEvents * 2 + bank_metadata_offsets[iEvt];
        }
        uint16_t* sizes = reinterpret_cast<uint16_t*>(bankSizes.data());

        auto& bankTypes = slice.banks[to_integral(bt)].types;
        bankTypes.resize(4 + bank_metadata_offsets[nEvents] / 4 + nEvents);
        // The offsets count uint8_t and are uint32_t
        for (unsigned iEvt = 0; iEvt < nEvents; iEvt++) {
          bankTypes[iEvt] = nEvents * 4 + bank_metadata_offsets[iEvt];
        }
        uint8_t* types = reinterpret_cast<uint8_t*>(bankTypes.data());

        auto& bankData = slice.banks[to_integral(bt)].data;
        bankData.resize(event_offsets[nEvents], 0); // TODO: check cost of memset 0

        for (unsigned iEvt = 0; iEvt < nEvents; iEvt++) {
          auto const& banks = rawBanks[iEvt];
          const unsigned nBanks = banks.size();
          const unsigned evtOffset = event_offsets[iEvt];

          *reinterpret_cast<uint32_t*>(bankData.data() + evtOffset) = nBanks;
          uint32_t* bank_offsets = reinterpret_cast<uint32_t*>(bankData.data() + evtOffset) + 1;
          bank_offsets[0] = 0;
          uint32_t* bank_data = reinterpret_cast<uint32_t*>(bankData.data() + evtOffset) + (2 + nBanks);

          int ibank = 0;
          unsigned offset = 0;
          for (auto& bank : banks) {
            const uint32_t sourceID = static_cast<uint32_t>(bank->sourceID());
            bank_data[offset++] = sourceID;

            size_t paddedSize = bank->totalSize() - bank->hdrSize();

            std::memcpy(bank_data + offset, bank->data(), paddedSize);
            offset += paddedSize / sizeof(uint32_t);

            bank_offsets[ibank + 1] = offset * sizeof(uint32_t);
            sizes[bankSizes[iEvt] + ibank] = bank->size();
            types[bankTypes[iEvt] + ibank] = static_cast<uint8_t>(bank->type());
            ibank++;
          }
        }

        slice.banks[to_integral(bt)].version = rawBanks[0].empty() ? -1 : rawBanks[0][0]->version();
      }
    }

    Config m_config;
    std::unordered_map<BankTypes, std::unordered_set<LHCb::RawBank::BankType>> m_mapping;

    // Thread management
    std::vector<std::thread> m_threads {};
    std::atomic<bool> m_done {false};
    std::atomic<bool> m_input_done {false};
    std::atomic<bool> m_error {false};

    // Input queue: prefetched events waiting to be transposed
    std::deque<PrefetchedEvents> m_input_queue;
    std::mutex m_input_mutex {};
    std::condition_variable m_input_cond {};

    // Output queue: completed slices waiting to be consumed
    std::deque<size_t> m_output_queue;
    std::mutex m_output_mutex {};
    std::condition_variable m_output_cond {};

    // Number of batches submitted but not yet placed in the output queue
    std::atomic<size_t> m_pending {0};

    // Free slice tracking
    std::deque<size_t> m_free_slices;
    std::mutex m_free_slices_mutex {};
    std::condition_variable m_free_slices_cond {};

    std::vector<TransposedSlice> m_slices;
    std::vector<bool> m_slice_in_use;
  };

  /**
   * @brief      A thread-safe pool of typed buffers with shared_ptr lifecycle management
   *
   * @details    Owns a vector of buffers of type T and provides shared_ptr<T> handles.
   *             When all shared_ptr handles to a buffer are destroyed, the buffer is
   *             automatically returned to the free pool.
   *
   * @tparam     T The type of buffer to manage
   */
  template<typename T>
  class BufferPool {
  public:
    /**
     * @brief      Construct a buffer pool
     * @param      n_buffers Number of buffers
     * @param      init Optional initialization function called for each buffer with (buffer)
     */
    BufferPool(size_t n_buffers, std::function<void(T&)> init = {})
    {
      m_buffers.resize(n_buffers);
      for (size_t i = 0; i < n_buffers; ++i) {
        if (init) init(m_buffers[i]);
        m_free_list.push_back(i);
      }
    }

    ~BufferPool()
    {
      {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_done = true;
      }
      m_cond.notify_all();
    }

    /**
     * @brief      Stop the pool, waking up any thread blocked in acquire().
     */
    void stop()
    {
      {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_done = true;
      }
      m_cond.notify_all();
    }

    // Non-copyable, non-movable (handles point into m_buffers)
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = delete;
    BufferPool& operator=(BufferPool&&) = delete;

    /**
     * @brief      Block until a buffer is available and return a handle to it
     * @return     shared_ptr<T> that returns the buffer to the pool when destroyed
     */
    std::shared_ptr<T> acquire()
    {
      size_t index = 0;
      {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return !m_free_list.empty() || m_done; });
        if (m_done) return nullptr;
        index = m_free_list.front();
        m_free_list.pop_front();
      }
      return std::shared_ptr<T>(&m_buffers[index], [this, index](T*) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_free_list.push_back(index);
        lock.unlock();
        m_cond.notify_one();
      });
    }

  private:
    bool m_done {false};
    std::vector<T> m_buffers;
    std::deque<size_t> m_free_list;
    std::mutex m_mutex;
    std::condition_variable m_cond;
  };

  struct FilePrefetcher {
    FilePrefetcher(InputProviderConfig config) : m_config {config} {}

    virtual ~FilePrefetcher() { stopAndJoin(); }

    void start()
    {
      m_thread = std::make_unique<std::thread>([this] { prefetch(); });
    }

    virtual void prefetch() = 0;

    size_t events_per_slice() const { return m_config.events_per_slice; }
    std::optional<size_t> const& n_events() const { return m_config.n_events; }

    bool done() const { return m_done; }
    bool read_error() const { return m_read_error; }

  protected:
    // Derived prefetchers must wake blocked reads and join before destroying
    // the resources used by their prefetch thread.
    void stopAndJoin()
    {
      m_done = true;
      if (m_thread && m_thread->joinable()) {
        m_thread->join();
      }
    }

    // Atomics to flag errors and completion
    std::atomic<bool> m_done = false;
    std::atomic<bool> m_read_error = false;

    InputProviderConfig m_config {};

  private:
    std::unique_ptr<std::thread> m_thread;
  };
} // namespace Allen
