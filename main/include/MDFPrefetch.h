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
// MDF-file prefetcher: reads raw events from MDF input files and submits
// prefetched batches to the transpose workers.
// ----------------------------------------------------------------------------
#pragma once

#include "TransposeWorkers.h"

struct MDFPrefetcher : Allen::FilePrefetcher {
  MDFPrefetcher(
    std::vector<std::string> connections,
    Allen::TransposeWorkers* transpose_workers,
    Allen::BufferPool<Allen::ReadBuffer>* buffer_pool,
    InputProviderConfig config) :
    Allen::FilePrefetcher(config),
    m_connections {std::move(connections)}, m_transpose_workers {transpose_workers}, m_buffer_pool {buffer_pool}
  {
    // Initialize the current input filename
    m_current = m_connections.begin();
    // Reserve 1MB for decompression
    m_compress_buffer.reserve(1u * MB);
  }

  ~MDFPrefetcher() override
  {
    // Wake the prefetch thread if it is blocked in BufferPool::acquire().
    // The base FilePrefetcher destructor then sets m_done and joins the thread.
    if (m_buffer_pool) {
      m_buffer_pool->stop();
    }
  }

  bool open_file()
  {
    bool good = false;

    // Check if there are still files available
    while (!good) {
      // If looping on input is configured, do it
      if (m_current == m_connections.end()) {
        if (++m_loop < m_config.n_loops) {
          m_current = m_connections.begin();
        }
        else {
          break;
        }
      }

      if (m_input && m_input->good) m_input->close();

      m_input = MDF::open(m_current->c_str(), O_RDONLY);
      if (m_input->good) {
        // read the first header, needed by subsequent calls to read_events
        ssize_t n_bytes = m_input->read(reinterpret_cast<char*>(&m_header), mdf_header_size);
        good = (n_bytes > 0);
      }

      if (good) {
        auto i = std::distance(m_connections.begin(), m_current) + 1;
        info_cout << "Opened " << *m_current << " (" << i << "/" << m_connections.size() << ")\n";
      }
      else {
        error_cout << "Failed to open " << *m_current << " " << strerror(errno) << "\n";
        m_read_error = true;
        return false;
      }
      ++m_current;
    }
    return good;
  }

  void prefetch() override
  {
    auto to_read = m_config.n_events;
    size_t eps = m_config.events_per_slice;

    // The batch we're building incrementally
    Allen::TransposeWorkers::PrefetchedEvents batch;
    int current_run = -1;

    // Helper to flush the current batch to transpose workers
    auto flush_batch = [&]() {
      if (!batch.empty()) {
        if (!m_transpose_workers->submit(std::move(batch))) {
          error_cout << "Failed to submit batch to transpose workers\n";
          m_read_error = true;
          return false;
        }
        batch = {};
        current_run = -1;
      }
      return true;
    };

    bool no_more_files = false;

    // Loop while there are no errors and the flag to exit is not set
    while (!m_done && !m_read_error && !no_more_files && (!to_read || *to_read > 0)) {

      // Get a buffer to fill. The buffer is grown (resized) as needed below,
      // so a full slice can be accumulated in a single buffer.
      auto read_buffer_ptr = m_buffer_pool->acquire();
      if (!read_buffer_ptr) break;

      // ---- Fill phase ----
      // Read raw event data into the buffer, recording only offsets. No
      // RawEvent objects (and hence no pointers into the buffer) are created
      // here, so the buffer can be resized safely while filling.
      std::vector<size_t> event_offsets;
      std::vector<size_t> event_lengths;
      std::vector<size_t> event_n_blocks;
      size_t buffer_offset = 0;
      size_t n_events_read = 0;

      while (!no_more_files && !m_read_error && (!to_read || *to_read > 0) && n_events_read < eps) {

        // Try to open a file if needed
        if (!m_input) {
          if (!open_file()) {
            no_more_files = true;
            break;
          }
        }

        // Estimate space needed for next event
        const auto event_size = MDF::read_banks_buffer_size(m_header);

        // Grow the buffer if the next event doesn't fit
        if (buffer_offset + event_size > read_buffer_ptr->event_buffer.size()) {
          auto new_size = read_buffer_ptr->event_buffer.size() * 3 / 2;
          if (new_size < buffer_offset + event_size) {
            new_size = buffer_offset + event_size;
          }
          read_buffer_ptr->event_buffer.resize(new_size);
        }

        // Read banks from file
        std::span<char> buffer_span {
          read_buffer_ptr->event_buffer.data() + buffer_offset,
          static_cast<size_t>(read_buffer_ptr->event_buffer.size() - buffer_offset)};

        std::span<const char> bank_span;
        bool error = false, eof = false;
        std::tie(eof, error, bank_span) =
          MDF::read_banks(*m_input, m_header, std::move(buffer_span), m_compress_buffer, m_config.check_checksum);

        if (error) {
          m_read_error = true;
          break;
        }

        if (eof) {
          // No more (complete) events in this file; move to the next one.
          info_cout << "Cannot read more data (Header). End-of-File reached.\n";
          if (m_input && m_input->good) m_input->close();
          m_input.reset();
          continue;
        }

        // Light parse to locate the payload (skipping the DAQ status bank) and
        // determine the number of sub-events (for TAE events).
        const char* payload = bank_span.data();
        auto const* first_bank = reinterpret_cast<LHCb::RawBank const*>(payload);
        if (first_bank->magic() != LHCb::RawBank::MagicPattern) {
          error_cout << "Bad magic in first bank.\n";
          m_read_error = true;
          break;
        }

        // Skip DAQ status bank if present
        if (first_bank->type() == LHCb::RawBank::BankType::DAQ && first_bank->version() == DAQ_STATUS_BANK) {
          payload += first_bank->totalSize();
          first_bank = reinterpret_cast<LHCb::RawBank const*>(payload);
        }

        // Check for TAE event
        bool is_tae = (first_bank->type() == LHCb::RawBank::BankType::TAEHeader);
        size_t n_blocks = is_tae ? first_bank->size() / sizeof(int) / 3 : 1;

        // Record this event for the build phase
        event_offsets.push_back(payload - read_buffer_ptr->event_buffer.data());
        event_lengths.push_back(bank_span.data() + bank_span.size() - payload);
        event_n_blocks.push_back(n_blocks);
        n_events_read += n_blocks;

        buffer_offset += bank_span.size();

        // Read next header
        ssize_t n_bytes = m_input->read(reinterpret_cast<char*>(&m_header), mdf_header_size);
        if (n_bytes == 0) {
          info_cout << "Cannot read more data (Header). End-of-File reached.\n";
          if (m_input && m_input->good) m_input->close();
          m_input.reset();
        }
        else if (n_bytes < 0) {
          error_cout << "Failed to read header " << strerror(errno) << "\n";
          m_read_error = true;
          break;
        }

        // Update event count
        if (to_read) {
          *to_read -= std::min(*to_read, n_blocks);
        }
      }

      if (m_read_error) break;

      // ---- Build phase ----
      // Parse the raw data into RawEvents. This happens only after the buffer
      // is filled and no longer resized, so the RawBank pointers adopted by the
      // RawEvents remain valid.
      for (size_t i = 0; i < event_offsets.size() && !m_read_error; ++i) {
        const char* payload = read_buffer_ptr->event_buffer.data() + event_offsets[i];
        const char* record_end = payload + event_lengths[i];
        auto const* first_bank = reinterpret_cast<LHCb::RawBank const*>(payload);
        bool is_tae = (first_bank->type() == LHCb::RawBank::BankType::TAEHeader);
        size_t n_blocks = event_n_blocks[i];

        // Process each sub-event
        for (size_t block_i = 0; block_i < n_blocks; ++block_i) {
          const char* event_start;
          size_t event_length;

          if (is_tae) {
            int const* block = reinterpret_cast<int const*>(first_bank);
            block += 2; // skip bank header
            block += 3 * block_i;
            block++; // skip bx_offset
            int offset = *block++;
            int size = *block++;
            // The sub-event offsets in the TAE header are relative to the end
            // of the TAE header bank, not to the start of the banks.
            event_start = payload + first_bank->totalSize() + offset;
            event_length = size;
          }
          else {
            event_start = payload;
            event_length = record_end - payload;
          }

          // Build RawEvent
          LHCb::RawEvent raw_event;
          LHCb::ODIN odin;
          bool has_odin = false;
          bool odin_error = false;

          const std::byte* start = reinterpret_cast<const std::byte*>(event_start);
          const std::byte* end = start + event_length;

          while (start < end) {
            LHCb::RawBank const* bank = reinterpret_cast<LHCb::RawBank const*>(start);
            if (bank->magic() != LHCb::RawBank::MagicPattern) {
              error_cout << "Bad magic pattern in bank\n";
              m_read_error = true;
              break;
            }

            // Extract ODIN
            auto allen_type = sd_from_bank_type(bank); // sd_from_sourceID(bank); // TODO: select correct method
            if (allen_type == BankTypes::ODIN) {
              odin_error = (bank->type() >= LHCb::RawBank::BankType::DaqErrorFragmentThrottled);
              if (!odin_error) {
                has_odin = true;
                odin = MDF::decode_odin(bank->range<unsigned>(), bank->version());
              }
            }
            raw_event.adoptBank(bank, false);
            start += bank->totalSize();
          }

          if (m_read_error) break;

          // Run number splitting
          if (m_config.split_by_run && has_odin) {
            int run = odin.runNumber();
            if (current_run == -1) {
              current_run = run;
            }
            else if (run != current_run) {
              if (!flush_batch()) break;
              current_run = run;
            }
          }

          // Add to batch
          batch.events.push_back(std::move(raw_event));
          if (has_odin) {
            batch.odin_data.push_back(odin);
            batch.event_ids.emplace_back(odin.runNumber(), odin.eventNumber());
            batch.event_mask.push_back(!odin_error);
          }
          else {
            batch.odin_data.emplace_back();
            batch.event_ids.emplace_back(0, 0);
            batch.event_mask.push_back(false);
          }

          // Track buffer ownership
          if (batch.buffers.empty() || batch.buffers.back() != read_buffer_ptr) {
            batch.buffers.push_back(read_buffer_ptr);
          }
        }
      }

      if (m_read_error) break;

      // Submit the batch. A full slice is submitted here; a partial batch
      // (e.g. on EOF) is submitted as well.
      if (!flush_batch()) break;
    }

    // Final flush (in case a partial batch remains)
    flush_batch();

    m_done = true;
    m_transpose_workers->set_input_done();
  }

private:
  std::vector<std::string> m_connections {};
  Allen::TransposeWorkers* m_transpose_workers {nullptr};
  Allen::BufferPool<Allen::ReadBuffer>* m_buffer_pool {nullptr};

  // Buffer to store data read from file if banks are compressed. The
  // decompressed data will be written to the buffers
  std::vector<char> m_compress_buffer {};

  // Storage to read the header into for each event
  LHCb::MDFHeader m_header {};

  // Storage for the currently open file
  std::optional<Allen::IO> m_input = std::nullopt;

  // Iterator that points to the filename of the currently open file
  std::vector<std::string>::iterator m_current;

  // Input data loop counter
  size_t m_loop = 0;
};
