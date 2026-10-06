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
// ROOT-file prefetcher: reads raw events and requested event branches from ROOT
// input files and submits prefetched batches to the transpose workers.
// ----------------------------------------------------------------------------
#pragma once

#include "TransposeWorkers.h"

#include <IOAlgorithms/RootIOHandler.h>
#include <GaudiKernel/Service.h>

struct ROOTPrefetcher : Allen::FilePrefetcher {
  ROOTPrefetcher(
    std::vector<std::string> connections,
    Allen::TransposeWorkers* transpose_workers,
    Allen::BufferPool<Allen::ReadBuffer>*,
    InputProviderConfig config,
    Service* provider,
    std::string eventTreeName,
    std::vector<std::pair<std::string, DataObjID>> const& eventBranches) :
    Allen::FilePrefetcher(config),
    m_transpose_workers {transpose_workers}, m_provider {provider}, m_ioHandler {*provider, connections},
    m_batch_pool {config.n_slices}
  {
    SmartIF<IService> service = provider->service("IncidentSvc/IncidentSvc");
    IIncidentSvc* incidentSvc = service.as<IIncidentSvc>();
    if (!incidentSvc) {
      throw GaudiException {
        "ROOTPrefetcher", "Unable to localize interface IIncidentSvc from service ROOTPrefetcher", StatusCode::FAILURE};
    }
    bool allowMissingInput = true;
    m_ioHandler.initialize(
      *provider, incidentSvc, config.events_per_buffer, 0, 0, eventTreeName, eventBranches, allowMissingInput, false);
  }

  ~ROOTPrefetcher() override
  {
    m_done = true;
    m_batch_pool.stop();
    stopAndJoin();
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

    // Loop while there are no errors and the flag to exit is not set
    while (!m_done && !m_read_error && (!to_read || *to_read > 0)) {
      try {
        // Unlike MDF input, ROOT input does not acquire a bounded ReadBuffer.
        // Keep a token with each batch until slice_free releases its buffers,
        // so the reader cannot queue the entire remaining dataset in memory.
        if (batch.empty()) {
          auto token = m_batch_pool.acquire();
          if (!token) break;
          batch.buffers.push_back(std::move(token));
        }
        EventContext ctx {}; // fake event context, the handler ignore it anyway..
        auto&& [eventData, buffer] = m_ioHandler.next(*m_provider, ctx);

        LHCb::RawEvent* evt = nullptr;
        for (DataObject* obj : eventData.first) {
          evt = dynamic_cast<LHCb::RawEvent*>(obj);
          if (evt) break;
        }

        LHCb::RawEvent raw_event; // RawEvent don't have a copy constructor, so copy manually:
        for (LHCb::RawBank const* bank : evt->banks()) {
          raw_event.adoptBank(bank, false);
        }

        bool has_odin = false; // TODO: odin, split on run change ?
        LHCb::ODIN odin {};

        batch.events.push_back(std::move(raw_event));
        if (has_odin) {
          batch.odin_data.push_back(odin);
          batch.event_ids.emplace_back(odin.runNumber(), odin.eventNumber());
          batch.event_mask.push_back(true);
        }
        else {
          batch.odin_data.emplace_back();
          batch.event_ids.emplace_back(0, 0);
          batch.event_mask.push_back(true);
        }
        batch.buffers.push_back(buffer); // TODO: deduplicate shared pointers
        // Slices may span files. These views borrow from the corresponding buffer,
        // retained above until the slice is released after event processing.
        batch.input_file_manifests.push_back(buffer->inputFileManifest());

        batch.branches.push_back(eventData.first);

        // Update event count
        if (to_read) {
          *to_read -= 1;
        }

        // Flush if slice is full
        if (batch.size() >= eps) {
          if (!flush_batch()) break;
        }
      } catch (LHCb::IO::EndOfInput const&) {
        break;
      }
    }

    // Final flush
    flush_batch();

    m_done = true;
    m_transpose_workers->set_input_done();
  }

private:
  Allen::TransposeWorkers* m_transpose_workers {nullptr};

  /// Pointers to services
  Service* m_provider {nullptr};
  LHCb::IO::RootIOHandler<false> m_ioHandler;

  Allen::BufferPool<char> m_batch_pool;
};
