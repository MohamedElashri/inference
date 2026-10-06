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
// Gaudi input-provider service reading MDF (or ROOT) files: prefetches raw
// events, transposes them into per-bank-type slices and serves slices to Allen
// through the get_slice/slice_free interface.
// ----------------------------------------------------------------------------
#pragma once

#include <thread>
#include <vector>
#include <array>
#include <deque>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <condition_variable>
#include <optional>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <Logger.h>
#include <InputProvider.h>
#include <mdf_header.hpp>
#include <sourceid.h>
#include <read_mdf.hpp>
#include <write_mdf.hpp>
#include <Event/RawBankType.h>
#include "BankMapping.h"

#include <SliceUtils.h>
#include <Transpose.h>
#include <ODINBank.cuh>

#include <BackendCommon.h>

#include "TransposeWorkers.h"
#include "MDFPrefetch.h"
#ifndef ALLEN_STANDALONE
#include "ROOTPrefetch.h"
#endif

namespace {
  using namespace Allen::Units;
  using namespace std::string_literals;
} // namespace

/**
 * @brief      Provide transposed events from MDF files
 *
 * @details    The provider has three main components
 *             - a prefetch thread to read from the current input
 *               file into prefetch buffers
 *             - N transpose threads that read from prefetch buffers
 *               and fill the per-bank-type slices with transposed sets
 *               of banks and the offsets to individual bank inside a
 *               given set
 *             - functions to obtain a transposed slice and declare it
 *               for refilling
 *
 *             Access to prefetch buffers and slices is synchronised
 *             using mutexes and condition variables.
 *
 * @param      Number of slices to fill
 * @param      Number of events per slice
 * @param      MDF filenames
 * @param      Configuration struct
 *
 */
#ifndef ALLEN_STANDALONE
#include <Gaudi/Parsers/Factory.h>
#include <GaudiKernel/DataObjID.h>
#include <GaudiKernel/DataObjectHandle.h>
#include <GaudiKernel/Service.h>

namespace Allen {
  enum class InputFileType { MDF, ROOT };
  std::string toString(InputFileType type);
  std::ostream& toStream(InputFileType type, std::ostream& stream);
  StatusCode parse(InputFileType& type, std::string_view input);
} // namespace Allen

namespace Gaudi::Parsers {
  StatusCode parse(std::map<std::string, DataObjID>& m, std::string_view in)
  {
    m.clear();

    // the first element is branchName and the second on is tesPath
    std::map<std::string, std::string> ms;
    return parse(ms, in).andThen([&m, &ms]() -> StatusCode {
      try {
        std::ranges::transform(ms, std::inserter(m, m.end()), [](const auto& p) {
          DataObjID id;
          parse(id, p.second).orThrow("bad parse");
          return std::pair {p.first, id};
        });
        return StatusCode::SUCCESS;
      } catch (GaudiException const& e) {
        return e.code();
      }
    });
  };
} // namespace Gaudi::Parsers

class MDFProvider final : public extends<Service, IInputProviderSvc> {
public:
  using extends::extends;

  Gaudi::Property<size_t> m_nslices {this, "NSlices", 6};
  Gaudi::Property<size_t> m_events_per_slice {this, "EventsPerSlice", 1000};
  Gaudi::Property<std::vector<std::string>> m_connections {this, "Connections", {}, "List of .mdf files"};
  Gaudi::Property<long> m_nevents {this, "EvtMax", -1};

  std::unordered_set<BankTypes> m_bank_types;

  Gaudi::Property<bool> m_check_checksum {this, "CheckChecksum", false, "verify MDF checksums"};
  Gaudi::Property<size_t> m_n_transpose_threads {this, "TransposeThreads", 2, "number of transpose threads"};
  Gaudi::Property<size_t> m_n_loops {this, "NLoops", 0, "number of loops over the input files"};
  Gaudi::Property<bool> m_split_by_run {this, "SplitByRun", false, "Whether to split slices by run number"};
  Gaudi::Property<bool> m_use_retina {this, "UseRetina", true, "Use Retina RawBanks instead of Super-pixels"};

  Gaudi::Property<Allen::InputFileType> m_input_type {this, "InputType", Allen::InputFileType::MDF, "MDF or ROOT"};

  Gaudi::Property<std::string> m_eventTreeName {
    this,
    "EventTreeName",
    "Event",
    "Name of the tree containing Events in the Root files"};
  // The twin of m_eventBranchesMap, to ducoment the index of branches in other vectors
  std::vector<std::pair<std::string, DataObjID>> m_eventBranches;
  Gaudi::Property<std::map<std::string, DataObjID>> m_eventBranchesMap {
    this,
    "EventBranches",
    {},
    [this](auto const&) {
      std::set<DataObjID> seen;
      m_eventBranches.clear();
      for (const auto& [branch, objID] : m_eventBranchesMap.value()) {
        // check the condition of bijectective
        // only allow one TES path used for one time in the memory
        auto [it, inserted] = seen.insert(objID);
        if (!inserted) {
          throw GaudiException(
            "Non-invertible EventBranchesMap detected: Branch " + branch, "RootIOAlgBase", StatusCode::FAILURE);
        }

        // create the twin of m_eventBranchesMap
        m_eventBranches.emplace_back(branch, objID);
      }
      debug() << "Updated EventBranches: " << m_eventBranches.size() << " entries." << endmsg;
    },
    "Map branch property name -> TES location to be retrieved from the Root files"};

  StatusCode initialize() override
  {
    auto sc = Service::initialize();
    if (!sc.isSuccess()) return sc;

    std::optional<size_t> n_events = std::nullopt;
    if (m_nevents.value() >= 0) n_events = static_cast<size_t>(m_nevents.value());

    m_bank_types = AllBankTypes; /// All banks

    m_config = InputProviderConfig {
      .check_checksum = m_check_checksum.value(), // verify MDF checksums
      .n_slices = m_nslices,
      .n_events = n_events,
      .events_per_slice = m_events_per_slice,
      .n_transpose_threads = m_n_transpose_threads,       // number of transpose threads
      .events_per_buffer = (m_events_per_slice + 9) / 10, // number of events per read buffer
      .n_loops = m_n_loops,                               // number of loops over the input files
      .split_by_run = m_split_by_run.value(),             // Whether to split slices by run number
      .use_ROOT_prefetcher = (m_input_type == Allen::InputFileType::ROOT),
      .use_retina = m_use_retina.value()};

    init_input(m_nslices, m_events_per_slice, m_bank_types, IInputProvider::Layout::Allen, n_events);
    init();
    return sc;
  }

  StatusCode finalize() override
  {
    // Stop and join the background prefetch and transpose threads. Gaudi
    // services are finalized but not always destructed, so joining here avoids
    // ThreadSanitizer reporting these threads as leaked.
    m_prefetch_thread.reset();
    m_transpose_workers.reset();
    m_buffer_pool.reset();
    return Service::finalize();
  }

  std::vector<DataObject*> getEventBranches(size_t const slice_index, unsigned const event) const override
  {
    auto& slice = m_transpose_workers->slice(slice_index);
    return slice.batch.branches[event];
  }

  LHCb::RawEvent getRawEvent(size_t const slice_index, unsigned const event) const override
  {
    auto& slice = m_transpose_workers->slice(slice_index);
    LHCb::RawEvent raw_event; // RawEvent don't have a copy constructor, so copy manually:
    for (LHCb::RawBank const* bank : slice.batch.events[event].banks()) {
      raw_event.adoptBank(bank, false);
    }
    return raw_event;
  }

  LHCb::IO::InputFileManifest getInputFileManifest(size_t const slice_index, unsigned const event) const override
  {
    if (m_input_type != Allen::InputFileType::ROOT) return IInputProviderSvc::getInputFileManifest(slice_index, event);
    return m_transpose_workers->slice(slice_index).batch.input_file_manifests.at(event);
  }

  LHCb::ODIN getODIN(size_t const slice_index) const override
  {
    return m_transpose_workers->slice(slice_index).batch.odin_data[0];
  }

#else
class MDFProvider final : public InputProvider {
  // File names to read
  std::vector<std::string> m_connections;

public:
  MDFProvider(
    size_t n_slices,
    size_t events_per_slice,
    std::optional<size_t> n_events,
    std::vector<std::string> connections,
    std::unordered_set<BankTypes> const& bank_types,
    InputProviderConfig config) :
    m_connections {std::move(connections)},
    m_config {config}
  {
    init_input(n_slices, events_per_slice, bank_types, IInputProvider::Layout::Allen, n_events);
    init();
  }
#endif

  void init();

  void startPrefetcher() const override { m_prefetch_thread->start(); }

  /**
   * @brief      Obtain event IDs of events stored in a given slice
   *
   * @param      slice index
   *
   * @return     EventIDs of events in given slice
   */
  EventIDs event_ids(size_t slice_index, std::optional<size_t> first = {}, std::optional<size_t> last = {})
    const override;

  /**
   * @brief      Obtain event mask in a given slice (ODIN error)
   *
   * @param      slice index
   *
   * @return     event mask in given slice
   */
  std::vector<char> event_mask(size_t slice_index) const override;

  /**
   * @brief      Obtain banks from a slice
   *
   * @param      BankType
   * @param      slice index
   *
   * @return     Banks and their offsets
   */
  BanksAndOffsets banks(BankTypes bank_type, size_t slice_index) const override;

  /**
   * @brief      Get a slice that is ready for processing; thread-safe
   *
   * @param      optional timeout
   *
   * @return     (good slice, input done, timed out, slice index, number of events in slice)
   */
  std::tuple<bool, bool, bool, size_t, size_t, std::any> get_slice(std::optional<unsigned int> timeout = {}) override;

  /**
   * @brief      Declare a slice free for reuse; thread-safe
   *
   * @param      slice index
   *
   * @return     void
   */
  void slice_free(size_t slice_index) override;

  void event_sizes(
    size_t const slice_index,
    std::span<unsigned int const> const selected_events,
    std::span<size_t> sizes) const override;

  void copy_banks(size_t const slice_index, unsigned int const event, std::span<char> output_buffer) const override;

private:
  // Configuration struct
  InputProviderConfig m_config;

  std::unique_ptr<Allen::BufferPool<Allen::ReadBuffer>> m_buffer_pool {nullptr};
  std::unique_ptr<Allen::TransposeWorkers> m_transpose_workers {nullptr};
  std::unique_ptr<Allen::FilePrefetcher> m_prefetch_thread {nullptr};
};
