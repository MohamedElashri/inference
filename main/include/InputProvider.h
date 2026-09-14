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

#include <unordered_set>
#include <map>
#include <vector>
#include <cmath>
#include <mutex>
#include <optional>
#include <any>
#include <span>

#include <Event/ODIN.h>
#include <Event/RawBankType.h>

#include "Logger.h"
#include "BankTypes.h"
#include "Common.h"
#include "AllenUnits.h"

/**
 * @brief      Configuration parameters for the InputProvider
 *
 */
struct InputProviderConfig {
  // check the MDF checksum if it is available
  bool check_checksum = false;

  size_t n_slices = 1;

  std::optional<size_t> n_events = std::nullopt;

  size_t events_per_slice = 1000;

  // number of transpose threads
  size_t n_transpose_threads = 5;

  // default of events per prefetch buffer
  size_t events_per_buffer = 1200;

  // number of loops over input data
  size_t n_loops = 0;

  bool split_by_run = false;

  std::unordered_set<LHCb::Event::Enum::RawBank::BankType> skip_banks {};

  bool use_ROOT_prefetcher = false;

  bool use_retina = true;
};

class IInputProvider {
public:
  enum class Layout { Allen, MEP };

  struct BufferStatus { // Used by MEPProvider, but really should be implementation details..
    bool writable = true;
    int work_counter = 0;
    std::vector<std::tuple<size_t, size_t>> intervals;
    size_t index = 0;
  };

  /// Desctructor
  virtual ~IInputProvider() {}

  /**
   * @brief      Are slices provided in MEP layout or not
   *
   * @return     layout
   */
  virtual Layout layout() const = 0;

  /**
   * @brief      Get the maximum number of events per slice
   *
   * @return     number of events per slice
   */
  virtual size_t events_per_slice() const = 0;

  /**
   * @brief      Get the number of slices
   *
   * @return     number of slices
   */
  virtual size_t n_slices() const = 0;

  /**
   * @brief      Get event ids in a given slice
   *
   * @param      slice index
   *
   * @return     event ids
   */
  virtual EventIDs event_ids(
    size_t slice_index,
    std::optional<size_t> first = std::nullopt,
    std::optional<size_t> last = std::nullopt) const = 0;

  /**
   * @brief      Get event mask in a given slice (ODIN erro bank)
   *
   * @param      slice index
   *
   * @return     event mask
   */
  virtual std::vector<char> event_mask(size_t slice_index) const = 0;

  /**
   * @brief      Indicate a slice is free for filling
   *
   * @param      slice index
   */
  virtual void slice_free(size_t slice_index) = 0;

  /**
   * @brief      Get a slice with n events
   *
   * @param      optional timeout in ms to wait for slice
   *
   * @return     tuple of (success, eof, timed_out, slice_index, n_filled)
   */
  virtual std::tuple<bool, bool, bool, size_t, size_t, std::any> get_slice(
    std::optional<unsigned int> timeout = std::nullopt) = 0;

  /**
   * @brief      Get banks and offsets of a given type
   *
   * @param      bank type requested
   *
   * @return     spans spanning bank and offset memory
   */
  virtual BanksAndOffsets banks(BankTypes bank_type, size_t slice_index) const = 0;

  virtual void event_sizes(
    size_t const slice_index,
    std::span<unsigned int const> const selected_events,
    std::span<size_t> sizes) const = 0;

  virtual void copy_banks(size_t const slice_index, unsigned int const event, std::span<char> buffer) const = 0;

  virtual void startPrefetcher() const {}

  virtual bool release_buffers() { return true; }
};

class InputProvider : public IInputProvider {
public:
  InputProvider() = default;

  InputProvider(
    size_t n_slices,
    size_t events_per_slice,
    std::unordered_set<BankTypes> const& types,
    Layout layout,
    std::optional<size_t> n_events)
  {
    init_input(n_slices, events_per_slice, types, layout, n_events);
  }

  /// Descturctor
  virtual ~InputProvider() = default;

  /**
   * @brief      Are slices provided in MEP layout or not
   *
   * @return     layout
   */
  Layout layout() const override { return m_layout; }

  /**
   * @brief      Get the bank types filled by this provider
   *
   * @return     unordered set of bank types
   */
  std::unordered_set<BankTypes> const& types() const { return m_types; }

  /**
   * @brief      Get the number of slices
   *
   * @return     number of slices
   */
  size_t n_slices() const override { return m_nslices; }

  /**
   * @brief      Get the maximum number of events per slice
   *
   * @return     number of events per slice
   */
  size_t events_per_slice() const override { return m_events_per_slice; }

  std::optional<size_t> const& n_events() const { return m_nevents; }

protected:
  void init_input(
    size_t n_slices,
    size_t events_per_slice,
    std::unordered_set<BankTypes> types,
    Layout layout,
    std::optional<size_t> n_events)
  {
    m_nslices = n_slices;
    m_events_per_slice = events_per_slice;
    m_types = types;
    m_layout = layout;
    m_nevents = n_events;
  }

  template<typename MSG>
  void debug_output(const MSG& msg, std::optional<size_t> const thread_id = std::nullopt) const
  {
    if (logger::verbosity() >= logger::debug) {
      std::unique_lock<std::mutex> lock {m_output_mut};
      debug_cout << (thread_id ? std::to_string(*thread_id) + " " : std::string {}) << msg << "\n";
    }
  }

protected:
  // MEP layout
  Layout m_layout = Layout::Allen;

  // Number of slices to be provided
  size_t m_nslices = 0;

  // Number of events per slice
  size_t m_events_per_slice = 0;

  // Optional total number of events to be provided
  std::optional<size_t> m_nevents = std::nullopt;

  // BankTypes provided by this provider
  std::unordered_set<BankTypes> m_types;

  // Mutex for ordered debug output
  mutable std::mutex m_output_mut;
};

#ifndef ALLEN_STANDALONE
#include "GaudiKernel/IInterface.h"
#include "Event/RawEvent.h"
class GAUDI_API IInputProviderSvc : public extend_interfaces<IInterface>, public InputProvider {
public:
  // Return the interface ID
  DeclareInterfaceID(IInputProviderSvc, 0, 0);
  virtual ~IInputProviderSvc() = default;

  virtual std::vector<DataObject*> getEventBranches(size_t const, unsigned const) const { return {}; }

  /**
   * @brief Single event interface to inputs
   */
  virtual LHCb::RawEvent getRawEvent(size_t const slice_index, unsigned int const event) const = 0;

  virtual LHCb::ODIN getODIN(size_t const slice_index) const = 0;
};
#else
using IInputProviderSvc = InputProvider;
#endif
