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

#ifndef ALLEN_STANDALONE
#include "GaudiKernel/IInterface.h"
#include "GaudiKernel/Service.h"

#include "SPSCRingBuffer.h"
#include "OutputManager.h"
#include "SingleEventPassthrough.cuh"

#include <thread>

class GAUDI_API IOutputWriter : public extend_interfaces<IInterface> {
public:
  // Return the interface ID
  DeclareInterfaceID(IOutputWriter, 0, 0);
  virtual ~IOutputWriter() = default;

  virtual void cancel() {}

  virtual void output_done() {}

  virtual const SingleEventPassthrough* singleEventPassthrough() const = 0;
};

namespace {
  void set_current_thread_name(const std::string& thread_name)
  {
#ifdef __linux__
    pthread_setname_np(pthread_self(), thread_name.c_str());
#else
    pthread_setname_np(thread_name.c_str());
#endif
  }
} // namespace

class OutputWriter : public extends<Service, IOutputWriter> {
public:
  using extends::extends;

  Gaudi::Property<unsigned> m_nStreams {this, "NStreams", 1, "Number of parallel independent stream (sequences)."};
  Gaudi::Property<unsigned> m_rbCapacity {this, "RBCapacity", 100 * 1024 * 1024, "Ring buffer capacity per stream."};
  Gaudi::Property<std::string> m_outputConnection {
    this,
    "OutputConnection",
    "",
    "Output file name or ZeroMQ connection string."};

  Gaudi::Property<unsigned> m_tck {this, "TCK", 0, ""};
  Gaudi::Property<unsigned> m_task_id {this, "TaskId", 0, ""};
  Gaudi::Property<std::map<std::string, uint32_t>> m_routingbit_map {
    this,
    "routingbit_map",
    {},
    "mapping of expressions to routing bits"};
  Gaudi::Property<bool> m_do_checksum {this, "DoChecksum", false, ""};

  std::unique_ptr<SingleEventPassthrough> m_single_event_passthrough {nullptr};

  virtual std::tuple<bool, size_t> output_selected_events(SPSCRingBuffer* ring_buffer)
  {
    ring_buffer->consume(); // consume data to avoid blocking the queue
    ring_buffer->release();
    return {true, 0};
  };

  StatusCode initialize() override
  {
    StatusCode sc = Service::initialize();
    if (!sc.isSuccess()) {
      error() << "Failed to initialize Service Base class." << endmsg;
      return StatusCode::FAILURE;
    }

    info() << "Initialize" << endmsg;

    // Init Output Manager
    OutputManager::get()->init(m_nStreams.value(), m_rbCapacity.value());

    unsigned passthrough_rbs = 0;
    const std::string passthrough_line = "Hlt1PassthroughLargeEvent";
    for (auto [expr, bit] : m_routingbit_map.value()) {
      std::smatch result;
      if (std::regex_match(passthrough_line, result, std::regex {expr})) {
        passthrough_rbs |= 1u << bit;
      }
    }

    m_single_event_passthrough = std::make_unique<SingleEventPassthrough>(
      m_tck.value(), m_task_id.value(), passthrough_rbs, m_do_checksum.value());
    m_single_event_passthrough->activateMonitoring(this);

    m_thread = std::thread(&OutputWriter::output_loop, this);

    return StatusCode::SUCCESS;
  }

  const SingleEventPassthrough* singleEventPassthrough() const override { return m_single_event_passthrough.get(); }

  StatusCode finalize() override
  {
    info() << "Finalize" << endmsg;

    m_done = true;
    if (m_thread.joinable()) {
      m_thread.join();
    }

    return Service::finalize();
  }

private:
  void check_and_write()
  {
    for (unsigned i = 0; i < m_nStreams.value(); i++) {
      SPSCRingBuffer* buffer = OutputManager::get()->buffer(i);
      auto [success, count] = this->output_selected_events(buffer);
      if (!success) {
        error() << "Failed to write output data" << endmsg;
      }
      else {
        debug() << "Wrote " << count << " events from stream " << i << endmsg;
      }
    }
  }

  void output_loop()
  {
    set_current_thread_name("OutputWriter");
    while (!m_done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      check_and_write();
    }
    check_and_write();
  }

  std::atomic<bool> m_done {false};
  std::thread m_thread {};
};

#endif
