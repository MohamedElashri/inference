/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// ----------------------------------------------------------------------------
// Concrete OutputWriter implementations: FileOutputWriter writes selected events
// to a file and ZMQOutputWriter sends them over ZeroMQ. Both drain the producer
// ring buffers of selected events.
// ----------------------------------------------------------------------------

#include <atomic>
#include <thread>

#include "IOutputWriter.h"
#include "OutputHandler.h"
#include "OutputManager.h"
#include "FileWriter.h"
#include "ZMQOutputSender.h"

#include <GaudiKernel/Service.h>

class FileOutputWriter final : public OutputWriter {
public:
  using OutputWriter::OutputWriter;

  StatusCode initialize() override
  {
    m_output_handler = std::make_unique<FileWriter>(m_outputConnection.value());
    return OutputWriter::initialize();
  }

  StatusCode finalize() override
  {
    auto sc = OutputWriter::finalize();
    if (!sc.isSuccess()) return sc;
    m_output_handler.reset();
    return sc;
  }

  std::tuple<bool, size_t> output_selected_events(SPSCRingBuffer* ring_buffer) override
  {
    return m_output_handler->output_selected_events(ring_buffer);
  };

private:
  std::unique_ptr<FileWriter> m_output_handler {nullptr};
};

class ZMQOutputWriter final : public OutputWriter {
public:
  using OutputWriter::OutputWriter;

  StatusCode initialize() override
  {
    m_output_handler = std::make_unique<ZMQOutputSender>(m_outputConnection.value(), m_zmq_svc.get());
    return OutputWriter::initialize();
  }

  StatusCode finalize() override
  {
    auto sc = OutputWriter::finalize();
    if (!sc.isSuccess()) return sc;
    m_output_handler.reset();
    return sc;
  }

  std::tuple<bool, size_t> output_selected_events(SPSCRingBuffer* ring_buffer) override
  {
    return m_output_handler->output_selected_events(ring_buffer);
  };

private:
  ServiceHandle<IZeroMQSvc> m_zmq_svc {this, "IZeroMQSvc", "ZeroMQSvc"};
  std::unique_ptr<ZMQOutputSender> m_output_handler {nullptr};
};

DECLARE_COMPONENT(OutputWriter)
DECLARE_COMPONENT(FileOutputWriter)
DECLARE_COMPONENT(ZMQOutputWriter)
