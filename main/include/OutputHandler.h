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
// Abstract interface for output handlers that consume selected events from a
// producer ring buffer and write them to a destination (file, network, ...).
// ----------------------------------------------------------------------------
#pragma once

#include <span>
#include <vector>
#include <tuple>
#include <zmq.hpp>
#include <SPSCRingBuffer.h>

class OutputHandler {
public:
  OutputHandler() {}

  OutputHandler(std::string const connection) : m_connection {connection} {}

  virtual ~OutputHandler() {}

  std::string const& connection() const { return m_connection; }

  virtual std::tuple<bool, size_t> output_selected_events(SPSCRingBuffer*) = 0;

  virtual zmq::socket_t* client_socket() const { return nullptr; }

  virtual void handle() {}

  virtual void cancel() {}

  virtual void output_done() {}

protected:
  void init(std::string const& connection) { m_connection = connection; }

  size_t count_events(std::span<char> const& buffer)
  {
    size_t count = 0;
    size_t i = 0;
    while (i < buffer.size()) {
      i += *reinterpret_cast<unsigned*>(buffer.data() + i); // size of the record is the first unsigned
      count++;
    }
    return count;
  }

private:
  std::string m_connection;
};
