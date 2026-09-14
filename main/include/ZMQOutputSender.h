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

#include <read_mdf.hpp>
#include <raw_helpers.hpp>

#include <ZeroMQ/IZeroMQSvc.h>
#include <OutputHandler.h>

namespace {
  void release_ringbuffer(void*, void* hint) { reinterpret_cast<SPSCRingBuffer*>(hint)->release(); }
} // namespace

class ZMQOutputSender final : public OutputHandler {
public:
  ZMQOutputSender(std::string receiver_connection, IZeroMQSvc* zmqSvc);

  std::tuple<bool, size_t> output_selected_events(SPSCRingBuffer* ring_buffer) override
  {
    std::span<char> buffer = ring_buffer->consume();
    if (m_connected && !buffer.empty()) {
      size_t count = count_events(buffer);
      // Use zero copy to build the message, release_ringbuffer will be called when zmq is done with the message
      zmq::message_t message_buffer {
        buffer.data(), buffer.size(), &release_ringbuffer, reinterpret_cast<void*>(ring_buffer)};
      m_zmq->send(*m_socket, "EVENT", zmq::send_flags::sndmore);
      m_zmq->send(*m_socket, message_buffer);
      return {true, count};
    }
    else {
      ring_buffer->release();
    }
    return {true, 0};
  }

  ~ZMQOutputSender();

  zmq::socket_t* client_socket() const override;

  void handle() override;

private:
  // ZeroMQSvc pointer for convenience.
  IZeroMQSvc* m_zmq = nullptr;

  // ID string
  std::string m_id;

  // are we connected to a receiver
  bool m_connected = false;

  // data socket
  mutable std::optional<zmq::socket_t> m_socket;

  // request socket
  std::optional<zmq::socket_t> m_request;
};
