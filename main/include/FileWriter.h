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
#pragma once

#include <read_mdf.hpp>
#include <OutputHandler.h>

class FileWriter final : public OutputHandler {
public:
  FileWriter(std::string filename) : OutputHandler {std::move(filename)}
  {
    std::cout << "Opening output file " << connection() << std::endl;
    m_output = MDF::open(connection(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (!m_output.good) {
      throw std::runtime_error {"Failed to open output file"};
    }
  }

  std::tuple<bool, size_t> output_selected_events(SPSCRingBuffer* ring_buffer) override
  {
    std::span<char> buffer = ring_buffer->consume();
    bool success = true;
    if (!buffer.empty()) {
      success = m_output.write(buffer.data(), buffer.size());
    }
    size_t count = count_events(buffer);
    ring_buffer->release();
    return {success, count};
  }

  ~FileWriter()
  {
    if (m_output.good) {
      std::cout << "Closing output file " << connection() << std::endl;
      m_output.close();
    }
  }

private:
  // Storage for the currently open output file
  Allen::IO m_output;
};
