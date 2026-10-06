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
#include "PVFinderVertexDump.h"
#include "PVFinderMCVertices.h"
#include "MEPTools.h"

#include <fstream>
#include <mutex>
#include <set>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_pv_dump::pvfinder_pv_dump_t)

namespace {
  // Serialises the writes of all instances and streams; each file is truncated
  // by the first write to it in the process.
  std::mutex dump_mutex;
  std::set<std::string> started_files;

  std::vector<PVFinder::MCVertex> mc_vertices(
    std::span<const char> banks,
    std::span<const unsigned> offsets,
    std::span<const unsigned> sizes,
    unsigned event)
  {
    if (event + 1 >= offsets.size() || offsets[event] > offsets[event + 1] || offsets[event + 1] > banks.size())
      throw StrException("pvfinder_pv_dump: invalid MC-PV event offsets");
    const auto raw = banks.subspan(offsets[event], offsets[event + 1] - offsets[event]);
    auto word = [&raw](size_t offset) {
      if (offset > raw.size() || raw.size() - offset < sizeof(uint32_t))
        throw StrException("pvfinder_pv_dump: truncated MC-PV bank header");
      uint32_t value;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      return value;
    };
    const unsigned count = word(0);
    if (raw.size() < 2 * sizeof(uint32_t) || count > raw.size() / sizeof(uint32_t) - 2)
      throw StrException("pvfinder_pv_dump: invalid MC-PV bank count");
    const size_t payload_offset = (static_cast<size_t>(count) + 2) * sizeof(uint32_t);
    std::vector<char> payload;
    for (unsigned bank = 0; bank < count; ++bank) {
      const size_t offset = payload_offset + word((bank + 1) * sizeof(uint32_t)) + sizeof(uint32_t);
      const size_t length = Allen::bank_size(sizes.data(), event, bank);
      if (offset > raw.size() || length > raw.size() - offset)
        throw StrException("pvfinder_pv_dump: truncated MC-PV bank payload");
      payload.insert(payload.end(), raw.data() + offset, raw.data() + offset + length);
    }
    return PVFinder::read_mc_vertices(payload);
  }
} // namespace

void pvfinder_pv_dump::pvfinder_pv_dump_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  const auto vertices = make_host_buffer<dev_multi_final_vertices_t>(arguments, context);
  const auto number_of_vertices = make_host_buffer<dev_number_of_multi_final_vertices_t>(arguments, context);
  const auto event_list = make_host_buffer<dev_event_list_t>(arguments, context);
  if (runtime_options.mep_layout || size<host_mc_pv_banks_t>(arguments) != 1)
    throw StrException("pvfinder_pv_dump: MC comparison requires the MDF bank layout");

  std::vector<char> buffer;
  auto put = [&buffer](const auto value) {
    const auto* bytes = reinterpret_cast<const char*>(&value);
    buffer.insert(buffer.end(), bytes, bytes + sizeof(value));
  };

  const auto batch = m_batch++;
  for (const auto event_number : event_list) {
    const auto mc_pvs = mc_vertices(
      first<host_mc_pv_banks_t>(arguments),
      first<host_mc_pv_offsets_t>(arguments),
      first<host_mc_pv_sizes_t>(arguments),
      std::get<0>(runtime_options.event_interval) + event_number);
    const unsigned n_rec = number_of_vertices[event_number];
    put(static_cast<uint32_t>(batch));
    put(static_cast<uint32_t>(std::get<0>(runtime_options.event_interval) + event_number));
    put(static_cast<uint32_t>(n_rec));
    put(static_cast<uint32_t>(mc_pvs.size()));
    for (unsigned i = 0; i < n_rec; ++i) {
      const auto& pv = vertices[event_number * PV::max_number_vertices + i];
      for (const float value :
           {pv.position.x,
            pv.position.y,
            pv.position.z,
            pv.cov00,
            pv.cov11,
            pv.cov22,
            pv.chi2,
            static_cast<float>(pv.ndof),
            pv.nTracks}) {
        put(value);
      }
    }
    for (const auto& mc_pv : mc_pvs) {
      for (const double value : {mc_pv.x, mc_pv.y, mc_pv.z, static_cast<double>(mc_pv.numberTracks)}) {
        put(value);
      }
    }
  }

  std::lock_guard<std::mutex> lock {dump_mutex};
  const std::string& filename = m_output_filename.value();
  const bool first_write = started_files.insert(filename).second;
  std::ofstream file {filename, std::ios::binary | (first_write ? std::ios::trunc : std::ios::app)};
  if (!file) throw StrException("pvfinder_pv_dump: cannot open " + filename);
  file.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
}
