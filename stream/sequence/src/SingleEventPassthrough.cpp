/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <SingleEventPassthrough.cuh>
#include <Logger.h>
#include <HltSubBanks.cuh>
#include <HltSelReport.cuh>
#include <HltDecReport.cuh>
#include <mdf_header.hpp>
#include <raw_helpers.hpp>
#include <read_mdf.hpp>
#include <write_mdf.hpp>
#include <regex>

namespace {
  // Size of the MDF header
  auto const header_size = LHCb::MDFHeader::sizeOf(Allen::mdf_header_version);
} // namespace

void SingleEventPassthrough::init()
{
  // The following code builds banks that are common to all single event passthrough with the current
  // configuration, it is then cached into the m_banks buffer.

  // Build banks
  std::vector<unsigned> dr_data(HltDecReports<false>::size(1u), 0u);
  HltDecReports<false> decrep({dr_data.data(), dr_data.size()}, 0u, 1u);
  decrep.set_number_of_lines(1u);
  decrep.set_key(m_passthrough_key);
  decrep.set_tck(m_tck);
  decrep.set_task_id(m_task_id);
  decrep.set_dec_report(
    0u,
    HltDecReport {
      true,
      std::byte {0},                    // error
      std::byte {1},                    // number of candidates
      std::byte {1},                    // execution stage
      static_cast<unsigned short>(1)}); // decision ID

  std::array<unsigned, 3> routing_bits {m_passthrough_rbs, 0, 0};

  // Make the substructure bank.
  // Substructure bank size. First word for bank size info and the second for
  // the substructure information.
  const unsigned substr_bank_size = 2;
  unsigned rb_substr[2] = {0, 0};
  // Selection list. There is one selection with ID 0.
  unsigned sel_list[1] = {0};
  // Line object offsets. Can be anything, but it needs to exist.
  unsigned line_object_offsets[1] = {0};
  // Multi-event containers. Need a container of nullptrs.
  Allen::IMultiEventContainer* mecs[1] = {nullptr};
  make_subbanks::make_rb_substr_bank(
    rb_substr,
    substr_bank_size,
    0,
    1, // Number of selections.
    0,
    0,
    0,
    1, // Size of selections in the substructure bank.
    0,
    0,
    mecs, // Need to be able to access the first pointer.
    nullptr,
    nullptr,
    nullptr,
    line_object_offsets,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    sel_list,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr);

  // Make the ObjTyp bank.
  // The ObjTyp bank size is 2. One word giving the bank size and one word for
  // each object type. We only save a selection here, so there is 1 object type.
  const unsigned n_objtyps = 1;
  const unsigned objtyp_bank_size = 1 + n_objtyps;
  unsigned rb_objtyp[2] = {0, 0};
  make_subbanks::make_rb_objtyp_bank(
    rb_objtyp,
    n_objtyps,
    1, // One selection, no other objects.
    0,
    0,
    0);

  // Make the StdInfo bank.
  // The StdInfo bank contains 1 word giving the structure of the bank, 8 bits
  // plus padding giving the number of values saved for the persisted object,
  // and 1 word for the selection.
  const unsigned stdinfo_bank_size = 3;
  unsigned rb_stdinfo[3] = {0, 0, 0};
  make_subbanks::make_rb_stdinfo_bank(
    rb_stdinfo,
    stdinfo_bank_size,
    1, // One selection, no other objects.
    0,
    0,
    0,
    sel_list,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr);

  // Make the hits bank. No hits are stored, so this is one word containing 0.
  const unsigned hits_bank_size = 1;
  unsigned rb_hits[1] = {0};

  // Extra info bank. This bank is empty, but it needs to exist. It is organized
  // similarly to the StdInfo bank, but there is no saved information. The size
  // is 2.
  const unsigned einfo_size = 2;

  // Make the selreport bank.
  const unsigned header_size = 10;
  const unsigned selrep_bank_size =
    header_size + substr_bank_size + stdinfo_bank_size + objtyp_bank_size + hits_bank_size + einfo_size;
  std::vector<unsigned> sr_data(selrep_bank_size, 0);
  make_selrep::make_selrep_bank(
    sr_data.data(),
    rb_objtyp,
    rb_hits,
    rb_substr,
    rb_stdinfo,
    selrep_bank_size,
    objtyp_bank_size,
    hits_bank_size,
    substr_bank_size,
    stdinfo_bank_size);

  // Compute size
  const unsigned dec_report_size = decrep.bank_data().size_bytes();
  const unsigned routing_bits_size = routing_bits.size() * sizeof(uint32_t);
  const unsigned selrep_bank_size_bytes = selrep_bank_size * sizeof(uint32_t);
  size_t hlt_size = 0;
  for (auto hlt_bank_size : {dec_report_size, routing_bits_size, selrep_bank_size_bytes}) {
    if (hlt_bank_size > 0) {
      hlt_size += bank_header_size + hlt_bank_size;
    }
  }
  m_banks.resize(hlt_size);

  // Fill buffer
  char* output = m_banks.data();

  using output_bank = std::tuple<LHCb::RawBank::BankType, unsigned, unsigned, std::span<char const>>;
  auto hlt_banks = std::make_tuple(
    // HltDecReports
    output_bank {LHCb::RawBank::BankType::HltDecReports, decrep.version(), decrep.source_id(), decrep.bank_data()},
    // HltRoutingBits
    output_bank {
      LHCb::RawBank::BankType::HltRoutingBits,
      0u,
      Hlt1::Constants::sourceID,
      {reinterpret_cast<char const*>(routing_bits.data()), static_cast<events_size>(routing_bits_size)}},
    // HltSelReports
    output_bank {
      LHCb::RawBank::BankType::HltSelReports,
      Hlt1::Constants::version_sel_reports,
      Hlt1::Constants::sourceID_sel_reports,
      {reinterpret_cast<char const*>(sr_data.data()), static_cast<events_size>(selrep_bank_size_bytes)}});

  // Lambda to add an HLT output bank to the output event
  auto add_hlt_bank = [](
                        LHCb::RawBank::BankType bank_type,
                        unsigned version,
                        unsigned source_id,
                        std::span<char const> data,
                        char* output) -> size_t {
    return data.empty() ? 0u : Allen::add_raw_bank((uint8_t) bank_type, version, source_id, data, output);
  };

  for_each(hlt_banks, [&output, &add_hlt_bank](auto b) {
    auto t = std::tuple_cat(b, std::tuple {output});
    output += std::apply(add_hlt_bank, t);
  });
}

void SingleEventPassthrough::write(
  size_t const slice_index,
  unsigned const start_event,
  IInputProvider const* input_provider,
  int producer_id) const
{
  size_t input_size = 0;
  input_provider->event_sizes(
    slice_index, std::span<unsigned const> {&start_event, 1u}, std::span<size_t> {&input_size, 1u});
  auto event_ids = input_provider->event_ids(slice_index);

  std::span<char> event_span =
    OutputManager::get()->reserve_write(producer_id, header_size + input_size + m_banks.size());

  std::array<unsigned, 3> routing_bits {m_passthrough_rbs, 0, 0};
  auto* header = Allen::add_mdf_header(event_span, static_cast<unsigned int>(std::get<0>(event_ids[0])), routing_bits);

  input_provider->copy_banks(slice_index, start_event, event_span.subspan(header_size, input_size));
  std::memcpy(event_span.data() + header_size + input_size, m_banks.data(), m_banks.size());

  if (m_do_checksum) {
    auto const skip = 4 * sizeof(int);
    auto c = LHCb::hash32Checksum(event_span.data() + skip, event_span.size() - skip);
    header->setChecksum(c);
  }
  else {
    header->setChecksum(0);
  }

  OutputManager::get()->commit(producer_id);

#ifndef ALLEN_STANDALONE
  if (m_npassthrough) ++(*m_npassthrough);
#endif
}

#ifndef ALLEN_STANDALONE
void SingleEventPassthrough::activateMonitoring(Service* svc)
{
  if (svc != nullptr) {
    m_npassthrough = std::make_unique<Gaudi::Accumulators::Counter<>>(svc, "NPassthrough");
  }
}
#endif
