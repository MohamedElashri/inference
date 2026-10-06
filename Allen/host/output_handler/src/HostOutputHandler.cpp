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
#include "HostOutputHandler.h"

namespace {
  // Size of the MDF header
  auto const header_size = LHCb::MDFHeader::sizeOf(Allen::mdf_header_version);
  // size of the RoutingBits RawBank
  constexpr unsigned routing_bits_size = RoutingBitsDefinition::n_words * sizeof(uint32_t);
} // namespace

INSTANTIATE_ALGORITHM(host_output_handler::host_output_handler_t)

void host_output_handler::host_output_handler_t::init()
{
#ifndef ALLEN_STANDALONE
  m_nprocessed = std::make_unique<Gaudi::Accumulators::Counter<>>(this, "NProcessed");
  m_noutput = std::make_unique<Gaudi::Accumulators::Counter<>>(this, "NOutput");
  m_ntae = std::make_unique<Gaudi::Accumulators::Counter<>>(this, "NTAEOutput");
  m_nbatches = std::make_unique<Gaudi::Accumulators::AveragingCounter<>>(this, "NBatches");
  m_batch_size = std::make_unique<Gaudi::Accumulators::AveragingCounter<>>(this, "BatchSize");
#endif
}

void host_output_handler::host_output_handler_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& ctx) const
{
  HLT1Outputs outputs {
    get<host_global_decision_t>(arguments),
    get<host_dec_reports_t>(arguments),
    get<host_routingbits_t>(arguments),
    get<host_sel_reports_t>(arguments),
    get<host_selrep_offsets_t>(arguments),
    get<host_lumi_summaries_t>(arguments),
    get<host_lumi_summary_offsets_t>(arguments),
    get<host_tae_events_t>(arguments)};

  const unsigned start_event = std::get<0>(runtime_options.event_interval);

  auto [success, n_output] =
    output_single_events(runtime_options.slice_index, start_event, outputs, runtime_options.input_provider.get(), ctx);
  if (!success) return;

  size_t n_tae = 0;
  std::tie(success, n_tae) =
    output_tae_events(runtime_options.slice_index, start_event, outputs, runtime_options.input_provider.get(), ctx);
}

std::span<char> host_output_handler::host_output_handler_t::buffer(const Allen::Context& ctx, size_t size) const
{
  return OutputManager::get()->reserve_write(ctx.stream_id, size);
}

void host_output_handler::host_output_handler_t::write_buffer(const Allen::Context& ctx) const
{
  OutputManager::get()->commit(ctx.stream_id);
}

std::tuple<bool, size_t> host_output_handler::host_output_handler_t::output_single_events(
  size_t const slice_index,
  size_t const start_event,
  HLT1Outputs const& outputs,
  IInputProvider const* input_provider,
  const Allen::Context& ctx) const
{
  // If TAE events should to be output as batches, that's done
  // separately in output_tae_event, so skip them here
  std::span<TAE::TAEEvent const> tae_events;
  bool output_tae = !outputs.tae_events.empty();
  if (output_tae) {
    tae_events = outputs.tae_events;
  }

  std::vector<unsigned> selected_events;
  selected_events.reserve(outputs.selected_events.size());
  size_t tae_index = 0;
  for (unsigned i = 0; i < outputs.selected_events.size(); ++i) {
    if (outputs.selected_events[i] && (!output_tae || i != tae_events[tae_index].central)) {
      // selected_events is passed to the InputProvider to get the
      // event sizes. The InputProvider doesn't know about slice
      // splitting, so we have to offset by start_event here.
      auto const event_number = i + start_event;
      selected_events.push_back(event_number);
    }
    if (output_tae && tae_index < (tae_events.size() - 1) && tae_events[tae_index].central <= i) {
      ++tae_index;
    }
  }

  auto const n_events = static_cast<size_t>(selected_events.size());
  if (n_events == 0) return {true, 0};

  // sizes will contain the total size of all input banks in the event

  OutputSizes sizes {input_provider->events_per_slice()};
  event_sizes(sizes, slice_index, outputs, input_provider, selected_events, start_event);
  auto event_ids = input_provider->event_ids(slice_index);

  bool output_success = true;

  // Output regular events in batches
  size_t n_output = 0;
  size_t n_batches = n_events / m_output_batch_size + (n_events % m_output_batch_size != 0);

#ifndef ALLEN_STANDALONE
  if (m_nbatches) (*m_nbatches) += n_batches;
  if (m_nprocessed) (*m_nprocessed) += outputs.selected_events.size();
#endif

  for (size_t i_batch = 0; i_batch < n_batches && output_success; ++i_batch) {

    size_t batch_buffer_size = 0;
    size_t output_event_offset = 0;
    size_t batch_size = std::min(m_output_batch_size.value(), n_events - n_output);

#ifndef ALLEN_STANDALONE
    if (m_noutput) (*m_noutput) += batch_size;
    if (m_batch_size) (*m_batch_size) += batch_size;
#endif

    for (size_t i = n_output; i < n_output + batch_size; ++i) {
      batch_buffer_size += sizes.input[i] + sizes.hlt[i] + header_size;
    }

    std::span<char> batch_span = buffer(ctx, batch_buffer_size);

    // In case output was cancelled
    if (batch_span.empty()) return {false, 0};

    for (size_t i = n_output; i < n_output + batch_size; ++i) {

      // The event number is constructed to index into a batch. The 0th
      // event of a batch is start_event in a slice, so we subtract
      // start_event that was added to selected_events to have a direct
      // index into the batch again.
      unsigned const event_number = selected_events[i] - start_event;

      // event sizes are indexed in the same way as selected_events
      size_t output_event_size = header_size + sizes.input[i] + sizes.hlt[i];

      // The memory range in the output buffer for this event
      auto event_span = batch_span.subspan(output_event_offset, output_event_size);

      // WORKING, fix changed type of members of HLT1Outputs and avoid
      // "used uninitialized warnings". Probably best to add a check
      // for all of the ones we assume are present.

      // Add the MDF header
      auto* header = Allen::add_mdf_header(
        event_span,
        static_cast<unsigned int>(std::get<0>(event_ids[event_number + start_event])),
        outputs.routing_bits.subspan(RoutingBitsDefinition::n_words * event_number, RoutingBitsDefinition::n_words));

      // Add the input banks and HLT1 banks to the event
      add_banks(
        outputs,
        input_provider,
        slice_index,
        start_event,
        event_number,
        sizes.input[i],
        event_span.subspan(header_size, output_event_size - header_size));

      add_checksum(header, event_span);

      output_event_offset += output_event_size;
    }

    write_buffer(ctx);
    n_output += batch_size;
  }
  assert(n_events - n_output == 0);

  return {output_success, n_output};
}

std::tuple<bool, size_t> host_output_handler::host_output_handler_t::output_tae_events(
  size_t const slice_index,
  size_t const start_event,
  HLT1Outputs const& outputs,
  IInputProvider const* input_provider,
  const Allen::Context& ctx) const
{
  // Main approach to adding TAE banks:
  // a) Output TAE events into a separate buffer
  // b) try to measure effect on throughout with buffer manager
  // c) optimize if needed by adding more threads, or including TAE
  //    events in the same buffers as the batches

  if (outputs.tae_events.empty()) return {true, 0};

  // The TAEHeader bank must be the first bank after the MDF header,
  // should by of type TAEHeader and have a body consisting of a
  // triplet of ints (nBx, offset, size); nBx starts at
  // -tae_half_window; the offset is with respect to the end of the
  // TAEHeader bank; and size is in bytes

  std::vector<unsigned> selected_events;
  std::vector<unsigned> tae_offsets;

  auto& tae_events = outputs.tae_events;
  std::vector<TAE::TAEEvent> selected_tae_events;
  // for now, set the size to the number of global decisions
  selected_events.reserve(outputs.selected_events.size() * (2 * tae_events[0].half_window + 1));
  tae_offsets.reserve(outputs.selected_events.size());
  selected_tae_events.reserve(outputs.tae_events.size());

  unsigned n_selected_tae_events = 0;

  for (auto tae_event : tae_events) {
    auto central_tae_in_global_decision = outputs.selected_events[tae_event.central];
    if (!central_tae_in_global_decision) continue; // tae event not in global decision, skip
    n_selected_tae_events++;
    tae_offsets.push_back(selected_events.size());
    selected_tae_events.push_back(tae_event);
    for (unsigned event_number = tae_event.central - tae_event.half_window;
         event_number <= tae_event.central + tae_event.half_window;
         ++event_number) {
      // selected_events is passed to the InputProvider to get the
      // event sizes. The InputProvider doesn't know about slice
      // splitting, so we have to offset by start_event here.
      selected_events.push_back(event_number + start_event);
    }
  }
  if (n_selected_tae_events == 0) return {true, 0};

  selected_events.resize(n_selected_tae_events * (2 * tae_events[0].half_window + 1));
  tae_offsets.resize(n_selected_tae_events);
  selected_tae_events.resize(n_selected_tae_events);

#ifndef ALLEN_STANDALONE
  if (m_ntae) (*m_ntae) += n_selected_tae_events;
#endif

  auto event_ids = input_provider->event_ids(slice_index);

  OutputSizes sizes {input_provider->events_per_slice()};
  event_sizes(sizes, slice_index, outputs, input_provider, selected_events, start_event);

  auto tae_bank_size = [](unsigned half_window) { return (2 * half_window + 1) * 3 * sizeof(int); };

  size_t tae_buffer_size = 0;
  for (size_t tae_index = 0; tae_index < n_selected_tae_events; ++tae_index) {
    auto const& tae_event = selected_tae_events[tae_index];
    auto const offset = tae_offsets[tae_index];
    size_t tae_size = header_size + bank_header_size + tae_bank_size(tae_event.half_window);
    for (unsigned sub_index = offset; sub_index < offset + 2 * tae_event.half_window + 1; ++sub_index) {
      tae_size += sizes.input[sub_index] + sizes.hlt[sub_index];
    }
    tae_buffer_size += tae_size;
    sizes.tae[tae_event.central] = tae_size;
  }

  std::span<char> tae_buffer = buffer(ctx, tae_buffer_size);

  size_t tae_output_offset = 0;
  for (size_t tae_index = 0; tae_index < n_selected_tae_events; ++tae_index) {
    auto const& tae_event = selected_tae_events[tae_index];
    auto const offset = tae_offsets[tae_index];
    auto const tae_size = sizes.tae[tae_event.central];
    auto tae_span = tae_buffer.subspan(tae_output_offset, tae_size);

    auto header = Allen::add_mdf_header(
      tae_span,
      static_cast<unsigned int>(std::get<0>(event_ids[tae_event.central + start_event])),
      outputs.routing_bits.subspan(RoutingBitsDefinition::n_words * tae_event.central, RoutingBitsDefinition::n_words));

    // Build the header of the TAEHeader bank
    auto* tae_header = reinterpret_cast<LHCb::RawBank*>(&tae_span[0] + header_size);
    tae_header->setMagic();
    tae_header->setType(LHCb::RawBank::BankType::TAEHeader);
    tae_header->setVersion(0);
    tae_header->setSourceID(0);
    tae_header->setSize(tae_bank_size(tae_event.half_window));

    auto const preamble_size = header_size + tae_header->totalSize();
    auto* tae_header_payload = tae_header->begin<int>();

    // Shrink tae_span to the combined size of the sub-events
    auto payload_span = tae_span.subspan(preamble_size, tae_size - preamble_size);

    int tae_offset = 0;

    // Copy the banks of the sub events and update the body of the TAEHeader bank
    for (int i = 0; i <= 2 * static_cast<int>(tae_event.half_window); ++i) {
      unsigned const event_number = tae_event.central - tae_event.half_window + i;
      unsigned const size_index = offset + i;
      // Add banks of this TAE sub event
      auto const sub_size = static_cast<int>(add_banks(
        outputs,
        input_provider,
        slice_index,
        start_event,
        event_number,
        sizes.input[size_index],
        payload_span.subspan(tae_offset, sizes.input[size_index] + sizes.hlt[size_index])));

      // Fill the next triplet in the header payload
      for (int v : {i - static_cast<int>(tae_event.half_window), tae_offset, sub_size}) {
        *tae_header_payload++ = v;
      }

      // next sub event
      tae_offset += sub_size;
    }

    add_checksum(header, tae_span);

    // Next TAE event
    tae_output_offset += tae_size;
  }

  write_buffer(ctx);
  return {true, n_selected_tae_events};
}

void host_output_handler::host_output_handler_t::event_sizes(
  OutputSizes& sizes,
  size_t const slice_index,
  HLT1Outputs const& outputs,
  IInputProvider const* input_provider,
  std::span<unsigned> const& selected_events,
  unsigned const start_event) const
{
  input_provider->event_sizes(slice_index, selected_events, sizes.input);

  // Add the HLT bank sizes to event sizes
  for (size_t i = 0; i < selected_events.size(); ++i) {
    auto const event_number = selected_events[i] - start_event;

    HltDecReports dec_reports {outputs.dec_reports, event_number};
    unsigned const dec_report_size = dec_reports.bank_data().size_bytes();

    // size of the SelReport RawBank
    // need the index into the batch here
    unsigned const sel_report_size =
      outputs.sel_reports_offsets.empty() ?
        0 :
        (outputs.sel_reports_offsets[event_number + 1] - outputs.sel_reports_offsets[event_number]) * sizeof(uint32_t);
    unsigned const lumi_summary_size =
      outputs.lumi_summary_offsets.empty() ?
        0 :
        (outputs.lumi_summary_offsets[event_number + 1] - outputs.lumi_summary_offsets[event_number]) *
          sizeof(uint32_t);

    for (auto hlt_bank_size : {dec_report_size, routing_bits_size, sel_report_size, lumi_summary_size}) {
      if (hlt_bank_size > 0) {
        sizes.hlt[i] += bank_header_size + hlt_bank_size;
      }
    }
  }
}

void host_output_handler::host_output_handler_t::add_checksum(LHCb::MDFHeader* header, std::span<char> event_span) const
{
  if (m_checksum.value()) {
    auto const skip = 4 * sizeof(int);
    auto c = LHCb::hash32Checksum(event_span.data() + skip, event_span.size() - skip);
    header->setChecksum(c);
  }
  else {
    header->setChecksum(0);
  }
}

// WORKING: Fix this function
size_t host_output_handler::host_output_handler_t::add_banks(
  HLT1Outputs const& outputs,
  IInputProvider const* input_provider,
  unsigned const slice_index,
  unsigned const start_event,
  unsigned const event_number,
  unsigned const input_size,
  std::span<char> event_span) const
{
  // The batch is offset by start_event with respect to the slice, so we add start_event
  input_provider->copy_banks(
    slice_index, event_number + start_event, {event_span.data(), static_cast<events_size>(input_size)});

  // Starting point of HLT banks
  char* output = event_span.data() + input_size;

  // size of the DecReport RawBank
  HltDecReports dec_reports {outputs.dec_reports, event_number};

  // size of the SelReport RawBank
  // need the index into the batch here
  const unsigned sel_report_offset =
    outputs.sel_reports_offsets.empty() ? 0 : outputs.sel_reports_offsets[event_number];
  const unsigned sel_report_size =
    outputs.sel_reports_offsets.empty() ?
      0 :
      (outputs.sel_reports_offsets[event_number + 1] - outputs.sel_reports_offsets[event_number]) * sizeof(uint32_t);

  // size of the lumi summary RawBank
  // need the index into the batch here
  const unsigned lumi_summary_offset =
    outputs.lumi_summary_offsets.empty() ? 0 : outputs.lumi_summary_offsets[event_number];
  const unsigned lumi_summary_size =
    outputs.lumi_summary_offsets.empty() ?
      0 :
      (outputs.lumi_summary_offsets[event_number + 1] - lumi_summary_offset) * sizeof(uint32_t);

  using output_bank = std::tuple<LHCb::RawBank::BankType, unsigned, unsigned, std::span<char const>>;
  auto hlt_banks = std::make_tuple(
    // HltDecReports
    output_bank {
      LHCb::RawBank::BankType::HltDecReports, dec_reports.version(), dec_reports.source_id(), dec_reports.bank_data()},
    // HltRoutingBits
    output_bank {
      LHCb::RawBank::BankType::HltRoutingBits,
      0u,
      Hlt1::Constants::sourceID,
      {reinterpret_cast<char const*>(outputs.routing_bits.data()) + routing_bits_size * event_number,
       static_cast<events_size>(routing_bits_size)}},
    // HltSelReports
    output_bank {
      LHCb::RawBank::BankType::HltSelReports,
      Hlt1::Constants::version_sel_reports,
      Hlt1::Constants::sourceID_sel_reports,
      {reinterpret_cast<char const*>(outputs.sel_reports.data()) + sel_report_offset * sizeof(uint32_t),
       static_cast<events_size>(sel_report_size)}},
    // HltLumiSummary
    output_bank {
      LHCb::RawBank::BankType::HltLumiSummary,
      2u,
      Hlt1::Constants::sourceID,
      {reinterpret_cast<char const*>(outputs.lumi_summaries.data()) + lumi_summary_offset * sizeof(uint32_t),
       static_cast<events_size>(lumi_summary_size)}});

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

  return static_cast<size_t>(output - event_span.data());
}
