/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
// Gaudi
#include "Event/RawEvent.h"
#include <vector>
#include "Kernel/STLExtensions.h"
#include "HltDecReport.cuh"
#include "HltConstants.cuh"
#include <RoutingBitsDefinition.h>
#include <Kernel/EventLocalAllocator.h>
#include "EventTransformer.h"
#include <AllenBuffer.cuh>

class GaudiAllenReportsToRawEvent
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<
      std::tuple<LHCb::RawEvent, LHCb::RawBank::View, LHCb::RawBank::View, LHCb::RawBank::View>(
        const Allen::device_buffer<unsigned>&,
        const Allen::device_buffer<unsigned>&,
        const Allen::device_buffer<unsigned>&,
        const Allen::host_buffer<unsigned>&)> {
public:
  // Standard constructor
  GaudiAllenReportsToRawEvent(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer {
      name,
      pSvcLocator,
      // Inputs
      {KeyValue {"allen_dec_reports", ""},
       KeyValue {"allen_selrep_offsets", ""},
       KeyValue {"allen_sel_reports", ""},
       KeyValue {"allen_routing_bits", ""}},
      // Outputs
      {KeyValue {"OutputRawReports", "Allen/Out/RawReports"},
       KeyValue {"OutputDecView", "Allen/Out/OutputDecView"},
       KeyValue {"OutputSelView", "Allen/Out/OutputSelView"},
       KeyValue {"OutputRoutingBitsView", "Allen/Out/OutputRoutingBitsView"}}}
  {}

  // Algorithm execution
  std::tuple<
    std::vector<LHCb::RawEvent>,
    std::vector<LHCb::RawBank::View>,
    std::vector<LHCb::RawBank::View>,
    std::vector<LHCb::RawBank::View>>
  operator()(
    const EventContext&,
    const Allen::device_buffer<unsigned>& allen_dec_reports,
    const Allen::device_buffer<unsigned>& allen_selrep_offsets,
    const Allen::device_buffer<unsigned>& allen_sel_reports,
    const Allen::host_buffer<unsigned>& allen_routing_bits) const override
  {
    const unsigned n_events = allen_selrep_offsets.size() - 1;

    std::tuple<
      std::vector<LHCb::RawEvent>,
      std::vector<LHCb::RawBank::View>,
      std::vector<LHCb::RawBank::View>,
      std::vector<LHCb::RawBank::View>>
      output;
    auto& [events, dec_views, sel_views, routing_bits_views] = output;
    events.reserve(n_events);
    dec_views.reserve(n_events);
    sel_views.reserve(n_events);
    routing_bits_views.reserve(n_events);

    const auto allen_dec_reports_host = allen_dec_reports.to_host();
    const auto allen_selrep_offsets_host = allen_selrep_offsets.to_host();
    const auto allen_sel_reports_host = allen_sel_reports.to_host();

    for (unsigned i = 0; i < n_events; i++) {
      LHCb::RawEvent raw_event;
      auto dec_reports = HltDecReports {allen_dec_reports_host, i};
      auto sel_reports = allen_sel_reports_host.subspan(
        allen_selrep_offsets_host[i], allen_selrep_offsets_host[i + 1] - allen_selrep_offsets_host[i]);
      auto routing_bits =
        allen_routing_bits.subspan(i * RoutingBitsDefinition::n_words, RoutingBitsDefinition::n_words);
      raw_event.addBank(
        Hlt1::Constants::sourceID_sel_reports,
        LHCb::RawBank::BankType::HltSelReports,
        Hlt1::Constants::version_sel_reports,
        sel_reports);
      raw_event.addBank(
        Hlt1::Constants::sourceID,
        LHCb::RawBank::BankType::HltDecReports,
        dec_reports.version(),
        dec_reports.bank_data());
      raw_event.addBank(Hlt1::Constants::sourceID, LHCb::RawBank::BankType::HltRoutingBits, 0u, routing_bits);

      auto [raw_event_out, dec_view, sel_view, routing_bits_view] = viewsFromRawEvent(
        std::move(raw_event),
        std::array {
          LHCb::RawBank::BankType::HltDecReports,
          LHCb::RawBank::BankType::HltSelReports,
          LHCb::RawBank::BankType::HltRoutingBits});

      events.emplace_back(std::move(raw_event_out));
      dec_views.emplace_back(std::move(dec_view));
      sel_views.emplace_back(std::move(sel_view));
      routing_bits_views.emplace_back(std::move(routing_bits_view));
    }
    return output;
  }
};

DECLARE_COMPONENT(GaudiAllenReportsToRawEvent)
