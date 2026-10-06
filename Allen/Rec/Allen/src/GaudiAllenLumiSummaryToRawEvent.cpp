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

// Standard
#include <vector>

// Gaudi
#include "GaudiKernel/StdArrayAsProperty.h"

// LHCb
#include <LHCbAlgs/Transformer.h>
#include "Kernel/STLExtensions.h"
#include "Event/RawEvent.h"
#include <HltConstants.cuh>
#include "EventTransformer.h"

#include <AllenBuffer.cuh>

struct GaudiAllenLumiSummaryToRawEvent final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<LHCb::RawEvent, LHCb::RawBank::View>(
      const Allen::host_buffer<unsigned>&,
      const Allen::host_buffer<unsigned>&)> {
  // Standard constructor
  GaudiAllenLumiSummaryToRawEvent(const std::string& name, ISvcLocator* pSvcLocator);

  // Algorithm execution
  std::tuple<std::vector<LHCb::RawEvent>, std::vector<LHCb::RawBank::View>> operator()(
    const EventContext&,
    const Allen::host_buffer<unsigned>& allen_lumi_summaries,
    const Allen::host_buffer<unsigned>& allen_lumi_summary_offsets) const override;
};

DECLARE_COMPONENT(GaudiAllenLumiSummaryToRawEvent)

GaudiAllenLumiSummaryToRawEvent::GaudiAllenLumiSummaryToRawEvent(const std::string& name, ISvcLocator* pSvcLocator) :
  MultiTransformer(
    name,
    pSvcLocator,
    // Inputs
    {KeyValue {"allen_lumi_summaries", ""}, KeyValue {"allen_lumi_summary_offsets", ""}},
    // Outputs
    {KeyValue {"OutputLumiSummary", "Allen/Out/LumiSummary"},
     KeyValue {"OutputLumiSummaryView", "Allen/Out/LumiSummaryView"}})
{}

std::tuple<std::vector<LHCb::RawEvent>, std::vector<LHCb::RawBank::View>> GaudiAllenLumiSummaryToRawEvent::operator()(
  const EventContext&,
  const Allen::host_buffer<unsigned>& allen_lumi_summaries,
  const Allen::host_buffer<unsigned>& allen_lumi_summary_offsets) const
{
  const unsigned n_events = allen_lumi_summary_offsets.size() - 1;
  // std::cout << "Number of events in slice: " << n_events << std::endl;

  std::tuple<std::vector<LHCb::RawEvent>, std::vector<LHCb::RawBank::View>> output;
  auto& [events, lumi_summary_views] = output;
  events.reserve(n_events);
  lumi_summary_views.reserve(n_events);

  for (unsigned i = 0; i < n_events; i++) {
    // std::cout << "Event " << i << ": lumi summary offset = " << allen_lumi_summary_offsets[i] << std::endl;

    LHCb::RawEvent raw_event;
    auto lumi_summaries = allen_lumi_summaries.subspan(
      allen_lumi_summary_offsets[i], allen_lumi_summary_offsets[i + 1] - allen_lumi_summary_offsets[i]);
    if (!lumi_summaries.empty()) {
      raw_event.addBank(Hlt1::Constants::sourceID, LHCb::RawBank::BankType::HltLumiSummary, 2u, lumi_summaries);
    }

    auto [raw_event_out, lumi_summary_view] =
      viewsFromRawEvent(std::move(raw_event), std::array {LHCb::RawBank::BankType::HltLumiSummary});
    events.push_back(std::move(raw_event_out));
    lumi_summary_views.push_back(std::move(lumi_summary_view));
  }
  return output;
}
