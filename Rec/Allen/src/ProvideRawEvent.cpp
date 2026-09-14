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

// ----------------------------------------------------------------------------
// Transformer publishing the current raw event from the input provider to the
// TES.
// ----------------------------------------------------------------------------
#include <array>
#include <map>
#include <string>

#include "LHCbAlgs/Transformer.h"
#include "Event/RawEvent.h"

#include "InputProvider.h"
#include "MultiEventContextExt.h"

class ProvideRawEvent final : public LHCb::Algorithm::Transformer<LHCb::RawEvent(const EventContext&)> {
  using LHCb::Algorithm::Transformer<LHCb::RawEvent(const EventContext&)>::Transformer;

public:
  ProvideRawEvent(const std::string& name, ISvcLocator* pSvcLocator) :
    Transformer(name, pSvcLocator, KeyValue {"RawEventLocation", ""})
  {}

  LHCb::RawEvent operator()(const EventContext& evtCtx) const override
  {
    const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(evtCtx);
    return m_input_provider->getRawEvent(ctxExt->slice_index, evtCtx.evt());
  }

private:
  ServiceHandle<IInputProviderSvc> m_input_provider {this, "InputProviderService", "MDFProvider"};
};

DECLARE_COMPONENT(ProvideRawEvent)
