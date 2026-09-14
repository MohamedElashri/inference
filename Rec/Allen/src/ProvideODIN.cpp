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
// Transformer publishing the ODIN of the current event from the input provider
// to the TES.
// ----------------------------------------------------------------------------
#include <array>

#include "LHCbAlgs/Transformer.h"
#include "Event/ODIN.h"

#include "InputProvider.h"
#include "MultiEventContextExt.h"

class ProvideODIN final : public LHCb::Algorithm::Transformer<LHCb::ODIN(const EventContext&)> {
  using LHCb::Algorithm::Transformer<LHCb::ODIN(const EventContext&)>::Transformer;

public:
  ProvideODIN(const std::string& name, ISvcLocator* pSvcLocator) :
    Transformer(name, pSvcLocator, KeyValue {"ODIN", LHCb::ODINLocation::Default})
  {}

  LHCb::ODIN operator()(const EventContext& evtCtx) const override
  {
    const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(evtCtx);
    return m_input_provider->getODIN(ctxExt->slice_index);
  }

private:
  ServiceHandle<IInputProviderSvc> m_input_provider {this, "InputProvider", "MDFProvider"};
};

DECLARE_COMPONENT(ProvideODIN)
