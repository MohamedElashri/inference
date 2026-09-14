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
// Transformer exposing requested raw-event branches: retrieves them from the
// input provider via EventBranches, puts each branch on the TES and returns the
// current LHCb::RawEvent.
// ----------------------------------------------------------------------------
#include <array>
#include <map>
#include <string>

#include <Gaudi/Parsers/Factory.h>
#include <GaudiKernel/DataObjID.h>
#include <GaudiKernel/DataObjectHandle.h>

#include "LHCbAlgs/Transformer.h"
#include "Event/RawEvent.h"

#include "InputProvider.h"
#include "MultiEventContextExt.h"

namespace Gaudi::Parsers {
  inline StatusCode parse(std::map<std::string, DataObjID>& m, std::string_view in)
  {
    m.clear();

    // the first element is branchName and the second on is tesPath
    std::map<std::string, std::string> ms;
    return parse(ms, in).andThen([&m, &ms]() -> StatusCode {
      try {
        std::ranges::transform(ms, std::inserter(m, m.end()), [](const auto& p) {
          DataObjID id;
          parse(id, p.second).orThrow("bad parse");
          return std::pair {p.first, id};
        });
        return StatusCode::SUCCESS;
      } catch (GaudiException const& e) {
        return e.code();
      }
    });
  };
} // namespace Gaudi::Parsers

class ProvideEventBranches final : public LHCb::Algorithm::Transformer<LHCb::RawEvent(const EventContext&)> {
  using LHCb::Algorithm::Transformer<LHCb::RawEvent(const EventContext&)>::Transformer;

public:
  ProvideEventBranches(const std::string& name, ISvcLocator* pSvcLocator) :
    Transformer(name, pSvcLocator, KeyValue {"RawEventLocation", ""})
  {}

  StatusCode initialize() override
  {
    return Transformer::initialize().andThen([&]() {
      // Create dynamically DataHandles for each branch requested
      for (auto& [branch, tesPath] : m_eventBranches) {
        // add a new DataHandle for this branch
        m_dataHandles.emplace_back(tesPath, Gaudi::DataHandle::Writer, this);
      }
    });
  }

  LHCb::RawEvent operator()(const EventContext& evtCtx) const override
  {
    const auto* ctxExt = Allen::Scheduler::getSchedulerExtension(evtCtx);

    auto eventData = m_input_provider->getEventBranches(ctxExt->slice_index, evtCtx.evt());
    assert(eventData.size() == m_dataHandles.size());

    for (unsigned i = 0; i < eventData.size(); i++) {
      m_dataHandles[i].put(std::unique_ptr<DataObject>(eventData[i]));
    }

    return m_input_provider->getRawEvent(ctxExt->slice_index, evtCtx.evt());
  }

private:
  ServiceHandle<IInputProviderSvc> m_input_provider {this, "InputProviderService", "MDFProvider"};

  // vector of datahandles created dynamically in initialize, one per requested branch
  std::vector<DataObjectHandle<DataObject>> m_dataHandles;

  Gaudi::Property<std::string> m_eventTreeName {
    this,
    "EventTreeName",
    "Event",
    "Name of the tree containing Events in the Root files"};
  // The twin of m_eventBranchesMap, to ducoment the index of branches in other vectors
  std::vector<std::pair<std::string, DataObjID>> m_eventBranches;
  Gaudi::Property<std::map<std::string, DataObjID>> m_eventBranchesMap {
    this,
    "EventBranches",
    {},
    [this](auto const&) {
      std::set<DataObjID> seen;
      m_eventBranches.clear();
      for (const auto& [branch, objID] : m_eventBranchesMap.value()) {
        // check the condition of bijectective
        // only allow one TES path used for one time in the memory
        auto [it, inserted] = seen.insert(objID);
        if (!inserted) {
          throw GaudiException(
            "Non-invertible EventBranchesMap detected: Branch " + branch, "ProvideEventBranches", StatusCode::FAILURE);
        }

        // create the twin of m_eventBranchesMap
        m_eventBranches.emplace_back(branch, objID);
      }
      debug() << "Updated EventBranches: " << m_eventBranches.size() << " entries." << endmsg;
    },
    "Map branch property name -> TES location to be retrieved from the Root files"};
};

DECLARE_COMPONENT(ProvideEventBranches)
