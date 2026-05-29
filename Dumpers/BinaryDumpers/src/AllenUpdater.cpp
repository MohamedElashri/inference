/*****************************************************************************\
* (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <chrono>

#include <Dumpers/AllenUpdater.h>
#include <ConstantsCondition.h>

DECLARE_COMPONENT(AllenUpdater)

/// Query interfaces of Interface
StatusCode AllenUpdater::queryInterface(const InterfaceID& riid, void** ppv)
{
  if (AllenUpdater::interfaceID().versionMatch(riid)) {
    *ppv = this;
    addRef();
    return StatusCode::SUCCESS;
  }
  return Service::queryInterface(riid, ppv);
}

StatusCode AllenUpdater::initialize()
{
  if (m_triggerEventLoop.value()) {
    m_evtProc = serviceLocator()->service<Gaudi::Interfaces::IQueueingEventProcessor>("ApplicationMgr");
    if (!m_evtProc) {
      error() << "Failed to obtain ApplicationMgr as IQueueingEventProcessor" << endmsg;
      return StatusCode::FAILURE;
    }
  }

  if (m_dumpToFile.value() && !DumpUtils::createDirectory(m_outputDirectory.value())) {
    this->error() << "Failed to create directory " << m_outputDirectory.value() << endmsg;
    return StatusCode::FAILURE;
  }

  return StatusCode::SUCCESS;
}

void AllenUpdater::update(std::span<unsigned const> odin_data)
{
  LHCb::ODIN odin {odin_data};
  if (m_odin && m_odin->runNumber() == odin.runNumber()) {
    return;
  }
  else if (msgLevel(MSG::DEBUG)) {
    debug() << "Running Update " << odin.runNumber() << endmsg;
  }

  // Store ODIN so it can be retrieved and then inserted into the event store
  m_odin = odin;

  // Run the "fake" event loop to produce the new data
  if (m_triggerEventLoop.value()) {
    auto sc = m_evtProc->nextEvent(1);
    if (!sc.isSuccess()) {
      throw GaudiException {"Failed to process event for conditions update", name(), StatusCode::FAILURE};
    }
  }
}
