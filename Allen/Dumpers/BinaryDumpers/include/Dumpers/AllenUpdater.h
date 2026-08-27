/*****************************************************************************\
* (c) Copyright 2019 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <map>
#include <memory>
#include <string>
#include <mutex>

#include <GaudiKernel/Service.h>
#include <Gaudi/Interfaces/IQueueingEventProcessor.h>
#include <GaudiKernel/StdArrayAsProperty.h>
#include <DetDesc/GenericConditionAccessorHolder.h>
#include <GaudiKernel/DataHandleHolderBase.h>

#include <Event/ODIN.h>

#include <Dumpers/IUpdater.h>
#include <Dumpers/Utils.h>
#include <BankTypes.h>

#include <tbb/task_arena.h>

namespace {
  std::string resolveEnvVars(std::string s)
  {
    std::regex envExpr {"\\$\\{([A-Za-z0-9_]+)\\}"};
    std::smatch m;
    while (std::regex_search(s, m, envExpr)) {
      std::string rep;
      System::getEnv(m[1].str(), rep);
      s = s.replace(m[1].first - 2, m[1].second + 1, rep);
    }
    return s;
  }
} // namespace

using ServiceWithConditions = LHCb::DetDesc::ConditionAccessorHolder<DataHandleHolderBase<Service>>;

/** @class AllenUpdater AllenUpdater.h
 *  LHCb implementation of the Allen non-event data manager
 *
 *  @author Roel Aaij
 *  @date   2019-05-24
 */
class AllenUpdater final : public Allen::NonEventData::IUpdater, public ServiceWithConditions {
public:
  /// Retrieve interface ID
  static const InterfaceID& interfaceID()
  {
    // Declaration of the interface ID.
    static const InterfaceID iid("AllenUpdater", 0, 0);
    return iid;
  }

  /// Query interfaces of Interface
  StatusCode queryInterface(const InterfaceID& riid, void** ppv) override;

  AllenUpdater(std::string name, ISvcLocator* loc) : ServiceWithConditions {name, loc} {}

  void acceptDHVisitor(IDataHandleVisitor* vis) const override { vis->visit(this); }

  StatusCode initialize() override;

  void update(std::span<unsigned const> odin) override;

  std::array<float, 2> getBeamlineOffset() const { return m_beamline_offset.value(); }

  std::string getParamDir() const { return resolveEnvVars(m_paramDir); }

  LHCb::ODIN odin() const { return m_odin ? *m_odin : LHCb::ODIN {}; }

  template<typename T>
  std::size_t dump(const T& cond)
  {
    if (!m_dumpToFile) return 0;

    auto data = cond.data();
    auto filename = m_outputDirectory.value() + "/" + T::filename;
    std::ofstream output {filename, std::ios::out | std::ios::binary};
    output.write(data.data(), data.size());
    return data.size();
  }

private:
  Gaudi::Property<std::string> m_outputDirectory {this, "OutputDirectory", "geometry"};
  Gaudi::Property<bool> m_dumpToFile {this, "DumpToFile", false};

  Gaudi::Property<std::array<float, 2>> m_beamline_offset {this, "BeamlineOffset", {0.f, 0.f}, "Beamline offset"};

  // set this explicitly, must match with the Condition tags:
  Gaudi::Property<std::string> m_paramDir {this, "ParamDir", "${PARAMFILESROOT}"};

  Gaudi::Property<bool> m_triggerEventLoop {this, "TriggerEventLoop", false};

  SmartIF<Gaudi::Interfaces::IQueueingEventProcessor> m_evtProc;

  std::optional<LHCb::ODIN> m_odin = std::nullopt;
};
