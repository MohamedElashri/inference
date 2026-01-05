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
#include "LHCbAlgs/Consumer.h"
#include <vector>
#include "Kernel/STLExtensions.h"
#include <GaudiKernel/Environment.h>
#include "AllenMonitoring.h"
#include "MVAModelsManager.h"

class GaudiAllenMonitoringAlg : public LHCb::Algorithm::Consumer<void()> {
public:
  // Standard constructor
  GaudiAllenMonitoringAlg(const std::string& name, ISvcLocator* pSvcLocator) :
    Consumer {name,
              pSvcLocator,
              // Outputs
              {}}
  {}

  StatusCode initialize() override
  {
    const StatusCode sc = LHCb::Algorithm::Consumer<void()>::initialize();
    if (sc.isFailure()) return sc;
    Allen::Monitoring::AccumulatorManager::get()->initAccumulators(1);
    std::string cached_root;
    System::resolveEnv("${PARAMFILESROOT}", cached_root).orThrow("ParamFileSvc", "Cannot resolve ${PARAMFILESROOT}");
    Allen::MVAModels::MVAModelsManager::get()->loadData((cached_root + "/data").c_str());
    return sc;
  }

  void operator()() const override {}
};

DECLARE_COMPONENT(GaudiAllenMonitoringAlg)
