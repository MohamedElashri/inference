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

// Gaudi
#include "GaudiAlg/Consumer.h"
#include "LHCbAlgs/Traits.h"

// Allen
#include "ConstantsCondition.h"

class ProvideConstants final
  : public Gaudi::Functional::Consumer<void(Constants const&), LHCb::Algorithm::Traits::usesConditions<Constants>> {

public:
  ProvideConstants(const std::string& name, ISvcLocator* pSvcLocator) :
    Consumer(
      name,
      pSvcLocator,
      {{"ConstantsConditionLocation", Allen::Conditions::ConstantsCondition::DefaultLocation}})
  {}

  StatusCode initialize() override
  {
    return Consumer::initialize().andThen(
      [&] { return Allen::Conditions::ConstantsCondition::addConditionDerivation(this); });
  }

  void operator()(Constants const&) const override {}
};

DECLARE_COMPONENT(ProvideConstants)
