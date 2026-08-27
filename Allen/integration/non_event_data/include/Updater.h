/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <string>
#include <map>
#include <memory>

#include "Dumpers/IUpdater.h"
#include <Event/ODIN.h>

namespace Allen::NonEventData {
  class Updater final : public IUpdater {
  public:
    virtual ~Updater() = default;
    void update(std::span<unsigned const>) override {};
  };
} // namespace Allen::NonEventData
