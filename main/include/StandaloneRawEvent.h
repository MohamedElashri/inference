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
#pragma once

#ifndef ALLEN_STANDALONE
#include <Event/RawEvent.h>
#else
#include <Event/RawBank.h>

// Since including RawEvent requires the full Gaudi framework,
// we define a (very) minimal RawEvent class here for use in standalone mode.
// Since we want to drop standalone, and that it is only used for preparing
// events before measuring througput, the performance of this class is not critical.
namespace LHCb {
  class RawEvent {
  public:
    [[nodiscard]] RawBank::View banks(RawBank::BankType bankType) const noexcept
    {
      return m_banks[static_cast<size_t>(bankType)];
    }

    [[nodiscard]] RawBank::View banks() const { return m_all_banks; }

    void adoptBank(const LHCb::RawBank* bank, bool /*adopt_memory*/)
    {
      m_banks[static_cast<size_t>(bank->type())].emplace_back(bank);
      m_all_banks.emplace_back(bank);
    }

  private:
    std::array<std::vector<const LHCb::RawBank*>, RawBank::types().size()> m_banks;
    std::vector<const LHCb::RawBank*> m_all_banks;
  };
} // namespace LHCb
#endif
