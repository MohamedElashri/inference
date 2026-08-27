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

#include <any>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <map>

#include <Event/ODIN.h>
#include <Constants.cuh>

namespace Allen::NonEventData {
  /** @class IUpdater IUpdater.h Dumpers/IUpdater.h
   *  Interface - shared with Allen - to the manager of Producer and Consumer
   *  of binary non-event data.
   *
   *  @author Roel Aaij
   *  @date   2019-05-27
   */
  class IUpdater {
  public:
    virtual ~IUpdater() {}

    virtual void update(std::span<unsigned const> odin_data) = 0;

    Constants& getConstants() { return m_constants; }

    template<typename T>
    void update_constants(T&& cond)
    {
      cond.update_constants(m_constants);
#ifdef ALLEN_STANDALONE
      // T are movable but not copyable, so wrap them in a shared pointer to make any happy
      m_conditions.emplace(T::id, std::any(std::make_shared<T>(std::forward<T>(cond))));
#endif
    }

    void release_buffers()
    {
#ifdef ALLEN_STANDALONE
      m_conditions.clear();
#endif
    }

  private:
    Constants m_constants {};
#ifdef ALLEN_STANDALONE
    std::map<std::string, std::any> m_conditions {};
#endif
  };
} // namespace Allen::NonEventData
