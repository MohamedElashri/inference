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

#include <Gaudi/Algorithm.h>

namespace Allen::details {
  template<typename Handles>
  Handles make_vector_of_handles(IDataHandleHolder* owner, const std::vector<DataObjID>& init)
  {
    Handles handles;
    handles.reserve(init.size());
    std::transform(
      init.begin(), init.end(), std::back_inserter(handles), [&](const auto& loc) -> typename Handles::value_type {
        return {loc, owner};
      });
    return handles;
  }
} // namespace Allen::details

namespace Allen {
  template<typename T>
  struct AggregateReadHandle {
    AggregateReadHandle(Gaudi::Algorithm* algo, std::string name, std::string) :
      locations {
        algo,
        name,
        {},
        [this, algo](Gaudi::Details::PropertyBase&) {
          handles = details::make_vector_of_handles<decltype(handles)>(algo, locations);
        },
        Gaudi::Details::Property::ImmediatelyInvokeHandler {true}}
    {}

    template<typename... Args, std::size_t... Is>
    AggregateReadHandle(std::tuple<Args...>&& args, std::index_sequence<Is...>) :
      AggregateReadHandle(std::get<Is>(std::move(args))...)
    {}

    template<typename... Args>
    AggregateReadHandle(std::tuple<Args...>&& args) :
      AggregateReadHandle(std::move(args), std::index_sequence_for<Args...> {})
    {}

    std::vector<DataObjectReadHandle<T>> handles;
    Gaudi::Property<std::vector<DataObjID>> locations;
  };
} // namespace Allen

#endif
