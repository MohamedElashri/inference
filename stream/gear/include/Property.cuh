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

#include "BackendCommon.h"
#include "BaseTypes.cuh"
#include "BankTypes.h"
#include "Logger.h"
#include "Common.h"
#include <string>
#include <map>
#include <list>
#include <set>
#include <regex>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <cxxabi.h>
#include <nlohmann/json.hpp>

#ifndef ALLEN_STANDALONE
#include <Gaudi/Algorithm.h>
#include <GaudiKernel/StdArrayAsProperty.h>

// Parsers are in namespace LHCb for ADL to work.
namespace Gaudi::Parsers {
  inline StatusCode parse(BankTypes& result, const std::string& in)
  {
    // This takes care of quoting
    std::string input;
    using Gaudi::Parsers::parse;
    auto sc = parse(input, in);
    if (!sc) return sc;

    result = bank_type(input);
    return StatusCode::SUCCESS;
  }

  inline StatusCode parse(dim3& result, const std::string& in)
  {
    std::array<unsigned, 3> input;
    using Gaudi::Parsers::parse;
    auto sc = parse(input, in);
    if (!sc) return sc;

    result = {input[0], input[1], input[2]};
    return StatusCode::SUCCESS;
  }
} // namespace Gaudi::Parsers

inline std::ostream& toStream(const BankTypes& bt, std::ostream& s)
{
  auto bn = bank_name(bt);
  return s << "'" << bn << "'";
}

inline std::ostream& toStream(const dim3& d, std::ostream& s)
{
  return Gaudi::Utils::toStream(std::array {d.x, d.y, d.z}, s);
}
#endif

void from_json(const nlohmann::json& j, dim3& d);
void to_json(nlohmann::json& j, const dim3& d);

namespace Allen {
  /**
   * @brief      Store and readout the value of a single configurable algorithm property
   *
   */
  template<typename V>
  class Property : public BaseProperty {
  public:
    Property(BaseAlgorithm* algo, const std::string& name, const V& default_value, const std::string& description) :
      BaseProperty {name, description, type_name()}, m_algo {algo}, m_cached_value {default_value}
    {
      algo->register_property(m_name, this);
    }

    V const& value() const { return m_cached_value; }

    void from_json(const nlohmann::json& value) override { nlohmann::from_json(value, m_cached_value); }

    nlohmann::json to_json() const override { return m_cached_value; }

    std::string to_string() const override
    {
      nlohmann::json j = m_cached_value;
      return j.dump();
    }

    operator V const&() const { return m_cached_value; }

    void set_value(const V& value) { m_cached_value = value; }

#ifndef ALLEN_STANDALONE
    Gaudi::Property<V> m_gaudi_prop;
    void register_as_gaudi_property(Gaudi::Algorithm* alg) override
    {
      m_gaudi_prop = {name(), value(), description()};
      alg->declareProperty(m_gaudi_prop);
      m_gaudi_prop.template setOwnerType<Gaudi::Algorithm>();
      m_gaudi_prop.declareUpdateHandler([this](auto&) { m_cached_value = m_gaudi_prop.value(); });
      m_gaudi_prop.useUpdateHandler();
    }
#endif

  private:
    std::string type_name()
    {
      int status;
      std::string tname = typeid(V).name();
      char* demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
      if (status == 0) {
        tname = demangled_name;
        std::free(demangled_name);
      }
      return tname;
    }

  private:
    BaseAlgorithm* m_algo = nullptr;
    V m_cached_value;
  };
} // namespace Allen
