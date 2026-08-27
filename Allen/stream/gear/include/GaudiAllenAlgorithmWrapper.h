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

#include <vector>

#include <Gaudi/Algorithm.h>
#include <GaudiKernel/FunctionalFilterDecision.h>
#include <GaudiKernel/Environment.h>
#include <DetDesc/GenericConditionAccessorHolder.h>

#include "AlgorithmConversionTools.h"
#include "AggregateHandle.h"
#include "AllenMonitoring.h"
#include "MVAModelsManager.h"

namespace Allen::Conditions::ConstantsCondition {
#ifdef USE_DD4HEP
  inline static std::string const DefaultLocation = "/world:AllenConditions-constants";
#else
  inline static std::string const DefaultLocation = "AllenConditions-constants";
#endif
} // namespace Allen::Conditions::ConstantsCondition

using namespace Gaudi::Functional;
using namespace LHCb::DetDesc;

struct Dummy {};

template<typename AllenAlgorithm>
class GaudiAllenAlgorithmWrapper final : public AlgorithmWithCondition<> {
public:
  using Algorithm = AlgorithmWithCondition<>;

  using Parameters = typename Allen::parameters_from_algorithm<AllenAlgorithm>::type;

  using parameters_tuple_t = typename Allen::Store::WrappedTuple<Parameters>::parameters_tuple_t;
  using aggregates_tuple_t = typename Allen::Store::WrappedTuple<Parameters>::aggregates_tuple_t;

  using inputs_tuple_t = typename Allen::filter_tuple<parameters_tuple_t, Allen::Store::is_input>::type;
  using outputs_tuple_t = typename Allen::filter_tuple<parameters_tuple_t, Allen::Store::is_output>::type;

  GaudiAllenAlgorithmWrapper(std::string name, ISvcLocator* pSvcLocator) : Algorithm(std::move(name), pSvcLocator)
  {
    for (auto& prop : m_algorithm.properties()) {
      prop->register_as_gaudi_property(this);
    }
  }

  StatusCode initialize() override
  {
    const StatusCode sc = Algorithm::initialize();
    if (sc.isFailure()) return sc;
    Allen::initialize_algorithm(m_algorithm);
    m_algorithm.set_name(this->name());

#ifdef USE_DD4HEP
    std::string key = std::string {"/world:AlgorithmSpecific-"} + this->name() + "-update";
#else
    std::string key = std::string {"AlgorithmSpecific-"} + this->name() + "-update";
#endif

    addConditionDerivation(
      {Allen::Conditions::ConstantsCondition::DefaultLocation},
      std::move(key),
      [this]([[maybe_unused]] Constants const& constants) {
        m_algorithm.update(constants);
        return Dummy {}; // returning basic type here (eg. boolean) result in a segfault for some reason..
      });
    return sc;
  }

  StatusCode start() override
  {
    const StatusCode sc = Algorithm::start();
    if (sc.isFailure()) return sc;
    Allen::Monitoring::AccumulatorManager::get()->initAccumulators(1);
    Allen::MVAModels::MVAModelsManager::get()->loadData((m_cached_root + "/data").c_str());
    return sc;
  }

  StatusCode stop() override
  {
    Allen::Monitoring::AccumulatorManager::get()->mergeAndReset(true);
    const StatusCode sc = Algorithm::stop();
    if (sc.isFailure()) return sc;
    return sc;
  }

private:
  AllenAlgorithm m_algorithm {};
  std::string m_cached_root;
  Gaudi::Property<std::string> m_root {
    this,
    "Root",
    "${PARAMFILESROOT}",
    [this](auto const&) {
      [[maybe_unused]] auto sc =
        System::resolveEnv(m_root, m_cached_root).orThrow("ParamFileSvc", "Cannot resolve  " + m_root);
    },
    Gaudi::Details::Property::ImmediatelyInvokeHandler {true}};

  Gaudi::Property<bool> m_isMultiEvent {this, "IsMultiEvent", true, ""};

  // Data handles:
  DataObjectReadHandle<RuntimeOptions> m_runtime_options {this, "runtime_options_t", ""};

  ConditionAccessor<Constants> m_constants {
    this,
    "ConstantsCondition",
    Allen::Conditions::ConstantsCondition::DefaultLocation};

  Allen::make_handles<inputs_tuple_t, DataObjectReadHandle>::type m_inputs {
    Allen::make_handles<inputs_tuple_t, DataObjectReadHandle>::create(this)};
  Allen::make_handles<outputs_tuple_t, DataObjectWriteHandle>::type m_outputs {
    Allen::make_handles<outputs_tuple_t, DataObjectWriteHandle>::create(this)};
  Allen::make_aggregate_handles<aggregates_tuple_t, Allen::AggregateReadHandle>::type m_aggregates {
    Allen::make_aggregate_handles<aggregates_tuple_t, Allen::AggregateReadHandle>::create(this)};

public:
  StatusCode execute(const EventContext& evtCtx) const override
  {
    auto const& runtime_options = *m_runtime_options.get();
    auto const& constants = m_constants.get(getConditionContext(evtCtx));
    Allen::Context context {};

    // Output container
    auto output_container = std::apply(
      [&](const auto&... handles) {
        [[maybe_unused]] auto create_vector = [&]<typename Handle>(const Handle&) {
          using vector_t = typename Allen::handle_type_extractor<Handle>::type;
          using data_t = typename vector_t::value_type;
          return Allen::parameter_vector<data_t> {LHCb::getMemResource(evtCtx)};
        };
        return std::make_tuple(create_vector(handles)...);
      },
      m_outputs);

    // Aggregates TES wrappers
    auto input_aggregates_wrappers = [&]<std::size_t... I>(std::index_sequence<I...>) {
      std::tuple<Allen::parameter_vector<
        Allen::TESWrapperInput<typename std::tuple_element_t<I, aggregates_tuple_t>::type::type>>...>
        wrappers_vecs {};
      // Fill wrappers for each aggregate:
      (
        [&] {
          using Param = std::tuple_element_t<I, aggregates_tuple_t>;
          using Aggregate = typename Param::type;

          Allen::parameter_vector<typename Aggregate::type> empty_vector {LHCb::getMemResource(evtCtx)};
          const auto& handles = std::get<I>(m_aggregates).handles;

          auto& wrappers_vec = std::get<I>(wrappers_vecs);
          wrappers_vec.reserve(handles.size());
          for (const auto& handle : handles) {
            auto* inp = handle.getIfExists();
            wrappers_vec.emplace_back(inp ? *inp : empty_vector, Param::name.data());
          }
        }(),
        ...);
      return wrappers_vecs;
    }(std::make_index_sequence<std::tuple_size_v<aggregates_tuple_t>> {});

    auto input_aggregates_tuple = [&]<std::size_t... I>(std::index_sequence<I...>) {
      // Finally return the tuple of aggregates
      return std::make_tuple([&] {
        using Aggregate = std::tuple_element_t<I, aggregates_tuple_t>::type;
        std::vector<std::reference_wrapper<Allen::Store::BaseArgument>> refs;
        auto& vecs = std::get<I>(input_aggregates_wrappers);
        refs.reserve(vecs.size());
        for (auto& w : vecs)
          refs.emplace_back(w);
        return Aggregate {refs};
      }()...);
    }(std::make_index_sequence<std::tuple_size_v<aggregates_tuple_t>> {});

    // TES wrappers
    auto tes_wrappers = [&]<std::size_t... I>(std::index_sequence<I...>) {
      [[maybe_unused]] auto make_wrapper = [&]<std::size_t J>() -> auto {
        using Param = std::tuple_element_t<J, parameters_tuple_t>;
        if constexpr (Allen::Store::is_input<Param>::value) {
          return Allen::TESWrapperInput<typename Param::type> {
            *std::get<Allen::tuple_index_of<inputs_tuple_t, Param>()>(m_inputs).get(), Param::name.data()};
        }
        else {
          return Allen::TESWrapperOutput<typename Param::type> {
            std::get<Allen::tuple_index_of<outputs_tuple_t, Param>()>(output_container), Param::name.data()};
        }
      };
      return std::make_tuple(make_wrapper.template operator()<I>()...);
    }(std::make_index_sequence<std::tuple_size_v<parameters_tuple_t>> {});

    auto tes_wrappers_references = std::apply(
      [](auto&... wrappers) {
        return std::array<std::reference_wrapper<Allen::Store::BaseArgument>, sizeof...(wrappers)> {wrappers...};
      },
      tes_wrappers);

    const auto argument_references = ArgumentReferences<Parameters> {tes_wrappers_references, input_aggregates_tuple};

    // set arguments size invocation
    m_algorithm.set_arguments_size(argument_references, runtime_options, constants);

    // algorithm operator() invocation
    m_algorithm(argument_references, runtime_options, constants, context);

    // Get filter decision
    const auto filter_decision = [&]<std::size_t I = Allen::find_mask_index<decltype(m_outputs)>()>()
    {
      if constexpr (I < std::tuple_size_v<decltype(m_outputs)>) {
        // mask_t exists - check size at runtime
        return std::get<I>(output_container).size() ? FilterDecision::PASSED : FilterDecision::FAILED;
      }
      else {
        // No mask_t found
        return FilterDecision::PASSED;
      }
    }
    ();

    // Move outputs to the TES:
    std::apply(
      [&](auto&... out_handle) {
        std::apply(
          [&out_handle...](auto&&... out_data) { (out_handle.put(std::move(out_data)), ...); }, output_container);
      },
      m_outputs);

    return filter_decision;
  }
};

namespace {
  constexpr const char* strip_namespace(const char* full_name)
  {
    const char* last_colon = nullptr;
    const char* p = full_name;

    // Find the last occurrence of "::"
    while (*p) {
      if (*p == ':' && *(p + 1) == ':') {
        last_colon = p;
        p += 2; // Skip past "::"
      }
      else {
        ++p;
      }
    }

    // If we found "::", return the character after it, otherwise return the original string
    return last_colon ? last_colon + 2 : full_name;
  }
} // namespace

#define MAKE_ALLEN_GAUDI_WRAPPER(ALGORITHM) \
  DECLARE_COMPONENT_WITH_ID(GaudiAllenAlgorithmWrapper<ALGORITHM>, strip_namespace(#ALGORITHM))

#define MAKE_ALLEN_GAUDI_WRAPPER_WITH_ID(ALGORITHM, ID) \
  DECLARE_COMPONENT_WITH_ID(GaudiAllenAlgorithmWrapper<ALGORITHM>, ID)

#else

#define MAKE_ALLEN_GAUDI_WRAPPER(ALGORITHM)
#define MAKE_ALLEN_GAUDI_WRAPPER_WITH_ID(ALGORITHM, ID)

#endif
