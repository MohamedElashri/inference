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
#pragma once

#ifndef ALLEN_STANDALONE

#include <algorithm>
#include <cstdio>
#include <BackendCommon.h>
#include <Datatype.cuh>
#include <Argument.cuh>
#include <BankTypes.h>
#include <AllenTypeTraits.h>
#include <Store.cuh>
#include <GaudiKernel/StatusCode.h>
#include <Gaudi/Parsers/Factory.h>
#include <GaudiKernel/StdArrayAsProperty.h>
#include <Gaudi/Algorithm.h>
#include <GaudiKernel/FunctionalFilterDecision.h>
#include <Kernel/EventLocalAllocator.h>

namespace Allen {
  // Helper to extract the type contained in a DataObjectHandle
  template<typename Handle>
  struct handle_type_extractor {};

  template<typename T>
  struct handle_type_extractor<DataObjectReadHandle<T>> {
    using type = T;
  };

  template<typename T>
  struct handle_type_extractor<DataObjectWriteHandle<T>> {
    using type = T;
  };

  // Filter tuple by trait
  template<typename Tuple, template<typename> class Trait>
  struct filter_tuple;

  template<typename... Types, template<typename> class Trait>
  struct filter_tuple<std::tuple<Types...>, Trait> {
    using type =
      decltype(std::tuple_cat(std::conditional_t<Trait<Types>::value, std::tuple<Types>, std::tuple<>> {}...));
  };

  // Make handles tuple
  template<typename ParamsTuple, template<typename> class Handle>
  struct make_handles;

  template<typename... Params, template<typename> class Handle>
  struct make_handles<std::tuple<Params...>, Handle> {
    using type = std::tuple<Handle<shared_buffer<Params::scope, std::remove_const_t<typename Params::type>>>...>;

    template<typename Algorithm>
    static type create([[maybe_unused]] Algorithm* algo)
    {
      return {std::make_tuple(algo, Params::name.data(), "")...};
    }
  };

  template<typename ParamsTuple, template<typename> class Handle>
  struct make_aggregate_handles;

  template<typename... Params, template<typename> class Handle>
  struct make_aggregate_handles<std::tuple<Params...>, Handle> {
    using type = std::tuple<Handle<shared_buffer<Params::scope, std::remove_const_t<typename Params::type::type>>>...>;

    template<typename Algorithm>
    static type create([[maybe_unused]] Algorithm* algo)
    {
      return {std::make_tuple(algo, Params::name.data(), "")...};
    }
  };

  // Find index of type in tuple
  template<typename Tuple, typename Type, std::size_t I = 0>
  constexpr std::size_t tuple_index_of()
  {
    if constexpr (I >= std::tuple_size_v<Tuple>) {
      static_assert(I < std::tuple_size_v<Tuple>, "Type not found in tuple");
      return I;
    }
    else if constexpr (std::is_same_v<std::tuple_element_t<I, Tuple>, Type>) {
      return I;
    }
    else {
      return tuple_index_of<Tuple, Type, I + 1>();
    }
  }

  // Compile-time index finder
  template<typename Tuple, std::size_t I = 0>
  constexpr std::size_t find_mask_index()
  {
    if constexpr (I >= std::tuple_size_v<Tuple>) {
      return I; // Not found
    }
    else {
      using T = std::tuple_element_t<I, Tuple>;
      if constexpr (std::is_same_v<typename T::type, mask_t>) {
        // Check for duplicates at this level
        static_assert(
          []<std::size_t... J>(std::index_sequence<J...>) {
            return ((J == I || !std::is_same_v<typename std::tuple_element_t<J, Tuple>::type, mask_t>) &&...);
          }(std::make_index_sequence<std::tuple_size_v<Tuple>> {}),
          "Multiple mask_t parameters found - only one allowed");
        return I;
      }
      else {
        return find_mask_index<Tuple, I + 1>();
      }
    }
  }

  template<typename Algorithm>
  struct parameters_from_algorithm {
    // Helper to extract Parameters from set_arguments_size signature
    template<typename Ret, typename Class, typename Arg1, typename Arg2, typename Arg3>
    static Arg1 extract_set_args_sig(Ret (Class::*)(Arg1, Arg2, Arg3) const);

    // Get the type of the first argument of set_arguments_size
    using set_args_first_arg = decltype(extract_set_args_sig(&Algorithm::set_arguments_size));

    // Now extract Parameters from ArgumentReferences<Parameters>
    template<typename>
    struct extract_param;

    template<typename U>
    struct extract_param<ArgumentReferences<U>> {
      using type = U;
    };

    using type = typename extract_param<set_args_first_arg>::type;
  };

  /**
  * @brief A TES wrapper. Allen TES objects are stored as std::vectors (of non-boolean types),
            and TES wrappers provide the Allen syntax on top of these.
  */
  template<Store::Kind K, Store::Scope S, typename T>
  struct TESWrapperArgument : public Store::BaseArgument {
  private:
    using vector_t = std::conditional_t<K == Store::Kind::Input, const shared_buffer<S, T>, shared_buffer<S, T>>;
    vector_t& m_data;

  protected:
    void* pointer() const override final
    {
      return const_cast<void*>(reinterpret_cast<forward_type_t<vector_t, void>*>(m_data.data()));
    }

    size_t size() const override final { return m_data.size(); }

  public:
    TESWrapperArgument(vector_t& data, const std::string& name) :
      Store::BaseArgument {std::in_place_type<T>, name, S}, m_data(data)
    {}

    const auto& get_shared_buffer() const { return m_data; }

    // set_pointer should never used, since vectors are allocated directly with set_size
    void set_pointer(void*) override final { throw; }

    // set_size resizes the vector of data, when it is an output datatype (not const)
    // If it is invoked on a const vector, it throws
    void set_size([[maybe_unused]] size_t size) override final
    {
      if constexpr (K == Store::Kind::Output) {
        m_data.resize(size);
      }
      else {
        // Note: static_assert wouldn't work here due to override
        throw;
      }
    }
  };

  // Shortcuts for input / output wrappers
  template<typename Param>
  using TESWrapperInput =
    TESWrapperArgument<Store::Kind::Input, Param::scope, std::remove_const_t<typename Param::type>>;

  template<typename Param>
  using TESWrapperInputAggregate =
    TESWrapperArgument<Store::Kind::Input, Param::scope, std::remove_const_t<typename Param::type::type>>;

  template<typename Param>
  using TESWrapperOutput =
    TESWrapperArgument<Store::Kind::Output, Param::scope, std::remove_const_t<typename Param::type>>;
} // namespace Allen

#endif // ndef ALLEN_STANDALONE
