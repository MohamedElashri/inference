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

#include <tuple>
#include <span>
#include <vector>
#include <optional>
#include <cstring>
#include <unordered_map>
#include "BackendCommon.h"
#include "Logger.h"
#include "AllenTypeTraits.h"
#include "Argument.cuh"
#include "Datatype.cuh"
#include "MemoryManager.cuh"
#include "AllenBuffer.cuh"
#include <boost/pfr/core.hpp>

namespace Allen::Store {
  /**
   * @brief Allen argument manager
   */
  class UnorderedStore {
    device_memory_manager_t m_device_memory_manager {"Device memory manager"};
    host_memory_manager_t m_host_memory_manager {"Host memory manager"};
    std::unordered_map<std::string, AllenArgument> m_store {};

    SlabAllocator<Allen::details::shared_buffer_metadata> meta_allocator {};
    SlabAllocator<Allen::details::type_erased_dependency> dep_allocator {};

  public:
    UnorderedStore() = default;
    UnorderedStore(const UnorderedStore&) = delete;
    UnorderedStore& operator=(const UnorderedStore&) = delete;
    UnorderedStore(UnorderedStore&&) = delete;
    UnorderedStore& operator=(UnorderedStore&&) = delete;

    template<Scope S, typename T>
    auto make_buffer(const size_t size)
    {
      // TODO: get actual context
      return Allen::shared_buffer<S, T> {
        size,
        memory_managers_t {&m_host_memory_manager, &m_device_memory_manager, &meta_allocator, &dep_allocator, {}}};
    }

    AllenArgument& at(const std::string& k)
    {
      if (m_store.find(k) != std::end(m_store)) {
        return m_store.at(k);
      }
      throw std::runtime_error("store does not contain key " + k);
    }

    const AllenArgument& at(const std::string& k) const
    {
      if (m_store.find(k) != std::end(m_store)) {
        return m_store.at(k);
      }
      throw std::runtime_error("store does not contain key " + k);
    }

    void register_entry(const std::string& k, AllenArgument&& arg)
    {
      auto ret = m_store.try_emplace(k, std::forward<AllenArgument>(arg));
      if (!ret.second) {
        throw std::runtime_error("store register_entry of " + k + " failed, entry already exists");
      }
    }

    void put(const std::string& k)
    {
      AllenArgument& arg = at(k);
      if (arg.scope() == Allen::Store::Scope::Host) {
        m_host_memory_manager.reserve(arg);
      }
      else if (arg.scope() == Allen::Store::Scope::Device) {
        m_device_memory_manager.reserve(arg);
      }
      else {
        throw std::runtime_error("argument scope not recognized");
      }
    }

    void reserve_memory(
      const size_t device_requested_mb,
      const size_t host_requested_mb,
      const unsigned required_memory_alignment)
    {
      m_device_memory_manager.reserve_memory(device_requested_mb * 1024 * 1024, required_memory_alignment);
      m_host_memory_manager.reserve_memory(host_requested_mb * 1024 * 1024, 512);
    }

    void free(const std::string& k)
    {
      auto& arg = at(k);
      if (arg.scope() == Allen::Store::Scope::Host) {
        m_host_memory_manager.free(arg);
      }
      else if (arg.scope() == Allen::Store::Scope::Device) {
        m_device_memory_manager.free(arg);
      }
      arg.set_pointer(nullptr);
    }

    void free_all()
    {
      m_device_memory_manager.free_all();
      m_host_memory_manager.free_all();
    }

    void reset()
    {
      m_store.clear();
      free_all();
    }

    void print_memory_manager_states() const { m_device_memory_manager.print(); }
  };

  /**
   * @brief Metaprogramming to extract ::type from each aggregated type.
   */
  template<typename Tuple>
  struct AggregateTypes;

  template<>
  struct AggregateTypes<std::tuple<>> {
    using aggregates_tuple_type_t = std::tuple<>;
  };

  template<typename T, typename... Ts>
  struct AggregateTypes<std::tuple<T, Ts...>> {
    using aggregates_tuple_type_t =
      prepend_to_tuple_t<typename T::type, typename AggregateTypes<std::tuple<Ts...>>::aggregates_tuple_type_t>;
  };

  /**
   * @brief Manager of argument references for every handler.
   */
  template<
    typename UnalteredParameterTuple,
    typename ParameterTuple,
    typename ParameterStruct,
    typename InputAggregatesTuple = std::tuple<>>
  struct StoreRef {
  public:
    using unaltered_parameters_tuple_t = UnalteredParameterTuple;
    using parameters_tuple_t = ParameterTuple;
    using aggregates_tuple_t = InputAggregatesTuple;
    using parameters_struct_t = ParameterStruct;
    using input_aggregates_t = typename AggregateTypes<InputAggregatesTuple>::aggregates_tuple_type_t;
    using arguments_t = std::array<std::reference_wrapper<BaseArgument>, std::tuple_size_v<parameters_tuple_t>>;

  private:
    mutable arguments_t m_arguments;
    input_aggregates_t m_input_aggregates;
    UnorderedStore* m_store = nullptr;
    const memory_managers_t* m_memory_managers = nullptr;

    template<typename T, std::enable_if_t<!std::is_base_of_v<aggregate_datatype, T>, bool> = true>
    decltype(m_arguments[index_of_v<T, parameters_tuple_t>].get()) arg() const
    {
      constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
      static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
      return m_arguments[index_of_T].get();
    }

  public:
    StoreRef(arguments_t arguments, input_aggregates_t input_aggregates, UnorderedStore& store) :
      m_arguments(arguments), m_input_aggregates(input_aggregates), m_store(&store)
    {}

    StoreRef(arguments_t arguments, input_aggregates_t input_aggregates, const memory_managers_t& memory_managers) :
      m_arguments(arguments), m_input_aggregates(input_aggregates), m_memory_managers(&memory_managers)
    {}

    template<Scope S, typename T>
    auto make_buffer(const size_t size) const
    {
      using type = std::remove_const_t<T>;
      if (m_memory_managers) {
        return Allen::shared_buffer<S, type> {size, *m_memory_managers};
      }
      return m_store->make_buffer<S, type>(size); // if no manager, we can expect a store
    }

    template<typename T>
    std::span<typename T::type> get() const
    {
      return arg<T>();
    }

    template<typename T>
    auto data() const
    {
      return get<T>().data();
    }

    template<typename T>
    auto first() const
    {
      static_assert(std::is_base_of_v<host_datatype, T> && "first can only access host datatypes");
      return get<T>()[0];
    }

    template<typename T>
    auto size() const
    {
      return get<T>().size();
    }

    template<typename T>
    auto size_bytes() const
    {
      return size<T>() * sizeof(T);
    }

    template<typename T>
    void set_size(const size_t size)
    {
      static_assert(!Allen::Store::is_input<T>::value && "set_size can only be used on output datatypes");
      arg<T>().set_size(size);
    }

    template<typename T>
    void resize(const size_t size) const
    {
      static_assert(!Allen::Store::is_input<T>::value && "resize can only be used on output datatypes");
      if (m_store) {
        m_store->free(name<T>());
      }
      arg<T>().set_size(size);
      if (m_store) {
        m_store->put(name<T>());
      }
    }

    /**
     * @brief Reduces the size of the container.
     * @details Reducing the size can be done in the operator(), hence this method is const.
     */
    template<typename T>
    void reduce_size(const size_t size) const
    {
      static_assert(!Allen::Store::is_input<T>::value && "reduce_size can only be used on output datatypes");
      assert(size <= get<T>().size());
      arg<T>().set_size(size);
    }

    template<typename T>
    std::string name() const
    {
      return arg<T>().name();
    }

    template<typename T, std::enable_if_t<std::is_base_of_v<aggregate_datatype, T>, bool> = true>
    auto input_aggregate() const
    {
      return std::get<index_of_v<T, InputAggregatesTuple>>(m_input_aggregates);
    }
  };

  /**
   * @brief Tuple wrapper that extracts tuples out of the
   *        Parameters struct. It extracts a tuple of parameters (parameters_tuple_t).
   */
  template<typename Tuple, typename I, typename Enabled = void>
  struct WrappedTupleDetails;

  template<typename Tuple>
  struct WrappedTupleDetails<Tuple, std::index_sequence<>> {
    using parameters_tuple_t = std::tuple<>;
    using aggregates_tuple_t = std::tuple<>;
    using unaltered_parameters_tuple_t = std::tuple<>;
  };

  template<typename Tuple, std::size_t I, std::size_t... Is>
  struct WrappedTupleDetails<
    Tuple,
    std::index_sequence<I, Is...>,
    std::enable_if_t<(
      std::is_base_of_v<device_datatype, boost::pfr::tuple_element_t<I, Tuple>> ||
      std::is_base_of_v<host_datatype, boost::pfr::tuple_element_t<I, Tuple>>) &&!std::
                       is_base_of_v<aggregate_datatype, boost::pfr::tuple_element_t<I, Tuple>>>> {
    using prev_wrapped_tuple = WrappedTupleDetails<Tuple, std::index_sequence<Is...>>;
    using prev_parameters_tuple_t = typename prev_wrapped_tuple::parameters_tuple_t;
    using parameters_tuple_t = prepend_to_tuple_t<boost::pfr::tuple_element_t<I, Tuple>, prev_parameters_tuple_t>;
    using aggregates_tuple_t = typename prev_wrapped_tuple::aggregates_tuple_t;
    using unaltered_parameters_tuple_t = prepend_to_tuple_t<
      boost::pfr::tuple_element_t<I, Tuple>,
      typename prev_wrapped_tuple::unaltered_parameters_tuple_t>;
  };

  template<typename Tuple, std::size_t I, std::size_t... Is>
  struct WrappedTupleDetails<
    Tuple,
    std::index_sequence<I, Is...>,
    std::enable_if_t<std::is_base_of_v<aggregate_datatype, boost::pfr::tuple_element_t<I, Tuple>>>> {
    using prev_wrapped_tuple = WrappedTupleDetails<Tuple, std::index_sequence<Is...>>;
    using parameters_tuple_t = typename prev_wrapped_tuple::parameters_tuple_t;
    using aggregates_tuple_t =
      prepend_to_tuple_t<boost::pfr::tuple_element_t<I, Tuple>, typename prev_wrapped_tuple::aggregates_tuple_t>;
    using unaltered_parameters_tuple_t = prepend_to_tuple_t<
      boost::pfr::tuple_element_t<I, Tuple>,
      typename prev_wrapped_tuple::unaltered_parameters_tuple_t>;
  };

  template<typename Tuple, std::size_t I, std::size_t... Is>
  struct WrappedTupleDetails<
    Tuple,
    std::index_sequence<I, Is...>,
    std::enable_if_t<
      !std::is_base_of_v<device_datatype, boost::pfr::tuple_element_t<I, Tuple>> &&
      !std::is_base_of_v<host_datatype, boost::pfr::tuple_element_t<I, Tuple>> &&
      !std::is_base_of_v<aggregate_datatype, boost::pfr::tuple_element_t<I, Tuple>>>> {
    using prev_wrapped_tuple = WrappedTupleDetails<Tuple, std::index_sequence<Is...>>;
    using parameters_tuple_t = typename prev_wrapped_tuple::parameters_tuple_t;
    using aggregates_tuple_t = typename prev_wrapped_tuple::aggregates_tuple_t;
    using unaltered_parameters_tuple_t = prepend_to_tuple_t<
      boost::pfr::tuple_element_t<I, Tuple>,
      typename prev_wrapped_tuple::unaltered_parameters_tuple_t>;
  };

  template<size_t... Is>
  auto gen_input_aggregates_tuple(
    const std::vector<std::vector<std::reference_wrapper<BaseArgument>>>& input_aggregates,
    std::index_sequence<Is...>)
  {
    return std::make_tuple(input_aggregates[Is]...);
  }

  template<typename T>
  using WrappedTuple = WrappedTupleDetails<T, std::make_index_sequence<boost::pfr::tuple_size_v<T>>>;
} // namespace Allen::Store

template<typename T>
using ArgumentReferences = Allen::Store::StoreRef<
  typename Allen::Store::WrappedTuple<T>::unaltered_parameters_tuple_t,
  typename Allen::Store::WrappedTuple<T>::parameters_tuple_t,
  T,
  typename Allen::Store::WrappedTuple<T>::aggregates_tuple_t>;
