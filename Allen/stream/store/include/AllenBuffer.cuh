/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <MemoryManager.cuh>
#include <span>
#include <variant>

namespace Allen {
  namespace details {
    /**
     * @brief Standalone backend for the buffer.
     * @details Uses the memory manager provided by the store
     *          to free and reserve the buffer.
     */
    template<Store::Scope S, typename T>
    struct standalone_buffer {
    private:
      Allen::Store::memory_manager_t<S>* m_mem_manager = nullptr;
      std::span<T> m_span {};
      bool m_allocated = false;

    public:
      __host__ standalone_buffer() {}
      __host__ standalone_buffer(Allen::Store::memory_manager_t<S>& mem_manager) : m_mem_manager(&mem_manager) {}
      __host__ standalone_buffer(Allen::Store::memory_manager_t<S>& mem_manager, size_t size) :
        m_mem_manager(&mem_manager), m_span(reinterpret_cast<T*>(m_mem_manager->reserve(size * sizeof(T))), size),
        m_allocated(true)
      {}
      __host__ standalone_buffer(standalone_buffer&& o) :
        m_mem_manager(o.m_mem_manager), m_span(o.m_span), m_allocated(o.m_allocated)
      {
        // Set o allocated to false to avoid the data being freed in the destructor of o
        o.m_allocated = false;
      }
      __host__ ~standalone_buffer()
      {
        if (m_allocated) {
          m_mem_manager->free(reinterpret_cast<char*>(m_span.data()));
        }
      }
      __host__ void resize(size_t size)
      {
        if (m_allocated) {
          m_mem_manager->free(reinterpret_cast<char*>(m_span.data()));
        }
        m_allocated = true;
        m_span = std::span<T> {reinterpret_cast<T*>(m_mem_manager->reserve(size * sizeof(T))), size};
      }
      __host__ std::span<T> get() { return m_span; }
      __host__ std::span<const T> get() const { return m_span; }
    };

    /**
     * @brief Non-standalone backend for the buffer, used in Gaudi sequencer.
     * @details Uses a std::vector as a backend.
     */
    template<typename T>
    struct nonstandalone_buffer {
    private:
      std::vector<bool_as_char_t<T>> m_vector;

    public:
      __host__ nonstandalone_buffer() : m_vector {} {}
      __host__ nonstandalone_buffer(size_t size) : m_vector(size) {}
      __host__ nonstandalone_buffer(nonstandalone_buffer&& o) : m_vector {std::move(o.m_vector)} {}
      __host__ void resize(size_t size) { m_vector.resize(size); }
      __host__ std::span<T> get()
      {
        if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
          return {Allen::forward_type_t<T, bool*>(m_vector.data()), m_vector.size()};
        }
        else {
          return m_vector;
        }
      }
      __host__ std::span<const T> get() const
      {
        if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
          return {Allen::forward_type_t<T, bool*>(m_vector.data()), m_vector.size()};
        }
        else {
          return m_vector;
        }
      }
    };
  } // namespace details

  /**
   * @brief A buffer that can be independently resized or operated upon.
   *        It can be moved, but it cannot be copied.
   * @details A buffer can be used as an independent datatype, and it requires
   *          supporting both the Allen and the Gaudi sequencer. This distinction
   *          can only be made in runtime, and therefore the following implementation
   *          needs to be able to manage the memory associated with the object in either case.
   */
  template<Store::Scope S, typename T>
  struct buffer {
  private:
    std::variant<details::standalone_buffer<S, T>, details::nonstandalone_buffer<T>> m_buffer;
    std::span<T> m_span;

  public:
    __host__ buffer(Allen::Store::memory_manager_t<S>& mem_manager) :
      m_buffer {details::standalone_buffer<S, T> {mem_manager}}
    {
      m_span = std::get<details::standalone_buffer<S, T>>(m_buffer).get();
    }
    __host__ buffer(Allen::Store::memory_manager_t<S>& mem_manager, size_t size) :
      m_buffer {details::standalone_buffer<S, T> {mem_manager, size}}
    {
      m_span = std::get<details::standalone_buffer<S, T>>(m_buffer).get();
    }
    __host__ buffer(size_t size) : m_buffer {details::nonstandalone_buffer<T> {size}}
    {
      m_span = std::get<details::nonstandalone_buffer<T>>(m_buffer).get();
    }
    buffer(buffer&&) = default;
    buffer& operator=(buffer&&) = default;

    __host__ void resize(size_t size)
    {
      std::visit(
        [&](auto& buf) {
          buf.resize(size);
          m_span = buf.get();
        },
        m_buffer);
    }
    constexpr __host__ std::span<T> get() { return m_span; }
    constexpr __host__ std::span<const T> get() const { return m_span; }
    constexpr __host__ auto begin() const
    {
      static_assert(S == Allen::Store::Scope::Host);
      return get().begin();
    }
    constexpr __host__ auto end() const
    {
      static_assert(S == Allen::Store::Scope::Host);
      return get().end();
    }
    constexpr __host__ auto size() const { return get().size(); }
    constexpr __host__ auto size_bytes() const { return get().size_bytes(); }
    constexpr __host__ auto data() const { return get().data(); }
    constexpr __host__ auto data() { return get().data(); }
    constexpr __host__ auto& operator[](int i)
    {
      static_assert(S == Allen::Store::Scope::Host);
      return get()[i];
    }
    constexpr __host__ const auto& operator[](int i) const
    {
      static_assert(S == Allen::Store::Scope::Host);
      return get()[i];
    }
    constexpr __host__ operator std::span<T>() { return get(); }
    constexpr __host__ auto operator->() const { return data(); }
    constexpr __host__ operator T*() const { return data(); }
    constexpr __host__ auto empty() const { return m_span.empty(); }
    constexpr __host__ auto subspan(const std::size_t offset) const { return m_span.subspan(offset); }
    constexpr __host__ auto subspan(const std::size_t offset, const std::size_t count) const
    {
      return m_span.subspan(offset, count);
    }

    buffer(const buffer&) = delete;
    buffer& operator=(const buffer&) = delete;
  };

  namespace details {
    struct shared_buffer_metadata;

    struct type_erased_dependency {
      void* m_data {nullptr};
      shared_buffer_metadata* m_meta {nullptr};
      Store::Scope m_scope {Store::Scope::Host};
      type_erased_dependency* m_next {nullptr};

      void release();
    };

    struct shared_buffer_metadata {
      size_t ref_count {0};
      Store::memory_managers_t memory_manager {};
      type_erased_dependency* dependencies {nullptr};

      void add_dependency(void* data, shared_buffer_metadata* meta, Store::Scope scope)
      {
        /*if (meta->is_depending_on(this)) { // TODO: enable in debug
          std::cerr << "Warning: Cycle detected, not adding dependency." << std::endl;
          return;
        }*/
        auto next = dependencies;
        meta->ref_count++;
        dependencies = memory_manager.dep_allocator->allocate();
        dependencies->m_data = data;
        dependencies->m_meta = meta;
        dependencies->m_scope = scope;
        dependencies->m_next = next;
      }

      bool is_depending_on(const shared_buffer_metadata* target) const
      {
        if (target == nullptr) return false;
        auto current = dependencies;
        while (current) {
          if (current->m_meta == target) {
            return true;
          }
          if (current->m_meta && current->m_meta->is_depending_on(target)) {
            return true;
          }
          current = current->m_next;
        }
        return false;
      }

      void release_dependencies()
      {
        auto dep = dependencies;
        while (dep) {
          auto next = dep->m_next;
          dep->release();
          // memory_manager.free<Store::Scope::Host>(dep);
          memory_manager.dep_allocator->deallocate(dep);
          dep = next;
        }
        dependencies = nullptr;
      }
    };

    inline void type_erased_dependency::release()
    {
      m_meta->ref_count--;
      if (m_meta->ref_count == 0) {
        if (m_data != nullptr) {
          if (m_scope == Store::Scope::Host) {
            m_meta->memory_manager.free<Store::Scope::Host>(m_data);
          }
          else {
            m_meta->memory_manager.free<Store::Scope::Device>(m_data);
          }
        }
        m_meta->release_dependencies();
        m_meta->memory_manager.meta_allocator->deallocate(m_meta);
      }
    }

    template<Allen::Store::Scope from, Allen::Store::Scope to>
    inline constexpr Allen::memcpy_kind get_memcpy_kind()
    {
      if constexpr (from == Allen::Store::Scope::Host && to == Allen::Store::Scope::Host)
        return Allen::memcpyHostToHost;
      else if constexpr (from == Allen::Store::Scope::Host && to == Allen::Store::Scope::Device)
        return Allen::memcpyHostToDevice;
      else if constexpr (from == Allen::Store::Scope::Device && to == Allen::Store::Scope::Host)
        return Allen::memcpyDeviceToHost;
      return Allen::memcpyDeviceToDevice;
    }
  } // namespace details

  template<Store::Scope S, typename T>
  struct shared_buffer {
    static constexpr Store::Scope scope = S;
    using value_type = T;

    shared_buffer(const Store::memory_managers_t& mem_manager)
    {
      m_meta = mem_manager.meta_allocator->allocate();
      m_meta->ref_count = 1;
      m_meta->memory_manager = mem_manager;
      m_meta->dependencies = nullptr;
    }

    shared_buffer(size_t size, const Store::memory_managers_t& mem_manager) : shared_buffer(mem_manager)
    {
      m_data = mem_manager.reserve<S, T>(size);
      m_size = size;
    }

    // copy semantics
    shared_buffer(const shared_buffer<S, T>& obj)
    {
      this->m_data = obj.m_data;
      this->m_size = obj.m_size;
      this->m_meta = obj.m_meta;
      m_meta->ref_count++;
    }

    shared_buffer& operator=(const shared_buffer<S, T>& obj)
    {
      cleanup();
      this->m_data = obj.m_data;
      this->m_size = obj.m_size;
      this->m_meta = obj.m_meta;
      m_meta->ref_count++;
      return *this;
    }

    // move semantics
    shared_buffer(shared_buffer<S, T>&& obj)
    {
      this->m_data = obj.m_data;
      this->m_size = obj.m_size;
      this->m_meta = obj.m_meta;
      obj.m_data = nullptr;
      obj.m_size = 0;
      obj.m_meta = nullptr;
    }

    shared_buffer& operator=(shared_buffer<S, T>&& obj)
    {
      cleanup();
      this->m_data = obj.m_data;
      this->m_size = obj.m_size;
      this->m_meta = obj.m_meta;
      obj.m_data = nullptr;
      obj.m_size = 0;
      obj.m_meta = nullptr;
      return *this;
    }

    // destructor
    ~shared_buffer() { cleanup(); }

    // size
    std::size_t size() const { return m_size; }
    std::size_t size_bytes() const { return m_size * sizeof(T); }
    bool empty() const { return size() == 0; }
    void resize(std::size_t size)
    {
      if (m_size >= size) {
        m_size = size;
        return;
      }
      auto rc = ref_count();
      if (rc != 1)
        throw std::runtime_error {
          "Cannot resize a shared buffer with more than 1 owner (ref_count=" + std::to_string(rc) + ")"};
      if (m_data != nullptr) m_meta->memory_manager.free<S>(m_data);
      m_data = m_meta->memory_manager.reserve<S, T>(size);
      m_size = size;
    }

    // accessors
    T* data() { return m_data; }
    const T* data() const { return m_data; }
    T& operator[](std::size_t index)
    {
      static_assert(S == Allen::Store::Scope::Host);
      return m_data[index];
    }
    const T& operator[](std::size_t index) const
    {
      static_assert(S == Allen::Store::Scope::Host);
      return m_data[index];
    }

    std::span<T> get() { return {data(), size()}; }
    std::span<const T> get() const { return {data(), size()}; }
    operator std::span<T>() { return get(); }
    auto operator->() const { return data(); }
    operator T*() const { return data(); }
    auto begin() const
    {
      static_assert(S == Allen::Store::Scope::Host);
      return get().begin();
    }
    auto end() const
    {
      static_assert(S == Allen::Store::Scope::Host);
      return get().end();
    }
    auto subspan(const std::size_t offset) const { return get().subspan(offset); }
    auto subspan(const std::size_t offset, const std::size_t count) const { return get().subspan(offset, count); }

    std::size_t ref_count() const { return m_meta->ref_count; }

    template<Store::Scope NewScope>
    shared_buffer<NewScope, T> to() const
    {
      shared_buffer<NewScope, T> new_buffer {m_meta->memory_manager};
      this->copy_to(new_buffer);
      return new_buffer;
    }

    auto to_host() const { return to<Store::Scope::Host>(); }
    auto to_device() const { return to<Store::Scope::Device>(); }

    template<Store::Scope S2>
    void copy_to(shared_buffer<S2, T>& dest) const
    {
      dest.resize(m_size);
      if (m_size == 0) return;
      Allen::memcpy_async(
        dest.data(),
        this->data(),
        this->size_bytes(),
        details::get_memcpy_kind<S, S2>(),
        m_meta->memory_manager.context);
      if constexpr (S == Allen::Store::Scope::Device && S2 == Allen::Store::Scope::Host) {
        // If copying from device to host, we need to synchronize to ensure the data is available before returning
        Allen::synchronize(m_meta->memory_manager.context);
      }
    }

    template<Store::Scope S2, typename T2>
    void depends_on(const shared_buffer<S2, T2>& other)
    {
      m_meta->add_dependency(reinterpret_cast<void*>(other.m_data), other.m_meta, S2);
    }

  private:
    void cleanup()
    {
      if (m_meta == nullptr) return;
      m_meta->ref_count--;
      if (m_meta->ref_count == 0) {
        if (m_data != nullptr) {
          m_meta->memory_manager.free<S>(m_data);
          m_data = nullptr;
          m_size = 0;
        }
        m_meta->release_dependencies();
        m_meta->memory_manager.meta_allocator->deallocate(m_meta);
        m_meta = nullptr;
      }
    }

    template<Store::Scope S2, typename T2>
    friend struct shared_buffer;

    T* m_data {nullptr};
    std::size_t m_size {0};
    details::shared_buffer_metadata* m_meta {nullptr};
  };

  template<typename T>
  using host_buffer = shared_buffer<Store::Scope::Host, T>;
  template<typename T>
  using device_buffer = shared_buffer<Store::Scope::Device, T>;
} // namespace Allen
