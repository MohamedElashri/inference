/*****************************************************************************\
* (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <boost/filesystem.hpp>
#include <boost/interprocess/streams/vectorstream.hpp>

#include <Kernel/STLExtensions.h>
#include <fstream>
#include <functional>

#include <string>
#include <type_traits>
#include <vector>

#include <Common.h>

template<class T>
void optional_resize(T&, size_t)
{}

template<class... Args>
void optional_resize(std::vector<Args...>& v, size_t s)
{
  v.resize(s);
}

namespace DumpUtils {

  bool createDirectory(boost::filesystem::path dir);

  namespace detail {
    inline std::ostream& write(std::ostream& os, std::span<const std::byte> bytes)
    {
      return os.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    template<class T, std::size_t N>
    std::ostream& write(std::ostream& os, std::span<T, N> s)
    {
      static_assert(std::is_trivially_copyable_v<T>);
      return write(os, as_bytes(s));
    }
    template<typename T>
    std::ostream& write(std::ostream& os, T const& t)
    {
      if constexpr (requires { LHCb::make_span(t); }) {
        return write(os, LHCb::make_span(t));
      }
      else {
        // if you would like to know why there is a check for trivially copyable,
        // please read the 'notes' section of https://en.cppreference.com/w/cpp/types/is_trivially_copyable
        static_assert(std::is_trivially_copyable_v<T>);
        return os.write(reinterpret_cast<char const*>(std::addressof(t)), static_cast<std::streamsize>(sizeof(T)));
      }
    }
  } // namespace detail

  class Writer {

    boost::interprocess::basic_vectorstream<std::vector<char>> m_buffer;

  public:
    template<typename... Args>
    Writer& write(Args&&... args)
    {
      (detail::write(m_buffer, std::forward<Args>(args)), ...);
      return *this;
    }

    std::vector<char> const& buffer() { return m_buffer.vector(); }
  };

  class FileWriter {
    std::ofstream m_f;

  public:
    FileWriter(const std::string& name) : m_f {name, std::ios::out | std::ios::binary} {}

    template<typename... Args>
    FileWriter& write(Args&&... args)
    {
      (detail::write(m_f, std::forward<Args>(args)), ...);
      return *this;
    }
  };

  using Dump = std::tuple<std::vector<char>, std::string, std::string>;
  using Dumps = std::vector<Dump>;

} // namespace DumpUtils
