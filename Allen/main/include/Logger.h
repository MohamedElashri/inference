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

#define verbose_cout logger::logger(logger::verbose)
#define debug_cout logger::logger(logger::debug)
#define info_cout logger::logger(logger::info)
#define warning_cout logger::logger(logger::warning)
#define error_cout logger::logger(logger::error)

#include <atomic>
#include <iosfwd>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <memory>
#include <string>
#include <utility>
#include "LoggerCommon.h"

// Forward declaration only: the logger stays independent of Gaudi.  In a Gaudi
// build the message service is registered at runtime through
// logger::setMessageSvc().
class IMessageSvc;

namespace logger {
  class Logger {
  public:
    std::atomic<int> verbosityLevel {3};
  };

  /// @brief Proxy object accumulating a single log message.
  ///
  /// This logger is used both when Allen runs standalone and when it is
  /// embedded in a Gaudi application, so it cannot rely on Gaudi's message
  /// service directly.  Instead of returning a reference to a shared ostream
  /// (where every `operator<<` is an independent write and concurrent log
  /// messages can interleave character by character), `logger()` returns this
  /// proxy.  It buffers the whole message and emits it when the full
  /// expression has been evaluated, i.e. on destruction:
  ///   - in a Gaudi build, through the registered message service,
  ///   - otherwise, as a single mutex-protected write to std::cout.
  class LogMessage {
  public:
    LogMessage(int level, bool enabled, bool flush = false);

    LogMessage(const LogMessage&) = delete;
    LogMessage& operator=(const LogMessage&) = delete;

    LogMessage(LogMessage&& other) :
      m_level(other.m_level), m_enabled(other.m_enabled), m_flush(other.m_flush), m_stream(std::move(other.m_stream))
    {
      other.m_enabled = false;
      other.m_flush = false;
    }

    ~LogMessage();

    template<typename T>
    LogMessage& operator<<(const T& value)
    {
      if (m_enabled) {
        m_stream << value;
      }
      return *this;
    }

    /// @brief Overload for stream manipulators (`std::hex`, `std::endl`, ...).
    ///
    /// The flushing manipulators are detected so that the underlying output
    /// stream is flushed once the buffered message has been written.
    LogMessage& operator<<(std::ostream& (*manipulator)(std::ostream&) );

  private:
    int m_level;
    bool m_enabled;
    bool m_flush;
    std::ostringstream m_stream;
  };

  LogMessage logger(int requestedLogLevel);

  int verbosity();

  void setVerbosity(int level);

  /// @brief Route log messages through a Gaudi message service.
  ///
  /// Only effective in Gaudi (non-standalone) builds; in standalone builds this
  /// is a no-op.  `source` is the message source reported to Gaudi (typically
  /// the component name) and is also used to derive Allen's verbosity from the
  /// service's output level for that source.  Passing a null service, or
  /// calling clearMessageSvc(), restores the std::cout fallback.
  ///
  /// Must be called before any thread that logs through this logger is started.
  void setMessageSvc(IMessageSvc* svc, std::string source);
  void clearMessageSvc();
} // namespace logger
