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
#include "Logger.h"

#include <iostream>
#include <mutex>
#include <string>

#ifndef ALLEN_STANDALONE
#include <GaudiKernel/IMessageSvc.h>
#endif

namespace logger {
  namespace {
    // Function-local statics use C++11 "magic statics" initialization, which is
    // thread-safe; this replaces a prior namespace-scope-pointer, unsynchronized
    // check-then-act lazy-init pattern that raced under concurrent first calls.
    Logger& instance()
    {
      static Logger ll;
      return ll;
    }

    // Per-thread formatting state.  Each log message is buffered in its own
    // stream (so that concurrent messages cannot mix), but stream formatting
    // such as std::hex, std::setfill or std::setprecision set in one message
    // must still apply to the following ones, exactly as it did when every
    // message wrote directly to the shared std::cout.
    std::ios& formatting_state()
    {
      static thread_local std::ios state {nullptr};
      return state;
    }

    // Serialise access to std::cout so that concurrent log messages cannot be
    // interleaved with each other.
    std::mutex& output_mutex()
    {
      static std::mutex mutex;
      return mutex;
    }

#ifndef ALLEN_STANDALONE
    // Gaudi message service and source used to forward messages, set through
    // setMessageSvc().  The mutex protects the source string; the pointer is
    // atomic so the common no-service case stays lock-free.
    std::atomic<IMessageSvc*> message_svc {nullptr};
    std::mutex config_mutex;
    std::string message_source {"Allen"};

    // Gaudi levels grow with severity (VERBOSE=1 .. FATAL=6), Allen levels grow
    // with verbosity (error=1 .. verbose=5), hence the inversion.
    int to_allen_verbosity(int gaudi_level)
    {
      if (gaudi_level <= 0) return verbose; // NIL/unset -> report everything
      if (gaudi_level > 5) return 0;        // FATAL/ALWAYS -> Allen has no fatal
      return 6 - gaudi_level;
    }

    int to_gaudi_level(int allen_level)
    {
      switch (allen_level) {
      case verbose: return MSG::VERBOSE;
      case debug: return MSG::DEBUG;
      case info: return MSG::INFO;
      case warning: return MSG::WARNING;
      case error: return MSG::ERROR;
      default: return MSG::INFO;
      }
    }
#endif

    void output_message(int level, const std::string& message, bool flush)
    {
      (void) level; // only used to select the Gaudi severity
#ifndef ALLEN_STANDALONE
      if (auto* svc = message_svc.load(std::memory_order_acquire)) {
        // Gaudi's message service formats, filters and serialises the output.
        // It appends a newline itself, so drop a single trailing one to avoid
        // introducing empty lines.
        std::string text = message;
        if (!text.empty() && text.back() == '\n') {
          text.pop_back();
        }
        std::string source;
        {
          const std::lock_guard<std::mutex> lock(config_mutex);
          source = message_source;
        }
        svc->reportMessage(std::move(source), to_gaudi_level(level), std::move(text));
        return;
      }
#endif
      if (message.empty() && !flush) {
        return;
      }
      const std::lock_guard<std::mutex> lock(output_mutex());
      std::cout << message;
      if (flush) {
        std::cout.flush();
      }
    }
  } // namespace
} // namespace logger

void logger::setVerbosity(int level) { instance().verbosityLevel = level; }

int logger::verbosity() { return instance().verbosityLevel; }

void logger::setMessageSvc(IMessageSvc* svc, std::string source)
{
#ifndef ALLEN_STANDALONE
  const std::lock_guard<std::mutex> lock(config_mutex);
  message_source = source;
  message_svc.store(svc, std::memory_order_release);
  if (svc) {
    setVerbosity(to_allen_verbosity(svc->outputLevel(message_source)));
  }
#else
  (void) svc; // standalone build: keep using the std::cout fallback
  (void) source;
#endif
}

void logger::clearMessageSvc()
{
#ifndef ALLEN_STANDALONE
  message_svc.store(nullptr, std::memory_order_release);
#endif
}

logger::LogMessage::LogMessage(int level, bool enabled, bool flush) : m_level(level), m_enabled(enabled), m_flush(flush)
{
  if (m_enabled) {
    m_stream.copyfmt(formatting_state());
  }
}

logger::LogMessage logger::logger(int requestedLogLevel)
{
  return LogMessage(requestedLogLevel, instance().verbosityLevel >= requestedLogLevel);
}

logger::LogMessage& logger::LogMessage::operator<<(std::ostream& (*manipulator)(std::ostream&) )
{
  if (m_enabled) {
    m_flush = m_flush || manipulator == static_cast<std::ostream& (*) (std::ostream&)>(std::endl) ||
              manipulator == static_cast<std::ostream& (*) (std::ostream&)>(std::flush);
    manipulator(m_stream);
  }
  return *this;
}

logger::LogMessage::~LogMessage()
{
  if (!m_enabled) {
    return;
  }

  try {
    // Remember the formatting state so that it carries over to the next message
    // on this thread.
    formatting_state().copyfmt(m_stream);
    output_message(m_level, m_stream.str(), m_flush);
  } catch (...) {
    // Never let a logging failure escape a destructor.
  }
}
