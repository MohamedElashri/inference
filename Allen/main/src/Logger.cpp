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

#if defined(__clang__) && __clang_major__ >= 10
#pragma clang diagnostic push
#if !defined(__APPLE__)
#pragma clang diagnostic ignored "-Wdeprecated-copy"
#endif
#endif

#include "boost/iostreams/stream.hpp"

#if defined(__clang__) && __clang_major__ >= 10
#pragma clang diagnostic pop
#endif

#include "boost/iostreams/device/null.hpp"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

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

    boost::iostreams::stream<boost::iostreams::null_sink>& null_ostream()
    {
      static boost::iostreams::stream<boost::iostreams::null_sink> os {boost::iostreams::null_sink()};
      return os;
    }
  } // namespace
} // namespace logger

void logger::setVerbosity(int level) { instance().verbosityLevel = level; }

int logger::verbosity() { return instance().verbosityLevel; }

std::ostream& logger::logger(int requestedLogLevel)
{
  if (instance().verbosityLevel >= requestedLogLevel) {
    return std::cout;
  }
  else {
    return null_ostream();
  }
}
