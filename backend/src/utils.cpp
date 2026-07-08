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

#include "BackendCommonInterface.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef __linux
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#endif

inline int backTrace([[maybe_unused]] void** addresses, [[maybe_unused]] const int depth)
{

#ifdef __linux

  int count = backtrace(addresses, depth);
  return count > 0 ? count : 0;

#else // windows and osx parts not implemented
  return 0;
#endif
}

inline bool getStackLevel(
  [[maybe_unused]] void* addresses,
  [[maybe_unused]] void*& addr,
  [[maybe_unused]] std::string& fnc,
  [[maybe_unused]] std::string& lib)
{

#ifdef __linux

  Dl_info info;

  if (dladdr(addresses, &info) && info.dli_fname && info.dli_fname[0] != '\0') {
    const char* symbol = info.dli_sname && info.dli_sname[0] != '\0' ? info.dli_sname : nullptr;

    lib = info.dli_fname;
    addr = info.dli_saddr;

    if (symbol) {
      int stat = -1;
      auto dmg =
        std::unique_ptr<char, decltype(free)*>(abi::__cxa_demangle(symbol, nullptr, nullptr, &stat), std::free);
      fnc = (stat == 0) ? dmg.get() : symbol;
    }
    else {
      fnc = "local";
    }
    return true;
  }
  else {
    return false;
  }

#else // not implemented for windows and osx
  return false;
#endif
}

bool Allen::utils::backTrace(std::string& btrace, const int depth, const int offset)
{
  try {
    // Always hide the first two levels of the stack trace (that's us)
    const int totalOffset = offset + 2;
    const int totalDepth = depth + totalOffset;

    std::string fnc, lib;

    std::vector<void*> addresses(totalDepth, nullptr);
    int count = ::backTrace(addresses.data(), totalDepth);
    for (int i = totalOffset; i < count; ++i) {
      void* addr = nullptr;

      if (getStackLevel(addresses[i], addr, fnc, lib)) {
        std::ostringstream ost;
        ost << "#" << std::setw(3) << std::setiosflags(std::ios::left) << i - totalOffset + 1;
        ost << std::hex << addr << std::dec << " " << fnc << "  [" << lib << "]" << std::endl;
        btrace += ost.str();
      }
    }
    return true;
  } catch (const std::bad_alloc& e) {
    return false;
  }
}
