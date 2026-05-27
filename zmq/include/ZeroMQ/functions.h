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
#ifndef ZEROMQ_FUNCTIONS_H
#define ZEROMQ_FUNCTIONS_H 1

#include <zmq.hpp>

namespace zmq {

  enum PollType : short { POLLIN = ZMQ_POLLIN, POLLOUT = ZMQ_POLLOUT };

} // namespace zmq

namespace ZMQ {

  template<class T>
  size_t defaultSizeOf(const T&)
  {
    return sizeof(T);
  }

  inline size_t stringLength(const char& cs) { return strlen(&cs); }

} // namespace ZMQ

#endif // ZEROMQ_FUNCTIONS_H
