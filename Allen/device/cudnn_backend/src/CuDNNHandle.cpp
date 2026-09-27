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
#include "CuDNNHandle.h"

#include <mutex>
#include <unordered_map>

cudnnHandle_t Allen::CuDNN::handle(cudaStream_t stream)
{
  thread_local cudaStream_t cached_stream = nullptr;
  thread_local cudnnHandle_t cached_handle = nullptr;
  if (cached_handle == nullptr || cached_stream != stream) {
    static std::mutex mutex;
    static std::unordered_map<cudaStream_t, cudnnHandle_t> handles;
    std::lock_guard<std::mutex> lock {mutex};
    auto [it, inserted] = handles.try_emplace(stream, nullptr);
    if (inserted) {
      ALLEN_CUDNN_CHECK(cudnnCreate(&it->second));
      ALLEN_CUDNN_CHECK(cudnnSetStream(it->second, stream));
    }
    cached_handle = it->second;
    cached_stream = stream;
  }
  return cached_handle;
}
