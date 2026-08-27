/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "SumReduction.cuh"

namespace SumReduction {
  template __global__ void
  parallel_sum_reduction<float>(const float* dev_input_buffer, const unsigned* dev_offsets, float* dev_output_buffer);

  template __global__ void parallel_sum_reduction<unsigned>(
    const unsigned* dev_input_buffer,
    const unsigned* dev_offsets,
    unsigned* dev_output_buffer);
} // namespace SumReduction
