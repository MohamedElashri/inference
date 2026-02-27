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
#include "QuirksLine.cuh"
#include <string>
#include <sstream>

// Explicit instantiation of the line
INSTANTIATE_LINE(quirks_line::quirks_line_t, quirks_line::Parameters)

// Get input function
__device__ unsigned
quirks_line::quirks_line_t::get_input(const Parameters& parameters, const unsigned event_number, const unsigned /*i*/)
{
  const unsigned input = parameters.dev_quirks_pairs[event_number];
  return input;
}

// Selection function
__device__ bool quirks_line::quirks_line_t::select(const Parameters&, unsigned input) { return input > 0; }

__device__ bool
quirks_line::quirks_line_t::fill_tuples(const Parameters&, unsigned /*input*/, unsigned /*index*/, bool sel)
{
  return sel;
}
