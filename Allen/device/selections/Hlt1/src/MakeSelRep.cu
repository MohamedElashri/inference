/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "MakeSelRep.cuh"
#include "HltSelReport.cuh"

INSTANTIATE_ALGORITHM(make_selrep::make_selrep_t)

void make_selrep::make_selrep_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<host_selrep_offsets_t>(arguments, size<dev_selrep_offsets_t>(arguments));
  set_size<host_sel_reports_t>(arguments, first<host_selrep_size_t>(arguments));
  set_size<dev_sel_reports_t>(arguments, first<host_selrep_size_t>(arguments));
}

void make_selrep::make_selrep_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  // Initialization might not be necessary.
  Allen::memset_async<dev_sel_reports_t>(arguments, 0, context);
  global_function(make_selrep::make_selrep)(dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(
    arguments, first<host_number_of_events_t>(arguments));

  Allen::copy_async<host_selrep_offsets_t, dev_selrep_offsets_t>(arguments, context);
  Allen::copy_async<host_sel_reports_t, dev_sel_reports_t>(arguments, context);
  Allen::synchronize(context);
}

__global__ void make_selrep::make_selrep(make_selrep::Parameters parameters, const unsigned number_of_events)
{
  for (unsigned event_number = blockIdx.x * blockDim.x + threadIdx.x; event_number < number_of_events;
       event_number += blockDim.x * gridDim.x) {
    const unsigned bank_offset = parameters.dev_selrep_offsets[event_number];
    const unsigned objtyp_offset = parameters.dev_rb_objtyp_offsets[event_number];
    const unsigned hits_offset = parameters.dev_rb_hits_offsets[event_number];
    const unsigned substr_offset = parameters.dev_rb_substr_offsets[event_number];
    const unsigned stdinfo_offset = parameters.dev_rb_stdinfo_offsets[event_number];
    const unsigned bank_size = parameters.dev_selrep_offsets[event_number + 1] - bank_offset;
    const unsigned objtyp_size = parameters.dev_rb_objtyp_offsets[event_number + 1] - objtyp_offset;
    const unsigned hits_size = parameters.dev_rb_hits_offsets[event_number + 1] - hits_offset;
    const unsigned substr_size = parameters.dev_rb_substr_offsets[event_number + 1] - substr_offset;
    const unsigned stdinfo_size = parameters.dev_rb_stdinfo_offsets[event_number + 1] - stdinfo_offset;
    const unsigned* event_rb_objtyp = parameters.dev_rb_objtyp + objtyp_offset;
    const unsigned* event_rb_hits = parameters.dev_rb_hits + hits_offset;
    const unsigned* event_rb_substr = parameters.dev_rb_substr + substr_offset;
    const unsigned* event_rb_stdinfo = parameters.dev_rb_stdinfo + stdinfo_offset;

    // Make the bank header.
    unsigned* event_selrep = parameters.dev_sel_reports + bank_offset;

    make_selrep_bank(
      event_selrep,
      event_rb_objtyp,
      event_rb_hits,
      event_rb_substr,
      event_rb_stdinfo,
      bank_size,
      objtyp_size,
      hits_size,
      substr_size,
      stdinfo_size);
  }
}
