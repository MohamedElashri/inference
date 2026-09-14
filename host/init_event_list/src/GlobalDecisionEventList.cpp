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
#include "GlobalDecisionEventList.h"

INSTANTIATE_ALGORITHM(global_decision_event_list::global_decision_event_list_t)

void global_decision_event_list::global_decision_event_list_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions& runtime_options,
  const Constants&) const
{
  const auto number_of_events =
    std::get<1>(runtime_options.event_interval) - std::get<0>(runtime_options.event_interval);

  // Initialize number of events
  set_size<host_event_list_output_t>(arguments, number_of_events);
  set_size<dev_event_list_output_t>(arguments, number_of_events);
}

void global_decision_event_list::global_decision_event_list_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  const auto number_of_events =
    std::get<1>(runtime_options.event_interval) - std::get<0>(runtime_options.event_interval);

  // Initialize buffers
  unsigned out = 0;
  for (unsigned i = 0; i < number_of_events; ++i) {
    if (data<host_global_decision_t>(arguments)[i]) {
      data<host_event_list_output_t>(arguments)[out++] = i;
    }
  }
  reduce_size<host_event_list_output_t>(arguments, out);
  reduce_size<dev_event_list_output_t>(arguments, out);
  Allen::copy_async<dev_event_list_output_t, host_event_list_output_t>(arguments, context);
}
