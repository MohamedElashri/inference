/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "RichInitPID.cuh"

INSTANTIATE_ALGORITHM(rich_init_pid::rich_init_pid_t);

void rich_init_pid::rich_init_pid_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  set_size<dev_pid_t>(arguments, number_of_tracks);
}

void rich_init_pid::rich_init_pid_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  const auto default_pid = static_cast<Allen::Rich::ParticleIDType>(m_default_pid.value());
  Allen::memset_async<dev_pid_t>(arguments, default_pid, context);
}
