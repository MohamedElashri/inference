/*****************************************************************************\
* (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This sofwqare is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "TTrackKs2PiPi.cuh"
#include <ROOTHeaders.h>
#include "ROOTService.h"

INSTANTIATE_LINE(ttrack_ks2pipi::ttrack_ks2pipi_t, ttrack_ks2pipi::Parameters)

__device__ bool ttrack_ks2pipi::ttrack_ks2pipi_t::select(
  const Parameters&,
  const DeviceProperties& props,
  std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned> input)
{
  const auto composite = std::get<0>(input);
  const auto vertex = composite.vertex();
  const auto childA = static_cast<const Allen::Views::Physics::BasicParticle*>(composite.child(0));
  const auto childB = static_cast<const Allen::Views::Physics::BasicParticle*>(composite.child(1));

  return (composite.charge() == 0) && (vertex.ttracks_doca() < props.max_doca) && (vertex.pt() > props.min_pt) &&
         (vertex.p() > props.min_p) && (vertex.x() * vertex.x() + vertex.y() * vertex.y() > props.min_r2) &&
         (composite.minp() > props.min_track_p) && (vertex.ttracks_ip2() < props.max_ip2) &&
         (composite.mdipi() < props.max_m) && (!childA->is_lepton()) && (!childB->is_lepton()) &&
         (vertex.z() < props.max_ovtx_z) && (vertex.ttracks_dira() > props.min_dira);
}

__device__ void ttrack_ks2pipi::ttrack_ks2pipi_t::monitor(
  const Parameters&,
  const DeviceProperties& properties,
  std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned> input,
  unsigned,
  bool sel)
{
  if (sel) {
    const auto part = std::get<0>(input);
    properties.histogram_mass.increment(part.mdipi());
    properties.histogram_p.increment(part.vertex().p());
    properties.histogram_pt.increment(part.vertex().pt());

    const float eta = atanhf(part.vertex().pz() / part.vertex().p());
    properties.histogram_eta.increment(eta);
  }
}

__device__ bool ttrack_ks2pipi::ttrack_ks2pipi_t::fill_tuples(
  const Parameters& parameters,
  const DeviceProperties&,
  std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned> input,
  unsigned index,
  bool sel)
{
  if (sel) {
    const auto part = std::get<0>(input);

    parameters.mon_mass[index] = part.mdipi();
    parameters.mon_p[index] = part.vertex().p();
    parameters.mon_pt[index] = part.vertex().pt();
    const float eta = atanhf(part.vertex().pz() / part.vertex().p());
    parameters.mon_eta[index] = eta;
    parameters.mon_vrtx_x[index] = part.vertex().x();
    parameters.mon_vrtx_y[index] = part.vertex().y();
    parameters.mon_vrtx_z[index] = part.vertex().z();
  }
  return sel;
}