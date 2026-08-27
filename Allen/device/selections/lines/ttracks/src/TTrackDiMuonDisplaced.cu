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
#include "TTrackDiMuonDisplaced.cuh"
#include <ROOTHeaders.h>
#include "ROOTService.h"

INSTANTIATE_LINE(ttrack_dimuon_displaced::ttrack_dimuon_displaced_t, ttrack_dimuon_displaced::Parameters)

__device__ bool ttrack_dimuon_displaced::ttrack_dimuon_displaced_t::select(
  const Parameters&,
  const DeviceProperties& props,
  std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned> input)
{
  const auto composite = std::get<0>(input);
  const auto vertex = composite.vertex();
  const auto childA = static_cast<const Allen::Views::Physics::BasicParticle*>(composite.child(0));
  const auto childB = static_cast<const Allen::Views::Physics::BasicParticle*>(composite.child(1));

  const auto eta = atanhf(vertex.pz() / vertex.p());
  const auto opposite_sign = composite.charge() == 0;

  return (opposite_sign == props.opposite_sign) && (vertex.ttracks_ip2() < props.max_ip2) &&
         (vertex.ttracks_dira() > props.min_dira) && (eta < props.max_eta) &&
         (vertex.x() * vertex.x() + vertex.y() * vertex.y() > props.min_r2) && (childA->is_muon()) &&
         (childB->is_muon()) && (composite.minp() > props.min_track_p) && (composite.maxp() > props.min_max_track_p) &&
         (vertex.p() > props.min_p) && (vertex.pt() > props.min_pt) && (composite.mdimu() > props.min_m) &&
         (composite.mdimu() <= props.max_m) && (vertex.ttracks_doca() < props.max_doca) &&
         (vertex.z() < props.max_ovtx_z);
}

__device__ void ttrack_dimuon_displaced::ttrack_dimuon_displaced_t::monitor(
  const Parameters&,
  const DeviceProperties& properties,
  std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned> input,
  unsigned,
  bool sel)
{
  if (sel) {
    const auto part = std::get<0>(input);
    properties.histogram_mass.increment(part.mdimu());
    properties.histogram_p.increment(part.vertex().p());
    properties.histogram_pt.increment(part.vertex().pt());

    const float eta = atanhf(part.vertex().pz() / part.vertex().p());
    properties.histogram_eta.increment(eta);
  }
}

__device__ bool ttrack_dimuon_displaced::ttrack_dimuon_displaced_t::fill_tuples(
  const Parameters& parameters,
  const DeviceProperties&,
  std::tuple<const Allen::Views::Physics::CompositeParticle, const unsigned> input,
  unsigned index,
  bool sel)
{
  if (sel) {
    const auto part = std::get<0>(input);
    parameters.mon_mass[index] = part.mdimu();
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
