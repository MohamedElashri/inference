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
#include "KsLLDetachedTrackLine.cuh"

INSTANTIATE_LINE(ks_ll_detached_track_line::ks_ll_detached_track_line_t, ks_ll_detached_track_line::Parameters)

__device__ bool ks_ll_detached_track_line::ks_ll_detached_track_line_t::select(
  const Parameters&,
  const DeviceProperties& properties,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input)
{
  const auto Ksh = std::get<0>(input);
  const auto vertex = Ksh.vertex();
  const auto Ks = Allen::dyn_cast<const Allen::Views::Physics::CompositeParticle*>(Ksh.child(0));
  const auto Ks_vx = Ks->vertex();
  const auto companion = Allen::dyn_cast<const Allen::Views::Physics::BasicParticle*>(Ksh.child(1));
  const auto companion_state = companion->state();
  const auto c0 = static_cast<const Allen::Views::Physics::BasicParticle*>(Ks->child(0)),
             c1 = static_cast<const Allen::Views::Physics::BasicParticle*>(Ks->child(1));
  const auto c0_state = c0->state(), c1_state = c1->state();
  const auto mK0 = Ks->m12(Allen::mPi, Allen::mPi);
  const auto mKSpi = Ksh.m12(Allen::mK0, Allen::mPi);
  return c0_state.pt() > properties.pi_PT_min && c1_state.pt() > properties.pi_PT_min &&
         c0->ip_chi2() > properties.pi_MIPCHI2_min && c1->ip_chi2() > properties.pi_MIPCHI2_min &&
         Ks->doca12() < properties.Ks_DOCA_max && Ks_vx.chi2() < properties.Ks_VCHI2_max &&
         Ks->fdchi2() > properties.Ks_BPVFDCHI2_min && properties.Ks_M_min < mK0 && mK0 < properties.Ks_M_max &&
         Ks_vx.pt() > properties.Ks_PT_min && companion_state.pt() > properties.h_PT_min &&
         companion->ip_chi2() > properties.h_MIPCHI2_min && mKSpi > properties.M_min && mKSpi < properties.M_max &&
         Ksh.dira() > properties.DIRA_min && Ks_vx.z() - vertex.z() > properties.KsVDZ_min &&
         properties.VZ_min < vertex.z() && vertex.z() < properties.VZ_max && Ksh.dz() > properties.BPVVDZ_min &&
         Ksh.drho() > properties.BPVVDRHO_min && Ks_vx.pt() + companion_state.pt() > properties.SUMPT_min;
}

__device__ bool ks_ll_detached_track_line::ks_ll_detached_track_line_t::fill_tuples(
  const Parameters& parameters,
  const DeviceProperties&,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input,
  unsigned index,
  bool sel)
{
  if (sel) {

    const auto Ksh = std::get<0>(input);
    const auto vertex = Ksh.vertex();
    const auto Ks = Allen::dyn_cast<const Allen::Views::Physics::CompositeParticle*>(Ksh.child(0));
    const auto companion = Allen::dyn_cast<const Allen::Views::Physics::BasicParticle*>(Ksh.child(1));
    const auto companion_state = companion->state();
    parameters.M[index] = Ksh.m12(Allen::mK0, Allen::mPi);
    parameters.MCORR[index] = Ksh.mcor();
    parameters.PT[index] = vertex.pt();
    parameters.DOCA[index] = Ksh.doca12();
    parameters.VZ[index] = vertex.z();
    parameters.BPVVDZ[index] = Ksh.dz();
    parameters.BPVVDRHO[index] = Ksh.drho();
    parameters.BPVDIRA[index] = Ksh.dira();
    parameters.BPVIP[index] = Ksh.ip();
    parameters.BPVFD[index] = Ksh.fd();
    parameters.h_P[index] = companion_state.p();
    parameters.h_PT[index] = companion_state.pt();
    parameters.h_MIPCHI2[index] = companion->ip_chi2();
    parameters.h_MIP[index] = companion->ip();
    parameters.h_CHI2NDF[index] = companion->chi2() / companion->ndof();
    parameters.h_Q[index] = companion_state.charge();

    const auto Ks_vx = Ks->vertex();
    const auto c0 = static_cast<const Allen::Views::Physics::BasicParticle*>(Ks->child(0)),
               c1 = static_cast<const Allen::Views::Physics::BasicParticle*>(Ks->child(1));
    const auto c0_state = c0->state(), c1_state = c1->state();

    parameters.Ks_M[index] = Ks->m12(Allen::mPi, Allen::mPi);
    parameters.pi1_P[index] = c1_state.p();
    parameters.pi1_PT[index] = c1_state.pt();
    parameters.pi1_MIPCHI2[index] = c1->ip_chi2();
    parameters.pi1_MIP[index] = c1->ip();
    parameters.pi1_CHI2NDF[index] = c1->chi2() / c1->ndof();
    parameters.pi1_Q[index] = c1_state.charge();
    parameters.pi2_P[index] = c0_state.p();
    parameters.pi2_PT[index] = c0_state.pt();
    parameters.pi2_MIPCHI2[index] = c0->ip_chi2();
    parameters.pi2_MIP[index] = c0->ip();
    parameters.pi2_CHI2NDF[index] = c0->chi2() / c0->ndof();
    parameters.pi2_Q[index] = c0_state.charge();
    parameters.Ks_VCHI2[index] = Ks_vx.chi2();
    parameters.Ks_VZ[index] = Ks_vx.z();
    parameters.pi1_pi2_DOCA[index] = Ks->doca12();
    parameters.Ks_PT[index] = Ks_vx.pt();
    parameters.Ks_BPVVDCHI2[index] = Ks->fdchi2();
    parameters.Ks_BPVVDZ[index] = Ks->dz();
    parameters.Ks_BPVVDRHO[index] = Ks->drho();
    parameters.Ks_BPVDIRA[index] = Ks->dira();
  }
  return sel;
}

__device__ void ks_ll_detached_track_line::ks_ll_detached_track_line_t::monitor(
  const Parameters&,
  const DeviceProperties& properties,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input,
  unsigned,
  bool sel)
{
  if (sel) {
    const auto Ksh = std::get<0>(input);
    const auto Ks = Allen::dyn_cast<const Allen::Views::Physics::CompositeParticle*>(Ksh.child(0));
    const auto mK0pi = Ksh.m12(Allen::mK0, Allen::mPi);
    const auto mK0 = Ks->m12(Allen::mPi, Allen::mPi);

    properties.histogram_Ks_mass.increment(mK0);
    properties.histogram_KsPi_mass.increment(mK0pi);
  }
}
