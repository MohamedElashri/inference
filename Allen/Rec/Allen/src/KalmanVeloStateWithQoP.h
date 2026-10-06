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
#pragma once

#include "States.cuh"
#include <algorithm>
#include <vector>

#include "Event/Track_v3.h"
#include "Event/StateParameters.h"

/**
 * Augmented KalmanVeloState that carries qop and its variance.
 * Shared between Velo→v3 and Velo→PrVelo converters.
 */
struct KalmanVeloStateWithQoP : public KalmanVeloState {
  float m_qop = 0.f, m_c44 = 0.f;

  KalmanVeloStateWithQoP() = default;
  KalmanVeloStateWithQoP(const KalmanVeloState& s, float qop, float c44) : KalmanVeloState(s), m_qop(qop), m_c44(c44) {}

  float qop() const { return m_qop; }
  float c44() const { return m_c44; }
};

/** Linear state extrapolation (duplicated from velo_kalman_filter). */
inline KalmanVeloStateWithQoP extrap_state(const KalmanVeloStateWithQoP& s, float z)
{
  KalmanVeloStateWithQoP out(s);
  float dz = z - out.z();
  out.x() += out.tx() * dz;
  out.y() += out.ty() * dz;
  out.z() += dz;
  float dz2 = dz * dz;
  out.c00() += dz2 * out.c22() + 2.f * dz * out.c20();
  out.c11() += dz2 * out.c33() + 2.f * dz * out.c31();
  out.c20() += out.c22() * dz;
  out.c31() += out.c33() * dz;
  return out;
}

/** Find the state closest to the given z. */
inline KalmanVeloStateWithQoP closest_state(std::vector<KalmanVeloStateWithQoP>& states, float z)
{
  return *std::min_element(
    states.begin(), states.end(), [z](auto& a, auto& b) { return std::abs(a.z() - z) < std::abs(b.z() - z); });
}

/** Extrapolate the closest state to z. */
inline KalmanVeloStateWithQoP extrap_from_closest_state(std::vector<KalmanVeloStateWithQoP>& states, float z)
{
  return extrap_state(closest_state(states, z), z);
}

namespace {
  using OutTracks = LHCb::Event::v3::Tracks;
  namespace OutTag = LHCb::Event::v3::Tag;
  using SL = OutTracks::StateLocation;

  template<SL L>
  float z_of(const KalmanVeloStateWithQoP& beamline_state, float first_meas_z, float last_meas_z)
  {
    if constexpr (L == SL::ClosestToBeam) return beamline_state.z();
    if constexpr (L == SL::FirstMeasurement) return first_meas_z;
    if constexpr (L == SL::LastMeasurement) return last_meas_z;
    if constexpr (L == SL::BegRich1) return StateParameters::ZBegRich1;
    if constexpr (L == SL::EndRich1) return StateParameters::ZEndRich1;
    if constexpr (L == SL::BegRich2) return StateParameters::ZBegRich2;
    if constexpr (L == SL::EndRich2) return StateParameters::ZEndRich2;
    return 0.f;
  }

  template<SL L, typename TrackProxy>
  void update_state(TrackProxy& outTrack, const KalmanVeloStateWithQoP& state)
  {
    outTrack.template field<OutTag::States>()[outTrack.state_index(L)].setPosition(state.x(), state.y(), state.z());
    outTrack.template field<OutTag::States>()[outTrack.state_index(L)].setDirection(state.tx(), state.ty());
    outTrack.template field<OutTag::States>()[outTrack.state_index(L)].setQOverP(state.qop());
    outTrack.template field<OutTag::StateCovs>()[outTrack.state_index(L)].setXCovariance(
      state.c00(), 0.f, state.c20(), 0.f, 0.f);
    outTrack.template field<OutTag::StateCovs>()[outTrack.state_index(L)].setYCovariance(
      state.c11(), 0.f, state.c31(), 0.f);
    outTrack.template field<OutTag::StateCovs>()[outTrack.state_index(L)].setTXCovariance(state.c22(), 0.f, 0.f);
    outTrack.template field<OutTag::StateCovs>()[outTrack.state_index(L)].setTYCovariance(state.c33(), 0.f);
    outTrack.template field<OutTag::StateCovs>()[outTrack.state_index(L)].setQoverPCovariance(state.c44());
  }

  template<SL... Ls, typename TrackProxy>
  void update_all_states(
    TrackProxy& outTrack,
    std::vector<KalmanVeloStateWithQoP>& states,
    float first_z,
    float last_z,
    LHCb::Event::v3::state_collection<Ls...>)
  {
    ((update_state<Ls>(outTrack, extrap_from_closest_state(states, z_of<Ls>(states[0], first_z, last_z)))), ...);
  }
} // namespace
