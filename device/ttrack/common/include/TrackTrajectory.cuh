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
#pragma once

#include "BackendCommon.h"
#include "States.cuh"
#include "SystemOfUnits.h"
// #include "ITrackTrajectory.cuh"
#include "LinearAlgebra.cuh"
#include "BinarySearch.cuh"

#include "TTrackDefinitions.cuh"
#include "FloatOperations.cuh"

namespace Trajectories {

  namespace Constants = TTracks::Constants;

  template<typename T>
  concept FloatElement = std::same_as<T, float> || std::same_as<T, const float>;

  template<typename T>
  concept MutableView = not std::is_const_v<T>;

  enum StateVar : unsigned { x = 0, y = 1, tx = 2, ty = 3, COUNT = 4 };

  struct TrajectoryExpansion {
    float3 p, dpdmu, d2pdmu2;
    float distTo1stError, distTo2ndError;
  };

  template<typename MemoryManager, typename T = float>
  class TrajectoryProxy {
  public:
    __device__ TrajectoryProxy(MemoryManager manager, unsigned trackIdx) : m_manager(manager), m_track_idx(trackIdx) {}

    template<StateVar var>
    __device__ inline float get(unsigned state_idx) const
    {
      return m_manager.template get<var>(m_track_idx, state_idx);
    }

    __device__ inline float3 get_p(unsigned state_idx) const { return m_manager.get_p(m_track_idx, state_idx); }

    __device__ inline float3 get_dpdmu(unsigned state_idx) const { return m_manager.get_dpdmu(m_track_idx, state_idx); }

  private:
    MemoryManager m_manager;
    unsigned m_track_idx;
  };

  template<FloatElement T>
  class GlobalMemoryManagerT {
  public:
    __device__ GlobalMemoryManagerT(T* base_ptr, unsigned num_of_trajectories) :
      base_pointer(base_ptr), m_num_of_trajectories(num_of_trajectories)
    {}

    template<StateVar var>
    __device__ inline void set(unsigned track_idx, unsigned state_idx, float value) const requires(MutableView<T>)
    {
      base_pointer[index<var>(track_idx, state_idx)] = value;
    }

    template<StateVar var>
    __device__ inline float get(unsigned track_idx, unsigned state_idx) const
    {
      return base_pointer[index<var>(track_idx, state_idx)];
    }

    template<typename State>
    __device__ inline void set_state(unsigned track_idx, unsigned state_idx, State value) const requires(MutableView<T>)
    {
      set<StateVar::x>(track_idx, state_idx, value.x);
      set<StateVar::y>(track_idx, state_idx, value.y);
      set<StateVar::tx>(track_idx, state_idx, value.tx);
      set<StateVar::ty>(track_idx, state_idx, value.ty);
    }

    __device__ inline float3 get_p(unsigned track_idx, unsigned state_idx) const
    {
      return make_float3(
        get<StateVar::x>(track_idx, state_idx),
        get<StateVar::y>(track_idx, state_idx),
        Constants::z_reference_states[state_idx]);
    }

    __device__ inline float3 get_dpdmu(unsigned track_idx, unsigned state_idx) const
    {
      return make_float3(get<StateVar::tx>(track_idx, state_idx), get<StateVar::ty>(track_idx, state_idx), 1.f);
    }

    __device__ inline auto get_proxy(unsigned track_idx) const
    {
      return TrajectoryProxy<GlobalMemoryManagerT<T>, T>(*this, track_idx);
    }

  private:
    template<StateVar var>
    __device__ inline unsigned index(unsigned track_idx, unsigned state_idx) const
    {
      assert(var < StateVar::COUNT);
      assert(state_idx < Constants::num_reference_states);
      assert(track_idx < m_num_of_trajectories);
      return track_idx + (state_idx * StateVar::COUNT + var) * m_num_of_trajectories;
    }

    T* base_pointer;
    unsigned m_num_of_trajectories;
  };

  using GlobalMemoryManager = GlobalMemoryManagerT<float>;
  using GlobalMemoryManagerConst = GlobalMemoryManagerT<const float>;

  template<typename Proxy>
  class Spline {

  public:
    __device__ Spline(const Proxy& proxy) : m_proxy(proxy) {}

    __device__ MiniState state(float z)
    {
      if (const short edge_state = get_edge_state(z); edge_state >= 0) {
        const auto p = m_proxy.get_p(edge_state);
        const auto dpdmu = m_proxy.get_dpdmu(edge_state);
        const auto dz = (z - Constants::z_reference_states[edge_state]);
        return MiniState(p.x + dpdmu.x * dz, p.y + dpdmu.y * dz, z, dpdmu.x, dpdmu.y);
      }

      update_cache(z);
      return MiniState(get_x(z), get_y(z), z, get_tx(z), get_ty(z));
    }

    __device__ inline float3 position(float z)
    {
      const auto st = state(z);
      return make_float3(st.x(), st.y(), st.z());
    }

    __device__ inline TrajectoryExpansion expansion(float z, float tolerance, float maxDistToErrors)
    {
      if (const short edge_state = get_edge_state(z); edge_state >= 0) {
        const auto p = m_proxy.get_p(edge_state);
        const auto dpdmu = m_proxy.get_dpdmu(edge_state);
        const auto dz = (z - Constants::z_reference_states[edge_state]);
        return TrajectoryExpansion {.p = p + dpdmu * dz,
                                    .dpdmu = dpdmu,
                                    .d2pdmu2 = make_float3(0.0f, 0.0f, 0.0f),
                                    .distTo1stError = maxDistToErrors,
                                    .distTo2ndError = maxDistToErrors};
      }

      update_cache(z);
      const auto tolerance_pair = distanceToErrors(z, tolerance);

      return TrajectoryExpansion {.p = make_float3(get_x(z), get_y(z), z),
                                  .dpdmu = make_float3(get_tx(z), get_ty(z), 1.f),
                                  .d2pdmu2 = make_float3(get_omegax(z), get_omegay(z), 0.f),
                                  .distTo1stError = tolerance_pair[0],
                                  .distTo2ndError = tolerance_pair[1]};
    }

  private:
    template<StateVar varX, StateVar varTx>
    __device__ inline void update_spline(float* target, unsigned short stateL)
    {
      target[0] = m_proxy.template get<varX>(stateL);
      target[1] = m_proxy.template get<varX>(stateL + 1) - target[0];
      target[2] = m_proxy.template get<varTx>(stateL);
      target[3] = m_proxy.template get<varTx>(stateL + 1);
    }

    __device__ inline void update_cache(float z)
    {
      const unsigned stateL = getLeftClosestRefStateIdx(z);
      if (m_stateL != stateL) {
        m_stateL = stateL;
        update_spline<StateVar::x, StateVar::tx>(m_x, stateL);
        update_spline<StateVar::y, StateVar::ty>(m_y, stateL);
      }
    }

    __device__ inline short get_edge_state(float z)
    {
      if (z < Constants::z_reference_states[0])
        return 0;
      else if (z > Constants::z_reference_states[Constants::num_reference_states - 1])
        return Constants::num_reference_states - 1;
      else
        return -1;
    }

    __device__ inline unsigned getLeftClosestRefStateIdx(float z)
    {
      UNROLL(Constants::num_reference_states - 1)
      for (unsigned i = Constants::num_reference_states - 2; i >= 1; --i) {
        if (z > Constants::z_reference_states[i]) {
          return i;
        }
      }
      return 0;
    }

    __device__ inline float get_x(float target_z) { return polyeval(m_x, target_z); }

    __device__ inline float get_tx(float target_z) { return poly1stderiveval(m_x, target_z); }

    __device__ inline float get_omegax(float target_z) { return poly2ndderiveval(m_x, target_z); }

    __device__ inline float get_y(float target_z) { return polyeval(m_y, target_z); }

    __device__ inline float get_ty(float target_z) { return poly1stderiveval(m_y, target_z); }

    __device__ inline float get_omegay(float target_z) { return poly2ndderiveval(m_y, target_z); }

    __device__ inline std::array<float, 2> distanceToErrors(float target_z, float tolerance)
    {
      const float d = Constants::z_reference_states[m_stateL + 1] - Constants::z_reference_states[m_stateL];
      const auto deriv = get_omegax(target_z);
      const auto c3 = (-2.f * m_x[1] + d * (m_x[2] + m_x[3])) / (d * d * d);
      const auto error2 = c3 != 0 ? cbrtf(fabsf(tolerance / c3)) : 10 * Allen::Units::km;
      return std::array<float, 2> {
        min(deriv != 0 ? sqrtf(fabsf(2 * tolerance / deriv)) : 10 * Allen::Units::km, error2), error2};
    }

    __device__ inline float polyeval(const float* pars, float target_z)
    {
      const float dz = target_z - Constants::z_reference_states[m_stateL];
      const float d = Constants::z_reference_states[m_stateL + 1] - Constants::z_reference_states[m_stateL];
      return dz * pars[2] + pars[0] + (dz * dz * dz * (d * (pars[2] + pars[3]) - 2 * pars[1])) / (d * d * d) -
             (dz * dz * (2 * d * pars[2] + d * pars[3] - 3 * pars[1])) / (d * d);
    }

    __device__ inline float poly1stderiveval(const float* pars, float target_z)
    {
      const float dz = target_z - Constants::z_reference_states[m_stateL];
      const float d = Constants::z_reference_states[m_stateL + 1] - Constants::z_reference_states[m_stateL];
      return pars[2] - (2 * dz * (2 * pars[2] + pars[3])) / d + (-6 * dz * dz * pars[1]) / (d * d * d) +
             (3 * dz * (dz * (pars[2] + pars[3]) + 2 * pars[1])) / (d * d);
    }

    __device__ inline float poly2ndderiveval(const float* pars, float target_z)
    {
      const float dz = target_z - Constants::z_reference_states[m_stateL];
      const float d = Constants::z_reference_states[m_stateL + 1] - Constants::z_reference_states[m_stateL];
      return (-2 *
              (d * d * (2 * pars[2] + pars[3]) + 6 * dz * pars[1] - 3 * d * (dz * (pars[2] + pars[3]) + pars[1]))) /
             (d * d * d);
    }

    Proxy m_proxy;
    unsigned short m_stateL = 0xFF;
    float m_x[4];
    float m_y[4];
  };

}; // namespace Trajectories
