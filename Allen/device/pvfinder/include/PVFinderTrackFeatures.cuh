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

// Per-track PVFinder input features from a VELO Kalman state, computed by
// the FC aggregation's CSR build (pvfinder_fc_aggregation). Uses dev_beamline
// (loaded by updateCommon, see BeamlinePVConstants.cuh).

#include "KalmanParametrizations.cuh"
#include "BeamlinePVConstants.cuh"
#include "ParticleTypes.cuh"

namespace pvfinder_track_features {

  __device__ inline float3 normalize(float3 v)
  {
    float mag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (mag > 0.0f) {
      return make_float3(v.x / mag, v.y / mag, v.z / mag);
    }
    else {
      return make_float3(0.0f, 0.0f, 0.0f);
    }
  }

  __device__ inline float3 cross(float3 a, float3 b)
  {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  }

  __device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

  // ---------------------------------------------------------------------------
  // Per-track features in the convention of the training data (pv-finder
  // t2hists arrays, rebuilt and checked against pv-finder_v2's
  // tools/ellipsoid.py and tools/split_data_intervals.py):
  //
  //   [0] poca_x, [1] poca_y  transverse POCA, relative to the beamline
  //   [2] poca_z              absolute; the FC stage replaces it by z relative
  //                           to each interval's lower edge
  //   [3..8] A..F             ellipsoid A x^2 + B y^2 + C z^2 + 2(D xy + E xz + F yz) = 1
  //                           of the POCA uncertainty: the INVERSE covariance,
  //                           sum over the three axes of u u^T / |u|^4
  //
  // The ellipsoid axes, as in the training ntuples: two minor axes of length
  // road_error perpendicular to the track (one along beam x track, one in the
  // plane of beam and track), and a major axis along the track of length
  // road_error * cos(theta) / sin(theta), theta being the angle between track
  // and beam axis. The ratio is capped at 2048, the largest value in the
  // training sample (tracks nearly parallel to the beam).
  // ---------------------------------------------------------------------------
  __device__ inline bool state_poca(
    const Allen::Views::Physics::KalmanState& state,
    float& poca_x,
    float& poca_y,
    float& poca_z,
    float& tx,
    float& ty)
  {
    // Work in the beamline frame: the training sample's beam sat on the z
    // axis, the 2024 conditions' does not.
    const float bx = dev_beamline.pos.x + dev_beamline.tx.x * (state.z() - dev_beamline.pos.z);
    const float by = dev_beamline.pos.y + dev_beamline.tx.y * (state.z() - dev_beamline.pos.z);
    tx = state.tx() - dev_beamline.tx.x;
    ty = state.ty() - dev_beamline.tx.y;
    const float x0 = (state.x() - bx) - state.z() * tx; // intercept at z = 0
    const float y0 = (state.y() - by) - state.z() * ty;
    const float t_sq = tx * tx + ty * ty;
    poca_z = t_sq > 1e-8f ? -(x0 * tx + y0 * ty) / t_sq : 0.0f; // parallel to the beam: take z = 0
    poca_x = x0 + tx * poca_z;
    poca_y = y0 + ty * poca_z;
    return sqrtf(poca_x * poca_x + poca_y * poca_y) < 1000.f;
  }

  __device__ inline void calculate_ellipsoid_params(
    const Allen::Views::Physics::KalmanState& state,
    float* ellipsoid_params,
    float& poca_x,
    float& poca_y,
    float& poca_z)
  {
    float tx, ty;
    if (!state_poca(state, poca_x, poca_y, poca_z, tx, ty)) {
      poca_x = poca_y = poca_z = 0.0f;
      for (int i = 0; i < 6; ++i)
        ellipsoid_params[i] = 0.0f; // fails the FC track selection
      return;
    }
    const float3 track_dir = normalize(make_float3(tx, ty, 1.0f));
    const float sin_t = sqrtf(track_dir.x * track_dir.x + track_dir.y * track_dir.y);
    const float cos_t = track_dir.z;
    // Unit axes: e1 = beam x track, e2 = track x e1 (in the beam-track plane), e3 = track.
    const float3 e1 =
      sin_t > 0.0f ? make_float3(-track_dir.y / sin_t, track_dir.x / sin_t, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    const float3 e2 = cross(track_dir, e1);
    const float3 e3 = track_dir;

    // A non-positive x variance (about 0.03% of the VELO Kalman states in 2024
    // minimum bias) has no ellipsoid: zeros, which fail the FC track selection.
    if (!(state.c00() > 0.0f)) {
      for (int i = 0; i < 6; ++i)
        ellipsoid_params[i] = 0.0f;
      return;
    }
    const float road_error = sqrtf(state.c00());
    const float ratio = sin_t > 0.0f ? fminf(cos_t / sin_t, 2048.0f) : 2048.0f;
    const float w12 = 1.0f / (road_error * road_error); // 1 / |minor axis|^2
    const float w3 = w12 / (ratio * ratio);             // 1 / |major axis|^2

    ellipsoid_params[0] = w12 * (e1.x * e1.x + e2.x * e2.x) + w3 * e3.x * e3.x; // A
    ellipsoid_params[1] = w12 * (e1.y * e1.y + e2.y * e2.y) + w3 * e3.y * e3.y; // B
    ellipsoid_params[2] = w12 * (e1.z * e1.z + e2.z * e2.z) + w3 * e3.z * e3.z; // C
    ellipsoid_params[3] = w12 * (e1.x * e1.y + e2.x * e2.y) + w3 * e3.x * e3.y; // D
    ellipsoid_params[4] = w12 * (e1.x * e1.z + e2.x * e2.z) + w3 * e3.x * e3.z; // E
    ellipsoid_params[5] = w12 * (e1.y * e1.z + e2.y * e2.z) + w3 * e3.y * e3.z; // F
  }

  // The 9 features, stored as (x, y, z, A..F); the FC stage feeds the network
  // (z - interval edge, x, y, A..F), the training order.
  __device__ inline void compute(const Allen::Views::Physics::KalmanState& state, float* f)
  {
    float ellipsoid_params[6];
    float poca_x, poca_y, poca_z;
    calculate_ellipsoid_params(state, ellipsoid_params, poca_x, poca_y, poca_z);
    f[0] = poca_x;
    f[1] = poca_y;
    f[2] = poca_z;
    for (int i = 0; i < 6; ++i)
      f[3 + i] = ellipsoid_params[i]; // A..F
  }

} // namespace pvfinder_track_features
