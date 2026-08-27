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
#pragma once

#include "BackendCommon.h"
#include "States.cuh"
#include "Rich.cuh"
#include "Common.h"
#include "RichDefinitions.cuh"

namespace Allen::Rich::PhotonReco {
  struct RotationMatrix {
    // Rotation matrix (transforms TO track reference frame)
    float m_00 {}, m_01 {}, m_02 {};
    float m_10 {}, m_11 {}, m_12 {};
    float m_20 {}, m_21 {}, m_22 {};

    /**
     * @brief Build rotation matrix from momentum vector
     *
     * Creates coordinate system where:
     * - Z-axis: along track momentum
     * - Y-axis: perpendicular to track and global X-axis
     * - X-axis: completes right-handed system
     *
     * @param momentum Track momentum vector (px, py, pz)
     */
    __device__ RotationMatrix(const float3& p)
    {
      // Get momentum magnitude
      const float p_mag = sqrtf(p.x * p.x + p.y * p.y + p.z * p.z);

      if (p_mag < 1e-10f) {
        // Identity matrix if no momentum
        m_00 = 1.0f;
        m_01 = 0.0f;
        m_02 = 0.0f;
        m_10 = 0.0f;
        m_11 = 1.0f;
        m_12 = 0.0f;
        m_20 = 0.0f;
        m_21 = 0.0f;
        m_22 = 1.0f;
        return;
      }

      // Z-axis: normalized momentum direction
      const float z_x = p.x / p_mag;
      const float z_y = p.y / p_mag;
      const float z_z = p.z / p_mag;

      // Y-axis: z × (1, 0, 0) = (0, z.z, -z.y)
      float y_x = 0.0f;
      float y_y = z_z;
      float y_z = -z_y;

      // Normalize y
      const float y_mag = sqrtf(y_y * y_y + y_z * y_z);

      if (y_mag < 1e-10f) {
        // Track is along x-axis, use z × (0, 1, 0) instead
        y_x = -z_z;
        y_y = 0.0f;
        y_z = z_x;
        const float y_mag2 = sqrtf(y_x * y_x + y_z * y_z);
        if (y_mag2 > 1e-10f) {
          y_x /= y_mag2;
          y_z /= y_mag2;
        }
      }
      else {
        y_y /= y_mag;
        y_z /= y_mag;
      }

      // X-axis: y × z
      const float x_x = y_y * z_z - y_z * z_y;
      const float x_y = y_z * z_x - y_x * z_z;
      const float x_z = y_x * z_y - y_y * z_x;

      // Build rotation matrix (transforms TO track frame)
      m_00 = x_x;
      m_01 = x_y;
      m_02 = x_z;
      m_10 = y_x;
      m_11 = y_y;
      m_12 = y_z;
      m_20 = z_x;
      m_21 = z_y;
      m_22 = z_z;
    }

    __device__ float3 globalToTrack(const float3& direction) const
    {
      return {
        m_00 * direction.x + m_01 * direction.y + m_02 * direction.z,
        m_10 * direction.x + m_11 * direction.y + m_12 * direction.z,
        m_20 * direction.x + m_21 * direction.y + m_22 * direction.z};
    }

    /**
     * @brief Calculate Cherenkov angles for a photon direction
     *
     * @param direction Photon direction vector in global frame
     * @param theta Output: Cherenkov angle (angle between track and photon)
     * @param phi Output: Azimuthal angle around track direction
     */
    __device__ void angleToDirection(const Allen::Rich::Vector& direction, float& theta, float& phi) const
    {
      // Transform direction to track reference frame
      const float3 r = globalToTrack(direction);

      // Compute theta and phi from vector components in track frame
      const float perp = sqrtf((r.x * r.x) + (r.y * r.y));
      theta = atan2f(perp, r.z);
      phi = atan2f(r.y, r.x);

      // Correct phi to range 0 - 2PI
      constexpr float twopi = 2.0f * Allen::constants::pi_f_float;
      if (phi < 0.0f) {
        phi += twopi;
      }
    }

    __device__ float3 vectorAtThetaPhi(const float theta, const float phi) const
    {
      float sinTheta, cosTheta, sinPhi, cosPhi;
      __sincosf(theta, &sinTheta, &cosTheta);
      __sincosf(phi, &sinPhi, &cosPhi);

      float3 direction {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
      return {
        m_00 * direction.x + m_10 * direction.y + m_20 * direction.z,
        m_01 * direction.x + m_11 * direction.y + m_21 * direction.z,
        m_02 * direction.x + m_12 * direction.y + m_22 * direction.z};
    }
  };
} // namespace Allen::Rich::PhotonReco
