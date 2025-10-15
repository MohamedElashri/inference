/*****************************************************************************\
* (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <ExtrapolatorCommon.cuh>
#include <MagneticField.cuh>

namespace Extrapolators {
  template<typename ftype = float>
  struct ParabolicExtrapolator {
    // Implementation taken from:
    // https://gitlab.cern.ch/lhcb/Rec/-/blob/master/Tr/TrackExtrapolators/src/TrackParabolicExtrapolator.cpp

    __device__ static void propagate(State& state, ftype dz, const MagneticField::Magfield& field)
    {
      // Get the B field at midpoint
      const ftype xMid = state.x + state.tx * dz * ftype {0.5};
      const ftype yMid = state.y + state.ty * dz * ftype {0.5};
      const ftype zMid = state.z + dz * ftype {0.5};
      const float3 P = make_float3(xMid, yMid, zMid);
      const float3 B = field.fieldVectorLinearInterpolation(P);

      // Update the state parameters (exact extrapolation)
      const auto d = derivative(state, B);
      state.x += dz * (d.dx + ftype {0.5} * dz * d.dtx);
      state.y += dz * (d.dy + ftype {0.5} * dz * d.dty);
      state.z += dz;
      state.tx += dz * d.dtx;
      state.ty += dz * d.dty;
    }
  };
} // namespace Extrapolators