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

#include <BackendCommon.h>
#include <ButcherTableau.cuh>
#include <ExtrapolatorCommon.cuh>
#include <MagneticField.cuh>

namespace Extrapolators {
  template<typename ftype = float, typename Table = ButcherTableau::CashKarp<ftype>>
  struct RungeKuttaExtrapolator {
    // Implementation taken from:
    // https://gitlab.cern.ch/lhcb/Rec/-/blob/master/Tr/TrackExtrapolators/src/TrackRungeKuttaExtrapolator.cpp

    __device__ static void propagate(State& state, State::Error& err, ftype dz, const MagneticField::Magfield& field)
    {
      State::Derivative k[Table::N_stages];
      UNROLL(10)
      for (int stage = 0; stage < Table::N_stages; stage++) {
        State s = state;
        UNROLL(10)
        for (int i = 0; i < stage - 1; i++) {
          s = s + k[i] * Table::a(stage, i);
        }
        float3 B = field.fieldVectorLinearInterpolation(make_float3(s.x, s.y, s.z));
        k[stage] = derivative(state, B) * dz;
      }

      err.clear();
      UNROLL(10)
      for (int i = 0; i < Table::N_stages; i++) {
        err = err + k[i] * (Table::b(i) - Table::b_star(i));
        state = state + k[i] * Table::b(i);
      }
    }
  };
} // namespace Extrapolators
