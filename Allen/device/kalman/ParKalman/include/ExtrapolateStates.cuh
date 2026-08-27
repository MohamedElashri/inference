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

#include "AlgorithmTypes.cuh"
#include "States.cuh"
#include "ParabolicExtrapolator.cuh"
#include "RungeKuttaExtrapolator.cuh"

namespace extrapolate_states {
  struct Parameters {
    HOST_INPUT(host_number_of_input_states_t, unsigned) host_number_of_input_states;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_kalman_states_view_t, Allen::Views::Physics::KalmanStates) dev_kalman_states_view;
    DEVICE_OUTPUT(dev_states_t, Extrapolators::State) dev_states;
  };

  struct extrapolate_states_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Allen::Property<float> m_step_dz {this, "step_dz", 100.0, "size of a step in z (in mm)"};
    Allen::Property<unsigned> m_n_steps {this, "n_steps", 100, "number of steps"};
  };
} // namespace extrapolate_states
