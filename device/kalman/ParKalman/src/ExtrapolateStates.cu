/*****************************************************************************\
* (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "ExtrapolateStates.cuh"

INSTANTIATE_ALGORITHM(extrapolate_states::extrapolate_states_t)

void extrapolate_states::extrapolate_states_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  auto n_states = first<host_number_of_input_states_t>(arguments);
  set_size<dev_states_t>(arguments, n_states);
}

__global__ void extrapolate_states_kernel(
  extrapolate_states::Parameters parameters,
  float dz,
  unsigned n_steps,
  const MagneticField::Magfield field)
{

  const auto* input_states = parameters.dev_kalman_states_view.data();
  const unsigned n_states = input_states->total_number_of_states();

  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n_states; i += blockDim.x * gridDim.x) {
    const auto& input = input_states->state(i);

    Extrapolators::State output {input.x(), input.y(), input.z(), input.tx(), input.ty(), input.qop()};

    for (unsigned s = 0; s < n_steps; s++) {
      // Extrapolators::ParabolicExtrapolator<float>::propagate(output, dz, field);
      Extrapolators::State::Error err;
      Extrapolators::RungeKuttaExtrapolator<float, ButcherTableau::CashKarp<float>>::propagate(output, err, dz, field);

      /*if (threadIdx.x == 0 && blockIdx.x == 0) {
        float3 B = field.fieldVectorLinearInterpolation(make_float3(output.x, output.y, output.z));
        printf("[%g, %g, %g, %g, %g, %g], // Bx=%g By=%g Bz=%g\n",
        output.x, output.y, output.z, output.tx, output.ty, output.qop, B.x, B.y, B.z);
      }*/
    }

    parameters.dev_states[i] = output;
  }
}

void extrapolate_states::extrapolate_states_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  const int block_dim = 256;
  const int grid_dim = (first<host_number_of_input_states_t>(arguments) + block_dim - 1) / block_dim;
  global_function(extrapolate_states_kernel)(grid_dim, block_dim, context)(
    arguments, m_step_dz, m_n_steps, *constants.magnetic_field);
}
