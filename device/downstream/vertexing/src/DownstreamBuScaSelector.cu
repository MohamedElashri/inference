/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration          *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "DownstreamBuScaSelector.cuh"

INSTANTIATE_ALGORITHM(downstream_busca_selector::downstream_busca_selector_t)

void downstream_busca_selector::downstream_busca_selector_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_downstream_mva_busca_t>(arguments, first<host_number_of_downstream_secondary_vertices_t>(arguments));
}

void downstream_busca_selector::downstream_busca_selector_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_downstream_mva_busca_t>(arguments, 0, context);

  global_function(downstream_busca_selector)(dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
    arguments, m_busca_selector.getDevicePointer());
}

__global__ void downstream_busca_selector::downstream_busca_selector(
  downstream_busca_selector::Parameters parameters,
  const DownstreamBuscaSelector::DeviceType* dev_downstream_busca_selector)
{
  // Basic
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  // Fetch input
  const auto downstream_composites = parameters.dev_multi_event_composites_view->container(event_number);
  const unsigned num_composites = downstream_composites.size();

  // Output
  auto downstream_busca_selector = parameters.dev_downstream_mva_busca + downstream_composites.offset();

  for (unsigned composite_idx = threadIdx.x; composite_idx < num_composites; composite_idx += blockDim.x) {

    // Fetch composite info
    const auto downstream_composite = downstream_composites.particle(composite_idx);
    const auto downstream_vertex = downstream_composite.vertex();

    // Fetch daughters
    const auto dA = static_cast<const Allen::Views::Physics::BasicParticle*>(downstream_composite.child(0));
    const auto dB = static_cast<const Allen::Views::Physics::BasicParticle*>(downstream_composite.child(1));

    // Daughters info. The downstream tracking builds the composites with an
    // atomicAdd-based compaction, so the (non-deterministic) track order decides
    // which daughter is child(0) and which is child(1). The BuSca NN is not
    // symmetric under exchanging the two daughters, so canonicalise their
    // order here: decreasing IP, then increasing chi2 for equal IP. The
    // (IP, chi2) pair is moved as a unit so the NN never mixes features from
    // different daughters. The comparison avoids `==` on floats on purpose.
    auto dA_ip = dA->ownpv_ip();
    auto dB_ip = dB->ownpv_ip();
    auto dA_chi2 = dA->chi2();
    auto dB_chi2 = dB->chi2();

    if (dA_ip < dB_ip || (!(dB_ip < dA_ip) && dA_chi2 > dB_chi2)) {
      const auto tmp_ip = dA_ip;
      const auto tmp_chi2 = dA_chi2;
      dA_ip = dB_ip;
      dA_chi2 = dB_chi2;
      dB_ip = tmp_ip;
      dB_chi2 = tmp_chi2;
    }

    const auto quality = downstream_composite.vertex().downstream_quality();

    // NN inputs
    const auto doca = downstream_vertex.downstream_doca();
    const auto min_ip = downstream_composite.minip();

    const auto ip = downstream_composite.ownpv_ip();
    // Compute scores
    float inputs_busca[DownstreamBuscaSelector::DeviceType::nInput] = {
      logf(dA_ip), logf(dA_chi2), logf(dB_ip), logf(dB_chi2), doca, logf(quality), logf(ip), logf(min_ip)};

    downstream_busca_selector[composite_idx] = dev_downstream_busca_selector->evaluate(inputs_busca);
  }

  __syncthreads();
}
