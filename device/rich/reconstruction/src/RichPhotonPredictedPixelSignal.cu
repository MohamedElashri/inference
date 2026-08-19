/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#include "RichPhotonPredictedPixelSignal.cuh"
#include <cmath>
#include <BinarySearch.cuh>

INSTANTIATE_ALGORITHM(rich_photon_predicted_pixel_signal::rich_photon_predicted_pixel_signal_t);

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_photon_predicted_pixel_signal::rich_photon_predicted_pixel_signal_t::updateRich(
  const Allen::Rich::RichDetector<richIdx>* rich) const
{
  const double R = static_cast<double>(rich->sphMirrorRadius());
  m_factor = static_cast<float>(4.0 / (R * R * std::pow(2.0 * M_PI, 1.5)));
}

void rich_photon_predicted_pixel_signal::rich_photon_predicted_pixel_signal_t::update(const Constants& constants) const
{
  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());
  if (rich == Allen::Rich::Detector::Rich1) {
    updateRich<Allen::Rich::Detector::Rich1>(constants.host_rich_1_geometry);
  }
  else {
    updateRich<Allen::Rich::Detector::Rich2>(constants.host_rich_2_geometry);
  }
}

template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_photon_predicted_pixel_signal_k(
  const Allen::Rich::RichDetector<richIdx>* rich,
  const unsigned number_of_events,
  const unsigned number_of_tracks,
  const unsigned number_of_photons,
  const unsigned* pixels_offsets,
  const Allen::Rich::ParticleHypos* hypos,
  const unsigned* photons_offsets,
  const Allen::Rich::PhotonReco::Photon* photons,
  Allen::Rich::HypoData<float>* photon_pix_signals,
  const float factor,
  const float minExpCKT,
  const float minPhotonProb)
{
  const unsigned number_of_pds =
    Allen::Rich::NPDPanelsPerRICH * number_of_events * Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned photon_id = threadId; photon_id < number_of_photons; photon_id += stride) {
    const unsigned track_id = binary_search_rightmost(photons_offsets, number_of_tracks + 1, photon_id);
    const unsigned pix_id = photons[photon_id].pixelIdx;
    const float theta = photons[photon_id].ckTheta;

    unsigned pd_id = binary_search_rightmost(pixels_offsets, number_of_pds + 1, pix_id);
    const auto side = static_cast<Allen::Rich::Detector::Side>(pd_id / (number_of_pds / Allen::Rich::NPDPanelsPerRICH));
    pd_id = pd_id % Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;

    const float A = rich->pdPanels()[side].pds()[pd_id].effectivePixelArea();
    const bool validTheta = std::isfinite(theta) && theta > 0.f;
    const float hypo_indep = validTheta ? A * factor / theta : 0.f;

    UNROLL(Allen::Rich::NParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      const float expTheta = hypos[track_id].ckTheta[hypo];
      float sig = 0.f;
      if (validTheta && !std::isnan(expTheta) && expTheta > minExpCKT) {
        const float res = hypos[track_id].ckRes[hypo];
        const float yield = hypos[track_id].yield[hypo];
        const float resInv = 1.f / res;

        // aij = yield * 1/((2pi)^(3/2)*sigma(theta)) * exp(-1/2 * sep^2) * 4A/(R^2 theta)
        const float sep = (theta - expTheta) * resInv;
        const float expArg = expf(-0.5f * sep * sep);
        // const float expArg = Allen::Rich::Maths::fast_exp(-0.5f * sep * sep);

        sig = hypo_indep * expArg * yield * resInv;
        if (sig <= minPhotonProb) sig = 0.f;
      }

      photon_pix_signals[photon_id][hypo] = sig;
    }
  }
}

void rich_photon_predicted_pixel_signal::rich_photon_predicted_pixel_signal_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_photons = first<host_number_of_photons_t>(arguments);
  set_size<dev_photon_pix_signals_t>(arguments, number_of_photons);
}

template<Allen::Rich::Detector::DetectorType richIdx>
void rich_photon_predicted_pixel_signal::rich_photon_predicted_pixel_signal_t::launchForRich(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context,
  const Allen::Rich::RichDetector<richIdx>* rich) const
{
  const unsigned number_of_events = first<host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  const unsigned number_of_photons = first<host_number_of_photons_t>(arguments);

  const auto& sig_kernel = rich_photon_predicted_pixel_signal_k<richIdx>;
  global_function(sig_kernel)(dim3(32), m_block_dim, context)(
    rich,
    number_of_events,
    number_of_tracks,
    number_of_photons,
    data<dev_rich_pd_offsets_t>(arguments),
    data<dev_rich_hypos_t>(arguments),
    data<dev_offsets_rich_photons_t>(arguments),
    data<dev_rich_photons_t>(arguments),
    data<dev_photon_pix_signals_t>(arguments),
    m_factor,
    m_minExpCKT.value()[richIdx],
    m_minPhotonProb.value()[richIdx]);
}

void rich_photon_predicted_pixel_signal::rich_photon_predicted_pixel_signal_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& options,
  const Constants& constants,
  const Allen::Context& context) const
{
  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());
  if (rich == Allen::Rich::Detector::Rich1) {
    launchForRich<Allen::Rich::Detector::Rich1>(arguments, options, constants, context, constants.dev_rich_1_geometry);
  }
  else {
    launchForRich<Allen::Rich::Detector::Rich2>(arguments, options, constants, context, constants.dev_rich_2_geometry);
  }
}
